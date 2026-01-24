import numpy as np
import torch
from ..models.lmdm import LMDM


"""
lmdm_cfg = {
    "model_path": "",
    "device": "cuda",
    "motion_feat_dim": 265,
    "audio_feat_dim": 1024+35,
    "seq_frames": 80,
}
"""


def _cvt_LP_motion_info(inp, mode, ignore_keys=()):
    ks_shape_map = [
        ['scale', (1, 1), 1], 
        ['pitch', (1, 66), 66],
        ['yaw',   (1, 66), 66],
        ['roll',  (1, 66), 66],
        ['t',     (1, 3), 3], 
        ['exp', (1, 63), 63],
        ['kp',  (1, 63), 63],
    ]
    
    def _dic2arr(_dic):
        arr = []
        for k, _, ds in ks_shape_map:
            if k not in _dic or k in ignore_keys:
                continue
            v = _dic[k].reshape(ds)
            if k == 'scale':
                v = v - 1
            arr.append(v)
        arr = np.concatenate(arr, -1)  # (133)
        return arr
    
    def _arr2dic(_arr):
        dic = {}
        s = 0
        for k, ds, ss in ks_shape_map:
            if k in ignore_keys:
                continue
            v = _arr[s:s + ss].reshape(ds)
            if k == 'scale':
                v = v + 1
            dic[k] = v
            s += ss
            if s >= len(_arr):
                break
        return dic
    
    if mode == 'dic2arr':
        assert isinstance(inp, dict)
        return _dic2arr(inp)   # (dim)
    elif mode == 'arr2dic':
        assert inp.shape[0] >= 265, f"{inp.shape}"
        return _arr2dic(inp)   # {k: (1, dim)}
    else:
        raise ValueError()
    

class Audio2Motion:
    def __init__(
        self,
        lmdm_cfg,
    ):
        self.lmdm = LMDM(**lmdm_cfg)

    def setup(
        self, 
        x_s_info, 
        overlap_v2=10,
        fix_kp_cond=0,
        fix_kp_cond_dim=None,
        sampling_timesteps=50,
        online_mode=False,
        v_min_max_for_clip=None,
        smo_k_d=3,
        vad_timeline=None,
    ):
        self.smo_k_d = smo_k_d
        self.vad_timeline = vad_timeline
        self.overlap_v2 = overlap_v2
        self.seq_frames = self.lmdm.seq_frames
        self.valid_clip_len = self.seq_frames - self.overlap_v2

        # for fuse
        self.online_mode = online_mode
        if self.online_mode:
            self.fuse_length = min(self.overlap_v2, self.valid_clip_len)
        else:
            self.fuse_length = self.overlap_v2
        self.fuse_alpha = np.arange(self.fuse_length, dtype=np.float32).reshape(1, -1, 1) / self.fuse_length

        self.fix_kp_cond = fix_kp_cond
        self.fix_kp_cond_dim = fix_kp_cond_dim
        self.sampling_timesteps = sampling_timesteps
        
        self.v_min_max_for_clip = v_min_max_for_clip
        if self.v_min_max_for_clip is not None:
            self.v_min = self.v_min_max_for_clip[0][None]    # [dim, 1]
            self.v_max = self.v_min_max_for_clip[1][None]

        kp_source = _cvt_LP_motion_info(x_s_info, mode='dic2arr', ignore_keys={'kp'})[None]
        self.s_kp_cond = kp_source.copy().reshape(1, -1)
        self.kp_cond = self.s_kp_cond.copy()

        self.lmdm.setup(sampling_timesteps)

        self.clip_idx = 0
        self.warp_offset = None
        self.brownian_pos = np.zeros_like(self.kp_cond)
        self.last_kp_frame = None # [v1.9.107] Persistence bridging
        self.global_time = 0 # [v1.9.107] For procedural breathing
        self.silence_frames = 0 # [v1.9.139] Track silence duration for adaptive boost
        self.brownian_momentum = np.zeros_like(self.kp_cond) # [v1.9.139] Postural inertia
        self.look_up_timer = 0 # [v1.9.141] Timer for anti-stall recovery
        self.is_recovering = False # [v1.9.155] Hysteresis state flag
        self.target_bias_deg = 0.0 # [v1.9.162/163] For scope visibility
        
        # [v1.9.165] Mathematical Posture Initializers
        self.photo_base_neutralizer = np.zeros_like(self.s_kp_cond)
        self.photo_base_neutralizer[0, 1:202] = -self.s_kp_cond[0, 1:202]
        self.current_neutralizer = self.photo_base_neutralizer.copy()
        self.target_neutralizer = self.photo_base_neutralizer.copy()
        
        # [v1.9.146] Capture Source Pitch Baseline
        # We need the "Original" degree of the source photo to detect relative stalls.
        s_pitch_bins = self.s_kp_cond[0, 1:67]
        e_s = np.exp(s_pitch_bins - np.max(s_pitch_bins))
        p_s = e_s / e_s.sum()
        self.s_pitch_deg = np.sum(p_s * np.arange(66)) * 3 - 97.5
        print(f"[Postural Setup] Source Pitch Base: {self.s_pitch_deg:.2f}°")

    def _fuse(self, res_kp_seq, pred_kp_seq, override_alpha=None, step_len=None):
        ## ========================
        ## offline fuse mode
        ## last clip:  -------
        ## fuse part:    *****
        ## curr clip:    -------
        ## output:       ^^
        #
        ## online fuse mode
        ## last clip:  -------
        ## fuse part:       **
        ## curr clip:    -------
        ## output:          ^^
        ## ========================

        if step_len is None:
            step_len = self.valid_clip_len

        fuse_r1_s = res_kp_seq.shape[1] - self.fuse_length
        fuse_r1_e = res_kp_seq.shape[1]
        
        # Calculate fuse range based on step_len
        fuse_r2_s = self.seq_frames - step_len - self.fuse_length
        fuse_r2_e = self.seq_frames - step_len

        r1 = res_kp_seq[:, fuse_r1_s:fuse_r1_e]     # [1, fuse_len, dim]
        r2 = pred_kp_seq[:, fuse_r2_s: fuse_r2_e]   # [1, fuse_len, dim]
        
        alpha = override_alpha if override_alpha is not None else self.fuse_alpha
        r_fuse = r1 * (1 - alpha) + r2 * alpha

        res_kp_seq[:, fuse_r1_s:fuse_r1_e] = r_fuse    # fuse last
        res_kp_seq = np.concatenate([res_kp_seq, pred_kp_seq[:, fuse_r2_e:]], 1)  # len(res_kp_seq) + valid_clip_len

        return res_kp_seq
    
    def _generate_gaussian_pitch(self, target_deg, sigma=1.5):
        # Ditto Pitch: 66 bins, 3 deg each, center at 32.5 (0 deg)
        # target_deg = (idx * 3) - 97.5 => idx = (target_deg + 97.5) / 3
        mu = (target_deg + 97.5) / 3.0
        x = np.arange(66)
        # We generate raw coefficients that will simulate the target degree
        gaussian = np.exp(-((x - mu)**2) / (2 * sigma**2))
        # Add tiny epsilon to avoid log(0) issues in model
        gaussian = np.clip(gaussian, 1e-6, 1.0)
        # Log-space coefficients for the categorical model
        return np.log(gaussian).reshape(1, 66)

    def _update_kp_cond(self, res_kp_seq, idx, step_len=0):
        # [v1.9.144] Unconditional State Tracking
        # We process these outside the fix_kp_cond branches to ensure monitoring never stops.
        if idx <= 0:
            return # Skip for first frame setup
        last_pose = res_kp_seq[:, idx-1]
        
        # Calculate current pitch degree (Ditto 3DMM logic)
        pitch_bins = last_pose[0, 1:67]
        e_x = np.exp(pitch_bins - np.max(pitch_bins))
        p_soft = e_x / e_x.sum()
        pitch_deg = np.sum(p_soft * np.arange(66)) * 3 - 97.5
        
        self.global_time += 1
        delta_p = pitch_deg - self.s_pitch_deg
        
        # [v1.9.150] Detailed Diagnostic Pulse
        # Always output current frame pitch and silence status
        real_f = res_kp_seq.shape[1]
        
        # [v1.9.155] Hysteresis Logic
        # [v1.9.160] Reverted Stress Test to Standard Safety Zone: 0 ~ -10.0 deg. 
        if self.silence_frames >= 25:
             if not self.is_recovering and delta_p < -10.0:
                  self.is_recovering = True
                  print(f"[Hysteresis Trigger] Delta={delta_p:+.2f}° Breach. Force engaged.")
             elif self.is_recovering and delta_p >= -1.0:
                  self.is_recovering = False
                  self.look_up_timer = 0
                  print(f"[Hysteresis Release] Delta={delta_p:+.2f}°. Neutral zone reached.")
        else:
             # Stop recovering if talking starts
             if self.is_recovering:
                  self.is_recovering = False
                  self.look_up_timer = 0
        
        if self.is_recovering:
             self.look_up_timer += step_len
        
        # [v1.9.165] FLUID POSTURE STATE MACHINE
        # Decisions are based on Silence Frames (Universal) + VAD Lookahead (Optional)
        target_bias_deg = 0.0 # Standard flat
        
        is_currently_talking = self.silence_frames < 25
        has_upcoming_speech = False
        
        if self.vad_timeline is not None:
             lookahead = self.vad_timeline[idx : idx + 50]
             has_upcoming_speech = len(lookahead) > 0 and np.max(lookahead) > 0.1
        
        if is_currently_talking or has_upcoming_speech:
             # COMBAT STANCE: Tucked chin for headroom
             target_bias_deg = -5.0 
        else:
             # IDLE SWAY: Natural breathing
             cycle = np.sin(self.global_time * 0.05)
             target_bias_deg = +2.0 + cycle * 5.0 # Slightly down/neutral sway
        
        # [v1.9.165] Synthetic Anchor Generation
        # Instead of photo bias, we anchor to a mathematical IDEAL distribution.
        self.target_bias_deg = target_bias_deg
        synthetic_pitch = self._generate_gaussian_pitch(target_bias_deg)
        
        # Apply to anchor target
        self.target_neutralizer = self.photo_base_neutralizer.copy()
        # Offset 1:67 is Pitch. We overwrite with synthetic distribution.
        # Since photo_base_neutralizer[1:67] is -s_kp[1:67], 
        # photo_base + s_kp + synthetic = synthetic.
        self.target_neutralizer[0, 1:67] = -self.s_kp_cond[0, 1:67] + synthetic_pitch[0]
        
        # Smooth Transition (0.05 EMA = ~1s glide)
        self.current_neutralizer = self.current_neutralizer * 0.95 + self.target_neutralizer * 0.05
        
        tag = "[HEARTBEAT]" if not self.is_recovering else "[RECOVERY]"
        if has_upcoming_speech and not is_currently_talking:
             tag = "[ANTICIPATION]"
             
        print(f"{tag} Frame {idx:04d} | Delta={delta_p:+.2f}° | Goal={self.target_bias_deg:+.1f}° | Silence={self.silence_frames:03d}")

        # 5. Postural Auto-Correction Logic [REPLACED by v1.9.160 STATIC + v1.9.162 PREDICTIVE]

        if self.fix_kp_cond == 0:  # 不重置
            # 1. Silence Intensity Boost
            boost_factor = 1.0
            if self.silence_frames > 50:
                boost_factor = min(1.6, 1.0 + (self.silence_frames - 50) * 0.004)
            
            noise_scale = 0.005 * boost_factor
            noise = np.random.normal(0, noise_scale, last_pose.shape).astype(np.float32)
            
            # 2. Brownian Momentum Logic
            drift_scales = np.ones_like(last_pose) * (0.0006 * boost_factor)
            new_drift = np.random.normal(0, drift_scales, last_pose.shape).astype(np.float32)
            
            # [v1.9.160] Reverted Uward Nudge Bias
            
            # [v1.9.164] Predictive Physics Push
            new_drift[0, 1:67] += self.current_push
            
            # [v1.9.152] Absolute Postural Force (Boosted to 0.0030)
            if self.is_recovering:
                # [v1.9.158] Decoupled Axis Force
                # High tension for Pitch (1:67), Soft tension for Yaw/Roll/T (67:202)
                new_drift[0, 1:67] -= 0.0030 
                new_drift[0, 67:202] -= 0.0010 # Softened Yaw/Roll pulse
                if self.look_up_timer > 100:
                    new_drift[0, 1:67] -= 0.0030 # Double impulse for Pitch
            
            self.brownian_momentum = self.brownian_momentum * 0.92 + new_drift
            self.brownian_pos += self.brownian_momentum
            
            # 3. Compound Sine Sway
            t = self.global_time
            sway_f1 = np.sin(t * 0.04) * 0.002
            sway_f2 = np.sin(t * 0.25) * 0.0004
            sway_f3 = np.sin(t * 0.005) * 0.0015
            sway = (sway_f1 + sway_f2 + sway_f3).astype(np.float32)
            current_s_kp = self.s_kp_cond + sway
            
            # 4. Micro-Brow Jitters (Expression Indices 15, 16, 18 -> offset 202)
            brow_indices = [217, 218, 220]
            brow_jitter = np.random.normal(0, 0.015 * boost_factor, len(brow_indices)).astype(np.float32)
            
            # 5. Postural Auto-Correction Logic [v1.9.150 Robustness Patch]
            # Shorten silence threshold to 25 frames (1 second) to combat jittery audio
            # Only trigger if notably higher than source (> 2.0 deg Higher -> Delta < -2.0)
            if self.silence_frames >= 25 and delta_p < -2.0:
                self.look_up_timer += step_len
                if self.look_up_timer > 50:
                     tag = "[纠偏活跃]" if self.look_up_timer <= 100 else "[极限强驱]"
                     print(f"{tag} Frame {real_f} | Delta={delta_p:+.2f}° | Applying Pressure.")
            else:
                # ONLY RESET if we are back in the safe zone (Hysteresis)
                if delta_p >= -0.5:
                     if self.look_up_timer > 50:
                          print(f"[Postural] Recovery Finished (Delta={delta_p:+.2f}°). Timer reset.")
                     self.look_up_timer = 0

            gravity_vec = np.ones_like(last_pose) * 0.05
            if self.is_recovering:
                # [v1.9.158] Decoupled Gravity
                # Pitch (Vertical) gets high gravity to prevent looking up
                g_p = 0.80 if self.look_up_timer > 100 else 0.60
                gravity_vec[0, 1:67] = g_p
                # Yaw/Roll/T gets moderate gravity to prevent stiffness/teeth issues
                gravity_vec[0, 67:202] = 0.40 
            
            # 6. Integration
            next_pose = last_pose + noise
            for i, idx_in_kp in enumerate(brow_indices):
                if idx_in_kp < next_pose.shape[1]:
                    next_pose[0, idx_in_kp] += brow_jitter[i]

            anchor = current_s_kp + self.brownian_pos
            self.kp_cond = next_pose * (1.0 - gravity_vec) + anchor * gravity_vec
            
        elif self.fix_kp_cond > 0:
            if self.clip_idx % self.fix_kp_cond == 0:  # 重置
                self.kp_cond = self.s_kp_cond.copy()  # 重置所有
                if self.fix_kp_cond_dim is not None:
                    ds, de = self.fix_kp_cond_dim
                    self.kp_cond[:, ds:de] = res_kp_seq[:, idx-1, ds:de]
            else:
                self.kp_cond = res_kp_seq[:, idx-1]

    def _smo(self, res_kp_seq, s, e):
        # [Revert v1.9.46] Back to Legacy Integer Window Smoothing (Pose Smoothing)
        # Note: 'mouth_smoothing' (EMA) is now handled in MotionStitch, separate from this.
        k = int(self.smo_k_d)
        if k <= 1:
            return res_kp_seq
            
        new_res_kp_seq = res_kp_seq.copy()
        n = res_kp_seq.shape[1]
        half_k = k // 2
        for i in range(s, e):
            ss = max(0, i - half_k)
            ee = min(n, i + half_k + 1)
            res_kp_seq[:, i, :202] = np.mean(new_res_kp_seq[:, ss:ee, :202], axis=1)
        return res_kp_seq
    
    def __call__(self, aud_cond, res_kp_seq=None, reset=False, step_len=None, seed=None):
        """
        aud_cond: (1, seq_frames, dim)
        step_len: int, optional. Frames to advance. Defaults to self.valid_clip_len.
        """
        if step_len is None:
            step_len = self.valid_clip_len

        # [v1.9.163] Restore Condition Update
        # Without this, self.kp_cond and self.current_neutralizer NEVER UPDATE.
        # We pass self.global_time or absolute frame count to sync with VAD.
        # res_kp_seq.shape[1] is the correct current global frame.
        if res_kp_seq is not None:
             self._update_kp_cond(res_kp_seq, res_kp_seq.shape[1], step_len)
        else:
             # First frame, lookahead from 0
             self._update_kp_cond(self.s_kp_cond.reshape(1, 1, -1), 0, step_len)

        if reset:
            # [SOFT RESET] Only reset random seed for lip-sync consistency.
            # DO NOT reset kp_cond to reference - let model continue from current idle state.
            # This allows natural transition from idle to speaking without visual discontinuity.
            
            # Reset Random Seed to ensure Noise Sampling is consistent
            # [Update v1.9.108] Add clip_idx offset to prevent identical onset micro-motion
            if seed is not None:
                offset_seed = (seed + self.clip_idx) % (2**32)
                print(f"[Ditto Debug] Soft Reset: Seed={seed} + Offset={self.clip_idx} (kp_cond preserved)")
                torch.manual_seed(offset_seed)
                torch.cuda.manual_seed(offset_seed)
                torch.cuda.manual_seed_all(offset_seed)
            
            # [v1.9.139] Speech resets silence counter
            self.silence_frames = 0
        else:
            # Increment silence counter during idle
            self.silence_frames += step_len

        pred_kp_seq = self.lmdm(self.kp_cond, aud_cond, self.sampling_timesteps)
        
        # [v1.9.165] FLUID POSTURE MODE
        # We now use "Mathematical Gravity" to pull the head into the safe zone.
        if True:
            # [v1.9.165] Fluid Pressure: 75% Pitch / 30% Yaw-Roll 
            # This restores expressiveness while maintaining vertical stability.
            pressure = 0.75 
            soft_p = 0.30 
            
            # Stable Anchor = Photo + Synthetic Offset + Brownian Drift
            anchor_p = (self.s_kp_cond + self.current_neutralizer + self.brownian_pos)[0, 1:202]
            
            for f in range(pred_kp_seq.shape[1]):
                # Absolute Pitch Locking (Halt baring distortion)
                pred_kp_seq[0, f, 1:67] = pred_kp_seq[0, f, 1:67] * (1.0 - pressure) + anchor_p[0:66] * pressure
                # Soft Rotational Locking (Organic movement)
                pred_kp_seq[0, f, 67:202] = pred_kp_seq[0, f, 67:202] * (1.0 - soft_p) + anchor_p[66:201] * soft_p
            
            if self.clip_idx % 20 == 0:
                 mode_s = "SPEECH" if self.target_bias_deg < -2.0 else "IDLE"
                 print(f"[v1.9.165 {mode_s}] Fluid Gravity: {self.target_bias_deg:+.1f}° goal.")
        
        # [v1.9.156] Virtual Last Frame for Startup Stabilization
        # If this is the VERY first chunk, we treat the source photo as the "prev frame"
        # to trigger a 50-frame gentle glide into the AI's path.
        effective_res_kp_seq = res_kp_seq
        if effective_res_kp_seq is None:
             if self.last_kp_frame is not None:
                  effective_res_kp_seq = self.last_kp_frame 
             else:
                  # Force warp from source photo for Frame 0 stabilization
                  effective_res_kp_seq = self.s_kp_cond.reshape(1, 1, -1)
             
        if effective_res_kp_seq is not None:
             # Calculate offset between last real frame and first new predicted frame
             # pred_kp_seq shape: [1, seq_frames, dim]
             actual_last = effective_res_kp_seq[:, -1:] # [1, 1, dim]
             predicted_first = pred_kp_seq[:, 0:1] # [1, 1, dim]
             
             offset = actual_last - predicted_first # [1, 1, dim]
             
             # Apply decaying warp to the new sequence
             # We want to return to the model's path over ~50 frames (v1.9.107: longer decay)
             decay_len = 50
             warp_len = min(decay_len, pred_kp_seq.shape[1])
             weights = np.linspace(1.0, 0.0, warp_len).reshape(1, -1, 1)
             
             # Apply offset to warp zone
             pred_kp_seq[:, :warp_len] += offset * weights

        if res_kp_seq is None:
            res_kp_seq = pred_kp_seq   # [1, seq_frames, dim]
            res_kp_seq = self._smo(res_kp_seq, 0, res_kp_seq.shape[1])
        else:
            # [Removed] fuse_alpha=1.0 override. Use normal fusion for smooth blending.
            res_kp_seq = self._fuse(res_kp_seq, pred_kp_seq, override_alpha=None, step_len=step_len)
            
            res_kp_seq = self._smo(res_kp_seq, res_kp_seq.shape[1] - step_len - self.fuse_length, res_kp_seq.shape[1])
        
        # Store for next batch
        self.last_kp_frame = res_kp_seq[:, -1:]
        
        # [v1.9.153] Anchor Suppression Logic:
        # If we are in recovery, the "Stable Center" should NO LONGER follow the AI's drift.
        # Instead, we should actively pull the anchor back towards the 0-point (The Original Photo).
        if self.is_recovering:
            # Force the anchor to decay back to neutral photo position
            # This ensures recovery gravity pulls the head DOWN, not back to its CURRENT tilted position.
            self.brownian_pos = (self.brownian_pos * 0.7).astype(np.float32)
            if self.clip_idx % 5 == 0:
                 print(f"[Postural] Anchor Resetting... (Dist={np.abs(self.brownian_pos[0, 1:67]).mean():.4f})")
        else:
            # Normal speech persistence: anchor follows AI slowly to prevent rubber-banding
            target_drift = (self.last_kp_frame - self.s_kp_cond).squeeze()
            self.brownian_pos = (self.brownian_pos * 0.9 + target_drift * 0.1).astype(np.float32)

        self.clip_idx += 1

        idx = res_kp_seq.shape[1] - self.overlap_v2
        self._update_kp_cond(res_kp_seq, idx, step_len=step_len)

        return res_kp_seq
    
    def cvt_fmt(self, res_kp_seq):
        # res_kp_seq: [1, n, dim]
        if self.v_min_max_for_clip is not None:
            tmp_res_kp_seq = np.clip(res_kp_seq[0], self.v_min, self.v_max)
        else:
            tmp_res_kp_seq = res_kp_seq[0]

        x_d_info_list = []
        for i in range(tmp_res_kp_seq.shape[0]):
            x_d_info = _cvt_LP_motion_info(tmp_res_kp_seq[i], 'arr2dic')   # {k: (1, dim)}
            x_d_info_list.append(x_d_info)
        return x_d_info_list
