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
        self.warp_offset = np.zeros_like(self.s_kp_cond) # [v1.9.197] Persistent Warp Offset
        self.warp_decay = 0.0 # [v1.9.197] Current decay multiplier
        self.brownian_pos = np.zeros_like(self.kp_cond)
        self.last_kp_frame = None # [v1.9.107] Persistence bridging
        self.global_time = 0 # [v1.9.107] For procedural breathing
        self.silence_frames = 0 # [v1.9.139] Track silence duration for adaptive boost
        self.brownian_momentum = np.zeros_like(self.kp_cond) # [v1.9.139] Postural inertia
        self.look_up_timer = 0 # [v1.9.141] Timer for anti-stall recovery
        self.is_recovering = False # [v1.9.155] Hysteresis state flag
        self.target_bias_deg = -1.5 # [v1.9.190] Tightened Sweet Spot (Target Delta)
        self.current_push = 0.0018 # [v1.9.190] Increased Chin-Tuck Pressure
        self.delta_p = 0.0 # [v1.9.190] SAFETY: Fixed AttributeError
        
        # [v1.9.170] Pure Photo Anchor (Reverted Neutralizer)
        self.photo_base_neutralizer = np.zeros_like(self.s_kp_cond)
        
        # [v1.9.146] Capture Source Pitch Baseline
        # We need the "Original" degree of the source photo to detect relative stalls.
        s_pitch_bins = self.s_kp_cond[0, 1:67]
        e_s = np.exp(s_pitch_bins - np.max(s_pitch_bins))
        p_s = e_s / e_s.sum()
        self.s_pitch_deg = np.sum(p_s * np.arange(66)) * 3 - 97.5
        print(f"[Postural Setup] Source Pitch Base: {self.s_pitch_deg:.2f}°")

        self.persistent_pressure = 0.60 # [v1.9.199] Persistent state for smooth transition

    def _fuse(self, res_kp_seq, pred_kp_seq, override_alpha=None, step_len=None):
        # [v1.9.208] Robust Streaming Fusion Fix
        if res_kp_seq is None: return pred_kp_seq
        
        # Never use self.seq_frames (80) for slicing pred_kp_seq, 
        # as pred_kp_seq might only contain step_len frames in streaming mode.
        fuse_r2_s = pred_kp_seq.shape[1] - step_len - self.fuse_length
        fuse_r2_e = pred_kp_seq.shape[1] - step_len
        
        if self.fuse_length > 0 and fuse_r2_s >= 0:
             r1 = res_kp_seq[:, -self.fuse_length:]
             r2 = pred_kp_seq[:, fuse_r2_s:fuse_r2_e]
             alpha = override_alpha if override_alpha is not None else self.fuse_alpha
             r_fuse = r1 * (1 - alpha) + r2 * alpha
             res_kp_seq[:, -self.fuse_length:] = r_fuse
             return np.concatenate([res_kp_seq, pred_kp_seq[:, fuse_r2_e:]], axis=1)
        else:
             return np.concatenate([res_kp_seq, pred_kp_seq], axis=1)
    
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
        self.delta_p = pitch_deg - self.s_pitch_deg
        
        # [v1.9.150] Detailed Diagnostic Pulse
        # Always output current frame pitch and silence status
        real_f = res_kp_seq.shape[1]
        
        # [v1.9.190] Postural Stability State Machine
        # Trigger Limit: -8.0 deg (User limit). Safe Release: -2.0 deg.
        if self.silence_frames >= 25:
             if not self.is_recovering and self.delta_p < -8.0:
                  self.is_recovering = True
                  print(f"[Hysteresis Trigger] Delta={self.delta_p:+.2f}° Breach (< -8.0). Engaging downward pressure.")
             elif self.is_recovering and self.delta_p >= -2.0:
                  self.is_recovering = False
                  self.look_up_timer = 0
                  print(f"[Hysteresis Release] Delta={self.delta_p:+.2f}° Zone Cleared (> -2.0).")
        else:
             # Stop recovering if talking starts
             if self.is_recovering:
                  self.is_recovering = False
                  self.look_up_timer = 0
        
        if self.is_recovering:
             self.look_up_timer += step_len
        
        # [v1.9.215] ZERO-PRESSURE SPEECH STATE MACHINE (Baseline Restoration)
        # Decisions are based on Silence Frames (Universal) + VAD Lookahead (Optional)
        is_currently_talking = self.silence_frames < 25
        has_upcoming_speech = False
        
        if self.vad_timeline is not None:
             lookahead = self.vad_timeline[idx : idx + 50]
             has_upcoming_speech = len(lookahead) > 0 and np.max(lookahead) > 0.1
        
        # We store the state for pressure selection in __call__
        self.is_talking_state = is_currently_talking or has_upcoming_speech
        
        # Log Heartbeat
        tag = "[HEARTBEAT]" if not self.is_recovering else "[RECOVERY]"
        if has_upcoming_speech and not is_currently_talking:
             tag = "[ANTICIPATION]"
             
        if self.clip_idx % 20 == 0:
             print(f"{tag} Frame {idx:04d} | Delta={self.delta_p:+.2f}° | Silence={self.silence_frames:03d}")

        # [Deactivated v1.9.198] Postural Anticipation (Speech Onset Prep)
        # We rely exclusively on the Persistent Warp logic for better isolation.

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
            
            # [v1.9.190] Decoupled Axis Force (Active Chin-Tuck)
            if self.is_recovering:
                # Add drift (push down/forward) to recover from high-tilt
                new_drift[0, 1:67] += 0.0045 
                new_drift[0, 67:202] -= 0.0005 
                if self.look_up_timer > 100:
                    new_drift[0, 1:67] += 0.0045 
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
            if self.silence_frames >= 25 and self.delta_p < -2.0:
                self.look_up_timer += step_len
                if self.look_up_timer > 50:
                     tag = "[纠偏活跃]" if self.look_up_timer <= 100 else "[极限强驱]"
                     print(f"{tag} Frame {real_f} | Delta={self.delta_p:+.2f}° | Applying Pressure.")
            else:
                # ONLY RESET if we are back in the safe zone (Hysteresis)
                if self.delta_p >= -0.5:
                     if self.look_up_timer > 50:
                          print(f"[Postural] Recovery Finished (Delta={self.delta_p:+.2f}°). Timer reset.")
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
        
        # [v1.9.199] SMOOTH PRESSURE TRANSITION
        target_pressure = 0.0 if getattr(self, "is_talking_state", False) else 0.60
        anchor_p = (self.s_kp_cond + self.brownian_pos)[0, 1:202]
        
        for f in range(pred_kp_seq.shape[1]):
             # Gradually move pressure towards target (10-frame transition)
             diff = target_pressure - self.persistent_pressure
             move = np.clip(diff, -0.06, 0.06) # Max 6% change per frame (~0.4s total)
             self.persistent_pressure += move
             
             # Apply current pressure state
             curr_p = self.persistent_pressure
             pred_kp_seq[0, f, 1:202] = pred_kp_seq[0, f, 1:202] * (1.0 - curr_p) + anchor_p[0:201] * curr_p
             
        if self.clip_idx % 20 == 0:
             mode_s = "SPEECH" if target_pressure == 0 else "IDLE"
             print(f"[v1.9.216 {mode_s}] Pressure: {self.persistent_pressure*100:.0f}% (Delta={self.delta_p:+.2f})")
        
        # Fusion Sequence
        # [v1.9.216] PRECISION JUNCTION DIAGNOSTIC
        fuse_r2_s = pred_kp_seq.shape[1] - step_len - self.fuse_length

        if (reset or res_kp_seq is None):
             actual_last = res_kp_seq[:, -1:] if res_kp_seq is not None else self.s_kp_cond.reshape(1, 1, -1)
             
             # CLAMP Junc to at least 0. We align to the EARLIEST relevant frame.
             junc_idx = max(0, fuse_r2_s)
             target_entry = pred_kp_seq[:, junc_idx : junc_idx + 1]
             self.warp_offset = actual_last - target_entry
             self.warp_decay = 1.0
             
             # [v1.9.216] Detailed Diagnostic Logging
             h_shape = res_kp_seq.shape if res_kp_seq is not None else "None"
             print(f"[Ditto] Precision Warp Engaged (v1.9.216). "
                   f"Junc={fuse_r2_s} (Clamped={junc_idx}), "
                   f"PredShape={pred_kp_seq.shape}, HistShape={h_shape}, "
                   f"Step={step_len}, FuseLen={self.fuse_length}, "
                   f"Offset={np.abs(self.warp_offset).mean():.4f}")

        # Apply Warp alignment starting from the junction point
        if self.warp_decay > 0.001:
             start_f = max(0, fuse_r2_s)
             for f in range(start_f, pred_kp_seq.shape[1]):
                  # ONLY warp head pose/pos (0:201). Leave expressions (201+) native.
                  pred_kp_seq[0, f, :201] += self.warp_offset[0, 0, :201] * self.warp_decay
                  self.warp_decay *= 0.985 # Slower, stable decay
             
             if self.warp_decay < 0.001:
                  self.warp_decay = 0.0
                  self.warp_offset = np.zeros_like(self.warp_offset)

        if res_kp_seq is None:
            res_kp_seq = pred_kp_seq[:, :step_len]
            res_kp_seq = self._smo(res_kp_seq, 0, res_kp_seq.shape[1])
        else:
            res_kp_seq = self._fuse(res_kp_seq, pred_kp_seq, override_alpha=None, step_len=step_len)
            res_kp_seq = self._smo(res_kp_seq, res_kp_seq.shape[1] - step_len - self.fuse_length, res_kp_seq.shape[1])
        
        # Store for next batch
        self.last_kp_frame = res_kp_seq[:, -1:]
        
        # [v1.9.153] Anchor Suppression Logic:
        if self.is_recovering:
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
