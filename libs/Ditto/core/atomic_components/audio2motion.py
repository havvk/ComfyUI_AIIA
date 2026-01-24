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
    ):
        self.smo_k_d = smo_k_d
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
    
    def _update_kp_cond(self, res_kp_seq, idx):
        if self.fix_kp_cond == 0:  # 不重置
            # [v1.9.139] Silence Tracker & Adaptive Logic
            # Detect silence intensity (Threshold < 0.1 VAD)
            is_silent = (idx >= res_kp_seq.shape[1]) or (np.mean(res_kp_seq[:, idx-1, :]) < 0.001) # fallback VAD check
            # Real VAD is better if accessible, but we track frames since last audio here.
            # In __call__, we reset silence_frames if audio is present.
            
            # [Method 3: Threshold Boost]
            # Beyond 50 frames (2s), we ramp noise and step size to counteract EMA stabilization.
            boost_factor = 1.0
            if self.silence_frames > 50:
                # Linear ramp up to 1.6x over the next 150 frames
                boost_factor = min(1.6, 1.0 + (self.silence_frames - 50) * 0.004)
            
            last_pose = res_kp_seq[:, idx-1]
            # Base Noise: 0.005 -> 0.008 at full boost
            noise_scale = 0.005 * boost_factor
            noise = np.random.normal(0, noise_scale, last_pose.shape).astype(np.float32)
            
            # [Method 2: Restless Brownian / Postural Shifts]
            # We add momentum to the random walk so the "wandering" feels like intentional shifting.
            drift_scales = np.ones_like(last_pose) * (0.0006 * boost_factor)
            new_drift = np.random.normal(0, drift_scales, last_pose.shape).astype(np.float32)
            # 0.92 persistence for momentum (Slow, weighted drift)
            self.brownian_momentum = self.brownian_momentum * 0.92 + new_drift
            self.brownian_pos += self.brownian_momentum
            
            # [Method 1: Compound Sine Sway]
            self.global_time += 1
            t = self.global_time
            # Layer 1: Main Breathing (~15-20 frames)
            sway_f1 = np.sin(t * 0.04) * 0.002
            # Layer 2: Micro Jitter/Heartbeat (~3-5 frames)
            sway_f2 = np.sin(t * 0.25) * 0.0004
            # Layer 3: Macro Postural Drift (~300 frames)
            sway_f3 = np.sin(t * 0.005) * 0.0015
            
            sway = (sway_f1 + sway_f2 + sway_f3).astype(np.float32)
            current_s_kp = self.s_kp_cond + sway
            
            # 3. Micro-Brow Jitters (Indices 15, 16, 18)
            brow_indices = [15, 16, 18]
            # [v1.9.141] Postural Auto-Correction (Anti-Stall)
            # Detect if head is stuck in an upward-tilted ("Looking Up") pose.
            # Pitch indices are 1:67 in the flattened vector.
            pitch_bins = last_pose[0, 1:67]
            # Use softmax + weighted sum to get current degree (Ditto 3DMM logic)
            # Note: scipy.special.softmax isn't imported here, so we do manual softmax
            e_x = np.exp(pitch_bins - np.max(pitch_bins))
            p_soft = e_x / e_x.sum()
            pitch_deg = np.sum(p_soft * np.arange(66)) * 3 - 97.5
            
            # If looking up (>1.0 deg) during long silence, start recovery timer
            if self.silence_frames > 50 and pitch_deg > 1.0:
                self.look_up_timer += 1
                if self.look_up_timer % 10 == 0:
                     print(f"[Postural Diag] Pitch={pitch_deg:.2f}° | Timer={self.look_up_timer}/50 | Silence={self.silence_frames}")
            else:
                self.look_up_timer = 0
            
            # Define gravity vector (Pitch gets special treatment during stall)
            # Default 5% (0.05)
            gravity_vec = np.ones_like(last_pose) * 0.05
            if self.look_up_timer > 50:
                if self.look_up_timer == 51:
                    print(f"[Postural RECOVERY] Pitch stalled at {pitch_deg:.2f}°. Applying 20% Gravity to Pitch.")
                # Ramp up restoring force to 20% to slowly pull head back to center
                gravity_vec[0, 1:67] = 0.20
            
            brow_jitter = np.random.normal(0, 0.015 * boost_factor, (len(brow_indices), 3)).astype(np.float32)
            
            next_pose = last_pose + noise
            # Inject jitter into specific points
            for i, idx_in_kp in enumerate(brow_indices):
                if idx_in_kp < next_pose.shape[1]:
                    next_pose[:, idx_in_kp] += brow_jitter[i]

            # Anchor = Reference + Compound Sway + Brownian Posture
            anchor = current_s_kp + self.brownian_pos
            # Use spatially-varying gravity for correction
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
        
        # [v1.9.104/107] Persistence & Pose Warping (Anti-Teleport)
        # Check for global continuity (even across audio batches)
        effective_res_kp_seq = res_kp_seq
        if effective_res_kp_seq is None and self.last_kp_frame is not None:
             effective_res_kp_seq = self.last_kp_frame # Mock sequence for offset calc
             
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
        
        # [v1.9.109] Dynamic Anchor:
        # Slowly pull the brownian anchor towards the AI's final pose.
        # This prevents the "Rubber Band" effect after long speech segments.
        # Brownian position is the offset relative to s_kp_cond.
        # res_kp_seq shape: [1, seq_frames, dim]
        target_drift = (self.last_kp_frame - self.s_kp_cond).squeeze()
        # Smoothly interpolate brownian_pos towards this target
        # 0.1 factor per call block means it follows moderately fast during speech
        self.brownian_pos = (self.brownian_pos * 0.9 + target_drift * 0.1).astype(np.float32)

        self.clip_idx += 1

        idx = res_kp_seq.shape[1] - self.overlap_v2
        self._update_kp_cond(res_kp_seq, idx)

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
