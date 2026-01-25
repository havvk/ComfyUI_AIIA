import copy
import random
import numpy as np
from scipy.special import softmax

from ..models.stitch_network import StitchNetwork


"""
# __init__
stitch_network_cfg = {
    "model_path": "",
    "device": "cuda",
}

# __call__
kwargs:
    fade_alpha
    fade_out_keys

    delta_pitch
    delta_yaw
    delta_roll

"""


def ctrl_motion(x_d_info, **kwargs):
    # pose + offset
    for kk in ["delta_pitch", "delta_yaw", "delta_roll"]:
        if kk in kwargs:
            k = kk[6:]
            x_d_info[k] = bin66_to_degree(x_d_info[k]) + kwargs[kk]

    # pose * alpha
    for kk in ["alpha_pitch", "alpha_yaw", "alpha_roll"]:
        if kk in kwargs:
            k = kk[6:]
            x_d_info[k] = x_d_info[k] * kwargs[kk]

    # exp + offset
    if "delta_exp" in kwargs:
        k = "exp"
        x_d_info[k] = x_d_info[k] + kwargs["delta_exp"]

    # mouth amp
    if "mouth_amp" in kwargs:
        _lip = [6, 7, 8, 9, 10, 12, 14, 17, 19, 20]
        x_d_info["exp"][:, _lip] *= kwargs["mouth_amp"]

    return x_d_info


def fade(x_d_info, dst, alpha, keys=None):
    if keys is None:
        keys = x_d_info.keys()
    for k in keys:
        if k == 'kp':
            continue
        x_d_info[k] = x_d_info[k] * alpha + dst[k] * (1 - alpha)
    return x_d_info


def ctrl_vad(x_d_info, dst, alpha):
    # 1. Blend Expression (Existing)
    # Relax to dst['exp'] (Reference/Neutral) as alpha -> 0
    exp = x_d_info["exp"]
    exp_dst = dst["exp"]
    x_d_info["exp"] = exp * alpha + exp_dst * (1 - alpha)
    
    # [REVERTED] Head Pose Relaxation
    # We found that forcing the head pose to reference during silence 
    # kills the natural "Micro-Drift" and makes the avatar look static/dead.
    # Since our "Pre-Fuse Warping" (in audio2motion.py) already solves the 
    # "Snap" problem perfectly, we don't need this aggressive locking.
    # Letting the head drift naturally gives more "Life".

    return x_d_info
    
    

def _mix_s_d_info(
    x_s_info,
    x_d_info,
    use_d_keys=("exp", "pitch", "yaw", "roll", "t"),
    d0=None,
):
    if d0 is not None:
        if isinstance(use_d_keys, dict):
            x_d_info = {
                k: x_s_info[k] + (v - d0[k]) * use_d_keys.get(k, 1)
                for k, v in x_d_info.items()
            }
        else:
            x_d_info = {k: x_s_info[k] + (v - d0[k]) for k, v in x_d_info.items()}

    for k, v in x_s_info.items():
        if k not in x_d_info or k not in use_d_keys:
            x_d_info[k] = v

    if isinstance(use_d_keys, dict) and d0 is None:
        for k, alpha in use_d_keys.items():
            x_d_info[k] *= alpha
    return x_d_info


def _set_eye_blink_idx(N, blink_n=15, open_n=-1, interval_min=60, interval_max=100, vad_timeline=None, speech_only_blink=False):
    """
    open_n:
        -1: no blink
        0: random open_n
        >0: fix open_n
        list: loop open_n
    speech_only_blink:
        True: Only insert blinks during speech segments (VAD > 0.1)
        False: Use natural intervals (faster during speech, slower during silence)
    """
    OPEN_MIN = interval_min
    OPEN_MAX = interval_max
    
    # [AIIA Diagnostic] Log parameter flow
    print(f"[AIIA Blink Logic] _set_eye_blink_idx: N={N}, interval={OPEN_MIN}-{OPEN_MAX}, speech_only={speech_only_blink}, VAD={'Set' if vad_timeline is not None else 'None'}")
    if vad_timeline is not None:
        vad_sum = np.sum(vad_timeline > 0.1)
        print(f"  > VAD Stats: {vad_sum} frames of speech / {len(vad_timeline)} total")

    idx = [0] * N
    if isinstance(open_n, int):
        if open_n < 0:  # no blink
            return idx
        elif open_n > 0:  # fix open_n
            open_ns = [open_n]
        else:  # open_n == 0:  # random open_n, 60-100
            open_ns = []
    elif isinstance(open_n, list):
        open_ns = open_n  # loop open_n
    else:
        raise ValueError()

    blink_idx = list(range(blink_n))

    if speech_only_blink and vad_timeline is not None:
        # Start scanning for speech from frame 0
        scan_start = 0
        while scan_start < len(vad_timeline) and vad_timeline[scan_start] <= 0.1:
            scan_start += 1
            
        if scan_start >= len(vad_timeline):
            print(f"  [Blink Init] Speech-Only: No speech found in entire timeline. Exit.")
            return idx
            
        # Find end of first segment
        scan_end = scan_start
        while scan_end < len(vad_timeline) and vad_timeline[scan_end] > 0.1:
            scan_end += 1
            
        # First blink: User expectation or random start
        jump = random.randint(OPEN_MIN, OPEN_MAX)
        target_i = scan_start + jump
        
        if target_i + blink_n <= scan_end:
             cur_i = target_i
        else:
             if (scan_end - scan_start) >= blink_n:
                 cur_i = random.randint(scan_start, scan_end - blink_n)
             else:
                 # Skip short segment, let loop handle it
                 cur_i = scan_end
                 
        print(f"  [Blink Init] Speech-Only: First speech at {scan_start}-{scan_end}, scheduling first blink at {cur_i}")
    else:
        start_n = open_ns[0] if open_ns else random.randint(OPEN_MIN, OPEN_MAX)
        cur_i = start_n
        print(f"  [Blink Init] Natural: Scheduling first blink at {cur_i}")

    end_n = open_ns[-1] if open_ns else random.randint(OPEN_MIN, OPEN_MAX)
    max_i = N - max(end_n, blink_n)

    cur_n_i = 1
    
    # Double Blink Probability
    # double_blink_prob = 0.15
    double_blink_prob = 0 # Disabled by user request
    
    # Hold Duration (Frames to keep eye closed)
    blink_hold_min = 1
    blink_hold_max = 1
    
    while cur_i < max_i:
        # [v1.9.101] Strict Speech-Only Validation
        # Check if current cur_i is valid for a blink
        if speech_only_blink and vad_timeline is not None:
             # If current position is silence, find next segment
             if vad_timeline[cur_i] <= 0.1:
                 scan_start = cur_i
                 while scan_start < len(vad_timeline) and vad_timeline[scan_start] <= 0.1:
                     scan_start += 1
                 
                 if scan_start >= len(vad_timeline): break
                 
                 # New segment found
                 scan_end = scan_start
                 while scan_end < len(vad_timeline) and vad_timeline[scan_end] > 0.1:
                     scan_end += 1
                     
                 # Force blink into this segment
                 if (scan_end - scan_start) >= blink_n:
                     cur_i = random.randint(scan_start, scan_end - blink_n)
                 else:
                     cur_i = scan_end
                     continue # Too short, try next
             else:
                 # Already in speech, check if it fits before segment ends
                 scan_end = cur_i
                 while scan_end < len(vad_timeline) and vad_timeline[scan_end] > 0.1:
                      scan_end += 1
                 
                 if cur_i + blink_n > scan_end:
                      # Not enough room here, jump to end and find next segment
                      cur_i = scan_end
                      continue

        # First Blink
        hold_frames = random.randint(blink_hold_min, blink_hold_max)
        
        # Construct Blink Sequence with Hold
        # split blink_idx into open->close (first half) and close->open (second half)
        mid_point = blink_n // 2
        # e.g. 0...7 (8 frames) + 7...7 (hold) + 8...14 (7 frames)
        
        # Safe slice
        seq_start = blink_idx[:mid_point]
        seq_mid = [blink_idx[mid_point]] * (hold_frames + 1) # +1 to include original
        seq_end = blink_idx[mid_point+1:]
        
        full_blink_seq = seq_start + seq_mid + seq_end
        full_len = len(full_blink_seq)
        
        if cur_i + full_len > len(idx): 
            break
            
        idx[cur_i : cur_i + full_len] = full_blink_seq
        
        # [AIIA Diagnostic] Log each blink event
        vad_val = vad_timeline[cur_i] if vad_timeline is not None else -1
        print(f"  [Blink] Scheduled at frame {cur_i}, VAD={vad_val:.2f}, Type={'Speech' if vad_val > 0.1 else 'Silence'}")
        
        # Decide if Double Blink
        is_double = random.random() < double_blink_prob
        
        # Advance current index
        cur_i += full_len
        
        if is_double:
             # Short gap: 4-8 frames (Clearer re-blink)
             short_gap = random.randint(4, 8)
             cur_i += short_gap
             
             # Second Blink
             hold_frames_2 = random.randint(0, 2) # Shorter hold for 2nd blink
             
             seq_mid_2 = [blink_idx[mid_point]] * (hold_frames_2 + 1)
             full_blink_seq_2 = seq_start + seq_mid_2 + seq_end
             full_len_2 = len(full_blink_seq_2)
             
             if cur_i + full_len_2 > len(idx): 
                 break # Safety break
                 
             idx[cur_i : cur_i + full_len_2] = full_blink_seq_2
             cur_i += full_len_2

        # Calculate Next Event
        if open_ns:
            cur_n = open_ns[cur_n_i % len(open_ns)]
            cur_n_i += 1
        else:
            if vad_timeline is not None and not speech_only_blink:
                # Natural Mode (Not speech-only)
                is_speaking = vad_timeline[cur_i] > 0.1 if cur_i < len(vad_timeline) else False
                if is_speaking:
                    cur_n = random.randint(OPEN_MIN, OPEN_MAX)
                    print(f"  [Blink Loop] Natural: Speaking at {cur_i}, next in {cur_n}")
                else:
                    cur_n = random.randint(int(OPEN_MIN * 1.5), int(OPEN_MAX * 1.5))
                    print(f"  [Blink Loop] Natural: Silence at {cur_i}, next in {cur_n} (1.5x Relaxed)")
            else:
                # Standard or Speech-Only prep (Speech-Only will re-validate at loop top)
                cur_n = random.randint(OPEN_MIN, OPEN_MAX)
                if speech_only_blink:
                    print(f"  [Blink Loop] Speech-Only Pre-Interval: +{cur_n}")

        cur_i += cur_n

    return idx


def _fix_exp_for_x_d_info(x_d_info, x_s_info, delta_eye=None, drive_eye=True):
    _eye = [11, 13, 15, 16, 18] # [Native v1.9.94] Restored full official eye indices. 18 (Brow Down) added back.



    _lip = [6, 7, 8, 9, 10, 12, 14, 17, 19, 20] # [v1.9.188] Added Upper Lip (7,8,9,10) for Symmetry

    alpha = np.zeros((21, 3), dtype=x_d_info["exp"].dtype)
    alpha[_lip] = 1
    if delta_eye is None and drive_eye:  # use d eye
        alpha[_eye] = 1
    alpha = alpha.reshape(1, -1)
    x_d_info["exp"] = x_d_info["exp"] * alpha + x_s_info["exp"] * (1 - alpha)

    if delta_eye is not None and drive_eye:
        alpha = np.zeros((21, 3), dtype=x_d_info["exp"].dtype)
        alpha[_eye] = 1
        alpha = alpha.reshape(1, -1)
        x_d_info["exp"] = (delta_eye + x_s_info["exp"]) * alpha + x_d_info["exp"] * (
            1 - alpha
        )

    return x_d_info


def _fix_exp_for_x_d_info_v2(x_d_info, x_s_info, delta_eye, a1, a2, a3):
    x_d_info["exp"] = x_d_info["exp"] * a1 + x_s_info["exp"] * a2 + delta_eye * a3
    return x_d_info


def bin66_to_degree(pred):
    if pred.ndim > 1 and pred.shape[1] == 66:
        idx = np.arange(66).astype(np.float32)
        pred = softmax(pred, axis=1)
        degree = np.sum(pred * idx, axis=1) * 3 - 97.5
        return degree
    return pred


def _eye_delta(exp, dx=0, dy=0):
    if dx > 0:
        exp[0, 33] += dx * 0.0007
        exp[0, 45] += dx * 0.001
    else:
        exp[0, 33] += dx * 0.001
        exp[0, 45] += dx * 0.0007

    exp[0, 34] += dy * -0.001
    exp[0, 46] += dy * -0.001
    return exp

def _fix_gaze(pose_s, x_d_info):
    x_ratio = 0.26
    y_ratio = 0.28
    
    yaw_s, pitch_s = pose_s
    yaw_d = bin66_to_degree(x_d_info['yaw']).item()
    pitch_d = bin66_to_degree(x_d_info['pitch']).item()

    delta_yaw = yaw_d - yaw_s
    delta_pitch = pitch_d - pitch_s

    dx = delta_yaw * x_ratio
    dy = delta_pitch * y_ratio
    
    x_d_info['exp'] = _eye_delta(x_d_info['exp'], dx, dy)
    return x_d_info


def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * np.pi
    yaw = yaw_ / 180 * np.pi
    roll = roll_ / 180 * np.pi

    if pitch.ndim == 1:
        pitch = pitch[:, None]
    if yaw.ndim == 1:
        yaw = yaw[:, None]
    if roll.ndim == 1:
        roll = roll[:, None]

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = np.ones((bs, 1), dtype=np.float32)
    zeros = np.zeros((bs, 1), dtype=np.float32)
    x, y, z = pitch, yaw, roll

    rot_x = np.concatenate([
        ones, zeros, zeros,
        zeros, np.cos(x), -np.sin(x),
        zeros, np.sin(x), np.cos(x)
    ], axis=1).reshape(bs, 3, 3)

    rot_y = np.concatenate([
        np.cos(y), zeros, np.sin(y),
        zeros, ones, zeros,
        -np.sin(y), zeros, np.cos(y)
    ], axis=1).reshape(bs, 3, 3)

    rot_z = np.concatenate([
        np.cos(z), -np.sin(z), zeros,
        np.sin(z), np.cos(z), zeros,
        zeros, zeros, ones
    ], axis=1).reshape(bs, 3, 3)

    rot = np.matmul(np.matmul(rot_z, rot_y), rot_x)
    return np.transpose(rot, (0, 2, 1))


def transform_keypoint(kp_info: dict):
    """
    transform the implicit keypoints with the pose, shift, and expression deformation
    kp: BxNx3
    """
    kp = kp_info['kp']    # (bs, k, 3)
    pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

    t, exp = kp_info['t'], kp_info['exp']
    scale = kp_info['scale']

    pitch = bin66_to_degree(pitch)
    yaw = bin66_to_degree(yaw)
    roll = bin66_to_degree(roll)

    bs = kp.shape[0]
    if kp.ndim == 2:
        num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
    else:
        num_kp = kp.shape[1]  # Bxnum_kpx3

    rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)

    # Eqn.2: s * (R * x_c,s + exp) + t
    kp_transformed = np.matmul(kp.reshape(bs, num_kp, 3), rot_mat) + exp.reshape(bs, num_kp, 3)
    kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty

    return kp_transformed


class MotionStitch:
    def __init__(
        self,
        stitch_network_cfg,
    ):
        self.stitch_net = StitchNetwork(**stitch_network_cfg)

    def set_Nd(self, N_d=-1):
        # only for offline (make start|end eye open)
        if N_d == self.N_d:
            return
        
        self.N_d = N_d
        if self.drive_eye and self.delta_eye_arr is not None:
            N = 3000 if self.N_d == -1 else self.N_d
            self.delta_eye_idx_list = _set_eye_blink_idx(
                N, len(self.delta_eye_arr), self.delta_eye_open_n,
                interval_min=getattr(self, 'blink_interval_min', 60),
                interval_max=getattr(self, 'blink_interval_max', 100),
                vad_timeline=self.vad_timeline if hasattr(self, 'vad_timeline') else None,
                speech_only_blink=getattr(self, 'speech_only_blink', False)
            )

    def setup(
        self,
        N_d=-1,
        use_d_keys=None,
        relative_d=True,
        drive_eye=None,  # use d eye or s eye
        delta_eye_arr=None,  # fix eye
        delta_eye_open_n=-1,  # int|list
        blink_interval_min=60,
        blink_interval_max=100,
        fade_out_keys=("exp",),
        fade_type="",   # "" | "d0" | "s"
        flag_stitching=True,
        is_image_flag=True,
        x_s_info=None,
        d0=None,
        ch_info=None,
        overall_ctrl_info=None,
        vad_timeline=None,
        blink_amp=1.0,
        speech_only_blink=False,
    ):
        self.blink_amp = blink_amp
        self.speech_only_blink = speech_only_blink
        self.blink_interval_min = blink_interval_min
        self.blink_interval_max = blink_interval_max
        self.vad_timeline = vad_timeline
        self.is_image_flag = is_image_flag
        if use_d_keys is None:
            if self.is_image_flag:
                self.use_d_keys = ("exp", "pitch", "yaw", "roll", "t")
            else:
                self.use_d_keys = ("exp", )
        else:
            self.use_d_keys = use_d_keys

        if drive_eye is None:
            if self.is_image_flag:
                self.drive_eye = True
            else:
                self.drive_eye = False
        else:
            self.drive_eye = drive_eye

        self.N_d = N_d
        self.relative_d = relative_d
        self.delta_eye_arr = delta_eye_arr
        self.delta_eye_open_n = delta_eye_open_n
        self.fade_out_keys = fade_out_keys
        self.fade_type = fade_type
        self.flag_stitching = flag_stitching

        _eye = [11, 13, 15, 16, 18] # [Native v1.9.94] Restored full official eye indices.


        _lip = [6, 7, 8, 9, 10, 12, 14, 17, 19, 20] # [v1.9.188] Full Lip Set

        _a1 = np.zeros((21, 3), dtype=np.float32)
        _a1[_lip] = 1
        _a2 = 0
        if self.drive_eye:
            if self.delta_eye_arr is None:
                _a1[_eye] = 1
            else:
                _a2 = np.zeros((21, 3), dtype=np.float32)
                _a2[_eye] = 1
                _a2 = _a2.reshape(1, -1)
        _a1 = _a1.reshape(1, -1)

        self.fix_exp_a1 = _a1 * (1 - _a2)
        self.fix_exp_a2 = (1 - _a1) + _a1 * _a2
        self.fix_exp_a3 = _a2
        
        # [Debug v1.9.208] Verify Code Sync
        print(f"[AIIA Debug] MotionStitch Setup: v1.9.300. LATEST VERSION LOADED.")


        if self.drive_eye and self.delta_eye_arr is not None:
            N = 3000 if self.N_d == -1 else self.N_d
            self.delta_eye_idx_list = _set_eye_blink_idx(
                N, 
                len(self.delta_eye_arr), 
                self.delta_eye_open_n,
                interval_min=blink_interval_min,
                interval_max=blink_interval_max,
                vad_timeline=self.vad_timeline,
                speech_only_blink=self.speech_only_blink
            )

        self.pose_s = None
        self.x_s = None
        self.fade_dst = None
        if self.is_image_flag and x_s_info is not None:
            yaw_s = bin66_to_degree(x_s_info['yaw']).item()
            pitch_s = bin66_to_degree(x_s_info['pitch']).item()
            self.pose_s = [yaw_s, pitch_s]
            self.x_s = transform_keypoint(x_s_info)

            if self.fade_type == "s":
                self.fade_dst = copy.deepcopy(x_s_info)

        if ch_info is not None:
            self.scale_a = ch_info['x_s_info_lst'][0]['scale'].item()
            if x_s_info is not None:
                self.scale_b = x_s_info['scale'].item()
                self.scale_ratio = self.scale_a / self.scale_b
                self._set_scale_ratio(self.scale_ratio)
            else:
                self.scale_ratio = None
        else:
            self.scale_ratio = 1

        self.overall_ctrl_info = overall_ctrl_info

        self.d0 = d0
        self.idx = 0

    def _set_scale_ratio(self, scale_ratio=1):
        if scale_ratio == 1:
            return
        if isinstance(self.use_d_keys, dict):
            self.use_d_keys = {k: v * (scale_ratio if k in {"exp", "pitch", "yaw", "roll"} else 1) for k, v in self.use_d_keys.items()}
        else:
            self.use_d_keys = {k: scale_ratio if k in {"exp", "pitch", "yaw", "roll"} else 1 for k in self.use_d_keys}

    @staticmethod
    def _merge_kwargs(default_kwargs, run_kwargs):
        if default_kwargs is None:
            return run_kwargs
        
        for k, v in default_kwargs.items():
            if k not in run_kwargs:
                run_kwargs[k] = v
        return run_kwargs
    
    def __call__(self, x_s_info, x_d_info, **kwargs):
        # return x_s, x_d

        kwargs = self._merge_kwargs(self.overall_ctrl_info, kwargs)

        if self.scale_ratio is None:
            self.scale_b = x_s_info['scale'].item()
            self.scale_ratio = self.scale_a / self.scale_b
            self._set_scale_ratio(self.scale_ratio)

        if self.relative_d and self.d0 is None:
            self.d0 = copy.deepcopy(x_d_info)

        x_d_info = _mix_s_d_info(
            x_s_info,
            x_d_info,
            self.use_d_keys,
            self.d0,
        )

        # [FIX] Instant Mouth Closing & Onset Ghosting
        # 1. Capture Buffer: Keep history to capture a truly "Open" frame before LMDM closes it.
        # 2. Instant Attack: Force full expression strength on onset to prevent attenuated "ghosting".
        
        vad_current = kwargs.get("vad_alpha", 1.0)
        
        # Initialize state
        if not hasattr(self, 'prev_vad_alpha'):
            self.prev_vad_alpha = 1.0
            self.last_speaking_exp = None
            self.exp_buffer = [] # Store last 5 frames of exp

        # Update Buffer
        # Copy to avoid reference issues. 
        # We only need to buffer when speaking (or transition), but buffering always is safer/simpler.
        self.exp_buffer.append(x_d_info["exp"].copy())
        if len(self.exp_buffer) > 5:
            self.exp_buffer.pop(0)

        # Logic Vars
        is_release_start = (self.prev_vad_alpha >= 0.99 and vad_current < 0.99)
        is_attack = (vad_current > self.prev_vad_alpha)
        
        # 1. Capture Logic (Release Start)
        if is_release_start:
            # Capture from buffer (e.g. -2 or -3) to ensure we get the Open frame 
            # before LMDM started closing.
            # Using -2 (approx 0.08s ago)
            idx = -2 if len(self.exp_buffer) >= 2 else -1
            self.last_speaking_exp = self.exp_buffer[idx].copy()
        
        # 2. Clear Logic (Attack / Re-entry)
        if is_attack:
            self.last_speaking_exp = None
            
        self.prev_vad_alpha = vad_current

        # [v1.9.138] Volumetric Vitality (3D Axis Expansion & Bias Purge)
        # Using Gamma 0.5 for a smoother, less "snappy" transition into speech.
        exp_blend_alpha = vad_current ** 0.5

        if exp_blend_alpha < 1.0:
            # If releasing (and not attacking), use frozen frame if available
            if self.last_speaking_exp is not None and not is_attack:
                 x_d_info["exp"] = self.last_speaking_exp.copy()
            
            x_d_info = ctrl_vad(x_d_info, x_s_info, exp_blend_alpha)

        # [FIX] Expression Temporal Smoothing (EMA)
        # Restore EMA decay to 0.3 for organic persistence.
        mouth_smoothing_mode = kwargs.get("mouth_smoothing", "Normal")
        if mouth_smoothing_mode == "None (Raw)":
            exp_decay = 0.0
        elif mouth_smoothing_mode == "Light":
            exp_decay = 0.2
        elif mouth_smoothing_mode == "Heavy":
            exp_decay = 0.6
        else:
            exp_decay = 0.3
        
        if exp_decay > 0:
            if not hasattr(self, 'prev_exp_ema') or self.prev_exp_ema is None:
                self.prev_exp_ema = x_d_info["exp"].copy()
            
            # [Update v1.9.103] Soft Reset EMA during Attack
            # If we are starting to speak, reset the EMA to the current frame 
            # to prevent the "Mouth Opening Lag" caused by long silence history.
            if is_attack:
                 self.prev_exp_ema = x_d_info["exp"].copy()
            
            # Apply EMA smoothing - let mouth opening transition naturally
            x_d_info["exp"] = self.prev_exp_ema * exp_decay + x_d_info["exp"] * (1.0 - exp_decay)
            self.prev_exp_ema = x_d_info["exp"].copy()
            
        # [v1.9.140] 3D-Aware Mouth Biomechanics (Exponential Jaw Excitation)
        # We target all 3 coordinates (X, Y, Z) for Procedural Breathing 
        # while using an Exponential Gain for the lower lip to boost lazy audio.
        exp_reshaped = x_d_info["exp"].reshape(-1, 21, 3)
        
        # 1. Exponential Jaw Gain (Boost micro-openings for lazy audio)
        # We amplify original AI intent without adding a fixed offset.
        lower_lip = [17, 19, 20]
        y_intent = exp_reshaped[:, lower_lip, 1]
        # Only boost if the AI is actually trying to open the mouth (y > 0)
        mask = y_intent > 0.001
        if np.any(mask):
            # [Reverted v1.9.192] Removed Volumetric Jaw Excitation
            # We revert to 1.0 gain to prevent potential mouth asymmetry (crooked mouth)
            gain = 1.0 
            exp_reshaped[:, lower_lip, 1][mask] *= gain

        # 2. Volumetric Mouth Micro-Motion (Breathing + Corners 6,12 Restored)
        if "delta_mouth" in kwargs:
             _lip_and_corners = [6, 12, 14, 17, 19, 20]
             # [Restored v1.9.195] Targeting all 3 axes [:, :] for high-fidelity 3D expansion
             exp_reshaped[:, _lip_and_corners] += kwargs["delta_mouth"]
             
        # Restore flattened state
        x_d_info["exp"] = exp_reshaped.reshape(1, -1)

        # [v1.9.137] Restore Blink Logic (Fix NameError)
        delta_eye = 0
        if self.drive_eye and self.delta_eye_arr is not None:
            delta_eye = self.delta_eye_arr[
                self.delta_eye_idx_list[self.idx % len(self.delta_eye_idx_list)]
            ][None] * self.blink_amp

        # [Revert v1.9.55] User reports "Original code didn't have twitch".
        # Switching back to Legacy Function (but with [11,13] fix applied inside it).
        x_d_info = _fix_exp_for_x_d_info(
            x_d_info,
            x_s_info,
            delta_eye,
            self.drive_eye
        )
        
        # [v1.9.92] Speech-Only Blink: Blink mirror removed.
        # With blinks only occurring during speech, the mesh coupling artifact
        # (slight mouth twitch) is naturally masked by speaking motion.
        
        x_d_info = ctrl_motion(x_d_info, **kwargs)

        # [Diagnostic v1.9.71] Output Logger
        if hasattr(self, '_log_blink_output') and self._log_blink_output:
             out_exp = x_d_info["exp"]
             # Log Eyelid, Squint, and Mouth Corners (6/12)
             print(f"  > Final Output: EyeL={float(out_exp[:, 11].mean()):.3f}, SquintL={float(out_exp[:, 15].mean()):.3f}, CornerL={float(out_exp[:, 6].mean()):.3f}, CornerR={float(out_exp[:, 12].mean()):.3f}")
             self._log_blink_output = False



        
        # [Moved v1.9.59] Auto-Center Moved to end of function.


        if self.fade_type == "d0" and self.fade_dst is None:
            self.fade_dst = copy.deepcopy(x_d_info)

        # fade
        if "fade_alpha" in kwargs and self.fade_type in ["d0", "s"]:
            fade_alpha = kwargs["fade_alpha"]
            fade_keys = kwargs.get("fade_out_keys", self.fade_out_keys)
            if self.fade_type == "d0":
                fade_dst = self.fade_dst
            elif self.fade_type == "s":
                if self.fade_dst is not None:
                    fade_dst = self.fade_dst
                else:
                    fade_dst = copy.deepcopy(x_s_info)
                    if self.is_image_flag:
                        self.fade_dst = fade_dst
            x_d_info = fade(x_d_info, fade_dst, fade_alpha, fade_keys)

        if self.drive_eye:
            if self.pose_s is None:
                yaw_s = bin66_to_degree(x_s_info['yaw']).item()
                pitch_s = bin66_to_degree(x_s_info['pitch']).item()
                self.pose_s = [yaw_s, pitch_s]
            x_d_info = _fix_gaze(self.pose_s, x_d_info)

        # [Feature v1.9.59] Auto-Center Roll (Anti-Tilt) + Hard Clamp (Late Stage)
        # Applied at the very end of motion calculation (before transform) to ensure it sticks.
        ref_roll = bin66_to_degree(x_s_info['roll'])
        
        # Spring Force (30%)
        x_d_info['roll'] = x_d_info['roll'] * 0.7 + ref_roll * 0.3
        
        # Hard Clamp (Max 1.5 deg drift)
        drift = x_d_info['roll'] - ref_roll
        # Fix dimensions
        if isinstance(drift, np.ndarray):
            drift_val = drift.item() if drift.size == 1 else drift.mean()
        else:
            drift_val = drift
            
        if drift_val > 1.5:
            x_d_info['roll'] = ref_roll + 1.5
        elif drift_val < -1.5:
            x_d_info['roll'] = ref_roll - 1.5
            
        # Diagnostic Logging (Supressed v1.9.78 for Scanner Clarity)
        # if self.idx % 25 == 0:
        #     c_roll = float(np.mean(x_d_info['roll']))
        #     r_roll = float(np.mean(ref_roll))
        #     print(f"[Ditto Tilt Diag] Frame {self.idx}: CurRoll={c_roll:.2f}° | RefRoll={r_roll:.2f}° | Drift={c_roll-r_roll:.2f}°")

        if self.x_s is not None:
            x_s = self.x_s
        else:
            x_s = transform_keypoint(x_s_info)
            if self.is_image_flag:
                self.x_s = x_s
        
        x_d = transform_keypoint(x_d_info)
        
        if self.flag_stitching:
            x_d = self.stitch_net(x_s, x_d)

        self.idx += 1

        return x_s, x_d
