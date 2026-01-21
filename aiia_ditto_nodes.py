import os
import sys
import torch
import numpy as np
from PIL import Image
import folder_paths
import logging

# Add Ditto library to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ditto_path = os.path.join(current_dir, "libs", "Ditto")
if ditto_path not in sys.path:
    sys.path.append(ditto_path)

# --- Logging Protection ---
import logging
class RestoreLogging:
    def __enter__(self):
        self.saved_handlers = logging.root.handlers[:]
        self.saved_stdout = sys.stdout
        self.saved_stderr = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. Restore sys streams
        if sys.stdout != self.saved_stdout:
            sys.stdout = self.saved_stdout
        if sys.stderr != self.saved_stderr:
            sys.stderr = self.saved_stderr
            
        # 2. Restore Logging Handlers
        # ABSL and other libs often add a StreamHandler to stderr.
        # We want to revert to the exact set of handlers we had before.
        
        # Identify new handlers
        current_handlers = logging.root.handlers[:]
        for h in current_handlers:
            if h not in self.saved_handlers:
                # This is a new handler added during the block. Remove it.
                # Common culprit: absl.logging.ABSLHandler
                logging.root.removeHandler(h)
                
        # Restore missing handlers
        for h in self.saved_handlers:
            if h not in logging.root.handlers:
                logging.root.addHandler(h)
                
        # 3. Force Level to INFO
        # absl often sets it to FATAL or something high.
        logging.root.setLevel(logging.INFO)
# --------------------------

StreamSDK = object # Default fallback to prevent NameError if import fails
DITTO_AVAILABLE = False

try:
    with RestoreLogging():
        # Import directly since we added ditto_path to sys.path
        # This avoids assuming 'libs' is a resolvable package
        from stream_pipeline_offline import StreamSDK
        from core.atomic_components.cfg import parse_cfg
    DITTO_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import Ditto libs: {e}")
    # Keep StreamSDK as object so class ComfyStreamSDK(StreamSDK) doesn't crash
    pass
except Exception as e:
    logging.error(f"Unexpected error importing Ditto: {e}")
    pass

from huggingface_hub import snapshot_download

class AIIA_DittoLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["ditto-talkinghead"],),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("DITTO_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "AIIA/Ditto"

    def load_model(self, model_name, device):
        print(f"[AIIA_DittoLoader] Loading Ditto model: {model_name} on {device}")
        
        # 1. Prepare Model Paths
        base_path = os.path.join(folder_paths.models_dir, "ditto")
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
            
        # Check for nested structure (common with hf download)
        # We look for 'ditto_pytorch' inside 'ditto-talkinghead' subdir first, then in base.
        nested_base = os.path.join(base_path, "ditto-talkinghead")
        
        if os.path.exists(os.path.join(nested_base, "ditto_pytorch")):
             final_model_root = nested_base
        else:
             final_model_root = base_path

        model_dir = os.path.join(final_model_root, "ditto_pytorch")
        cfg_dir = os.path.join(final_model_root, "ditto_cfg")
        
        # 2. Check and Download if missing
        if not os.path.exists(model_dir) or not os.path.exists(cfg_dir):
            print(f"[AIIA_DittoLoader] Model not found at {model_dir}. Downloading from HuggingFace...")
            snapshot_download(
                repo_id="digital-avatar/ditto-talkinghead",
                local_dir=base_path, # Download to base (ditto/) which creates ditto/.huggingface AND ditto/ditto_pytorch etc OR ditto/ditto-talkinghead depending on args
                # snapshot_download usually flattens if local_dir provided, UNLESS accessing a folder in repo?
                # Actually, HF snapshot_download by default structures as tree.
                allow_patterns=["ditto_pytorch/*", "ditto_cfg/*"],
                local_dir_use_symlinks=False
            )
            # Recheck paths after download
            if os.path.exists(os.path.join(base_path, "ditto-talkinghead", "ditto_pytorch")):
                final_model_root = os.path.join(base_path, "ditto-talkinghead")
            else:
                final_model_root = base_path
            
            model_dir = os.path.join(final_model_root, "ditto_pytorch")
            cfg_dir = os.path.join(final_model_root, "ditto_cfg")
            
            print("[AIIA_DittoLoader] Download complete.")
            
        print(f"[AIIA_DittoLoader] Model Dir: {model_dir}")
        print(f"[AIIA_DittoLoader] Config Dir: {cfg_dir}")
        
        # 3. Initialize SDK
        # We need to construct the cfg_pkl path
        cfg_pkl = os.path.join(cfg_dir, "v0.4_hubert_cfg_pytorch.pkl")
        data_root = model_dir # ditto_pytorch IS the data_root expected by Ditto?
        # StreamSDK uses data_root to find 'aux_models', 'models' etc.
        # Yes, based on ditto_pytorch structure.
        
        if not os.path.exists(cfg_pkl):
             raise FileNotFoundError(f"Config file not found: {cfg_pkl}")
        
        # Initialize StreamSDK in offline mode
        # NOTE: StreamSDK.__init__ creates worker threads.
        # We might want to delay initialization or manage it carefully.
        # For now, we init it here.
        
        # Adjust sys.path for internal imports within Ditto to work (it imports 'core.xxx')
        # Ditto expects to be imported as 'libs.Ditto...' or we need to be careful.
        # StreamSDK uses 'from core.atomic_components...' which implies 'core' must be top level or relative.
        # Wait, 'libs/Ditto/stream_pipeline_offline.py' has 'from core...'.
        # If we added 'libs/Ditto' to sys.path, then 'import core' works.
        
        
        # Initialize ComfyStreamSDK (Subclass of StreamSDK that avoids File I/O for frames)
        try:
            # Also wrap this instantiation because StreamSDK.__init__ calls parse_cfg
            # which might also trigger logging config changes if not careful.
            with RestoreLogging():
                sdk = ComfyStreamSDK(cfg_pkl, data_root)
        except Exception as e:
            logging.error(f"Failed to initialize Ditto SDK: {e}")
            raise e

        # We need to manually load the models here if they aren't loaded by __init__?
        # SDK.__init__ calls parse_cfg -> setups all components.
        # But wait, SDK.__init__ starts threads immediately! 
        # Ideally we want to START threads only when generating.
        # But StreamSDK design is "init = start threads".
        # We can keep the SDK instance alive in the pipeline.
        
        pipeline = {
            "sdk": sdk,
            "device": device,
            "cfg_pkl": cfg_pkl,
            "data_root": data_root
        }
        
        return (pipeline,)

class MockWriter:
    def close(self): pass
    def update(self, *args): pass
    def __call__(self, *args, **kwargs): pass

class ComfyStreamSDK(StreamSDK):
    """
    Subclass of StreamSDK to support in-memory frame capture and PIL input.
    """
    def __init__(self, cfg_pkl, data_root, **kwargs):
        # We delay thread start or we override methods to handle in-memory data
        super().__init__(cfg_pkl, data_root, **kwargs)
        self.generated_frames = [] # Store result frames here
        
        # NOTE: StreamSDK.__init__ does NOT start threads or create queues (they are created in setup).
        # So we do NOT need to call self.close() here. Calling it causes AttributeError because queues don't exist yet.
        # We just leave it as is. setup() will be called by the Sampler node.
        pass
        
    def setup(self, source_image_pil, total_frames=0, **kwargs):
        # Override setup to accept PIL Image instead of path
        
        # ======== Prepare Options ========
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)

        print("=" * 20, "ComfyStreamSDK setup", "=" * 20)
        # print_cfg not imported, skip
        
        # -- avatar_registrar: template cfg --
        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)

        # -- avatar_registrar: crop cfg --
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        
        # -- avatar_registrar: smo for video --
        self.smo_k_s = kwargs.get('smo_k_s', 13)

        # -- condition_handler: ECS --
        self.emo = kwargs.get("emo", 4)    # int | [int] | [[int]] | numpy
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)    # for video
        self.ch_info = kwargs.get("ch_info", None)    # dict of np.ndarray

        # -- audio2motion: setup --
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)  # [ds,de]
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)

        # -- motion_stitch: setup --
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        # drive_eye might be passed in kwargs
        self.drive_eye = kwargs.get("drive_eye", None)    # None: true4image, false4video
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")    # "" | "d0" | "s"
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)

        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        self.wav2feat = self.wav2feat # ensure exist? (Initialized in __init__)
        
        # Assert online mode support
        # assert self.wav2feat.support_streaming or not self.online_mode

        crop_kwargs = {
            "crop_scale": self.crop_scale,
            "crop_vx_ratio": self.crop_vx_ratio,
            "crop_vy_ratio": self.crop_vy_ratio,
            "crop_flag_do_rot": self.crop_flag_do_rot,
        }
        n_frames = self.template_n_frames if self.template_n_frames > 0 else self.N_d
        
        # Save temp file for AvatarRegistrar
        import tempfile
        import cv2
        
        # Convert PIL to BGR for cv2
        img_np = np.array(source_image_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Use a temporary directory that cleans up automatically
        # Note: We need to keep this dir alive during setup
        if not hasattr(self, 'temp_dir'):
            self.temp_dir = tempfile.TemporaryDirectory()
            
        temp_img_path = os.path.join(self.temp_dir.name, "ref.png")
        cv2.imwrite(temp_img_path, img_bgr)
        
        source_info = self.avatar_registrar(
            temp_img_path, 
            max_dim=self.max_size, 
            n_frames=n_frames, 
            **crop_kwargs,
        )

        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            from libs.Ditto.core.atomic_components.avatar_registrar import smooth_x_s_info_lst
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(source_info["x_s_info_lst"], smo_k=self.smo_k_s)

        self.source_info = source_info
        self.source_info_frames = len(source_info["x_s_info_lst"])

        # ======== Setup Condition Handler ========
        self.condition_handler.setup(source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info)

        # ======== Setup Audio2Motion (LMDM) ========
        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0, 
            overlap_v2=self.overlap_v2,
            fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim,
            sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode,
            v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )

        # ======== Setup Motion Stitch ========
        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info['x_s_info_lst'][0]
        self.motion_stitch.setup(
            N_d=self.N_d,
            use_d_keys=self.use_d_keys,
            relative_d=self.relative_d,
            drive_eye=self.drive_eye,
            delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n,
            fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type,
            flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag,
            x_s_info=x_s_info,
            d0=None,
            ch_info=self.ch_info,
            overall_ctrl_info=self.overall_ctrl_info,
        )

        # ======== Video Writer Bypass ========
        # Mock the writer components so SDK.close() doesn't fail
        self.writer = MockWriter()
        self.writer_pbar = MockWriter() # We replace this with our own logic
        self.generated_frames = []
        
        # Prepare Progress Bar
        from comfy.utils import ProgressBar
        if total_frames > 0:
            self.pbar = ProgressBar(total_frames)
        else:
            self.pbar = None
            
        # ======== Setup queues and threads (Copied from StreamSDK.setup) ========
        # We need these because we are starting fresh threads every setup()
        import queue
        import threading
        
        QUEUE_MAX_SIZE = 100
        self.worker_exception = None
        self.stop_event = threading.Event()

        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.writer_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        
        # Reset logic states/buffers
        if not self.online_mode:
            self.audio_feat = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
        self.cond_idx_start = 0 - len(self.audio_feat)
        self.clip_idx = 0 

        self.thread_list = [
            threading.Thread(target=self.audio2motion_worker),
            threading.Thread(target=self.motion_stitch_worker),
            threading.Thread(target=self.warp_f3d_worker),
            threading.Thread(target=self.decode_f3d_worker),
            threading.Thread(target=self.putback_worker),
            threading.Thread(target=self.writer_worker),
        ]

        for thread in self.thread_list:
            thread.start()
        
    def _writer_worker(self):
        # Override to append to list instead of writing to disk
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except Exception: # queue.Empty
                continue

            if item is None:
                break
            
            res_frame_rgb = item # This is numpy RGB array usually
            self.generated_frames.append(res_frame_rgb)
            
            # Explicit Console Logging for ComfyUI
            count = len(self.generated_frames)
            if count % 10 == 0:
                print(f"[Ditto] Processing frame {count}...")
                
            if self.pbar:
                self.pbar.update(1)

    def cleanup(self):
        pass

class AIIA_DittoSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("DITTO_PIPE",),
                "ref_image": ("IMAGE",),
                "audio": ("AUDIO",),
                "sampling_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "fps": ("INT", {"default": 25, "min": 15, "max": 60}),
                "crop_scale": ("FLOAT", {"default": 2.3, "min": 1.0, "max": 5.0, "step": 0.1}),
                "emo": (["Neutral", "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Contempt"], {"default": "Neutral"}),
                "drive_eye": ("BOOLEAN", {"default": True}),
                "chk_eye_blink": ("BOOLEAN", {"default": True}),
                "smo_k_d": ("INT", {"default": 3, "min": 1, "max": 9}),
                "hd_rot_p": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0, "step": 1.0}),
                "hd_rot_y": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0, "step": 1.0}),
                "hd_rot_r": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0, "step": 1.0}),
                "mouth_amp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "relax_on_silence": ("BOOLEAN", {"default": True, "label_on": "Relax Face on Silence", "label_off": "Disabled"}),
                "ref_threshold": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 1.0, "step": 0.001}),
                "blink_mode": (["Random (Normal)", "Fast", "Slow", "None"], {"default": "Random (Normal)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "generate"
    CATEGORY = "AIIA/Ditto"

    def generate(self, pipe, ref_image, audio, sampling_steps, fps, crop_scale, emo, drive_eye, chk_eye_blink, smo_k_d, hd_rot_p, hd_rot_y, hd_rot_r, mouth_amp, relax_on_silence, ref_threshold, blink_mode):
        # pipe is the dict we returned in Loader
        master_sdk = pipe["sdk"]
        cfg_pkl = pipe["cfg_pkl"]
        data_root = pipe["data_root"]
        
        # 1. Prepare Audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        import torchaudio
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform_16k = resampler(waveform)
        else:
            waveform_16k = waveform
        if waveform_16k.shape[0] > 1:
            waveform_16k = torch.mean(waveform_16k, dim=0, keepdim=True) # Mono
        audio_np = waveform_16k.squeeze().cpu().numpy()
        
        # 2. Prepare ref image (PIL)
        ref_image_np = (ref_image[0].cpu().numpy() * 255).astype(np.uint8)
        ref_image_pil = Image.fromarray(ref_image_np)
        
        # 3. Setup SDK with this run's data
        import math
        
        target_fps = 25 # Force 25 for stability first
        num_frames = math.ceil(len(audio_np) / 16000 * target_fps)
        
        # VAD / Volume Analysis
        ctrl_info = {}
        if relax_on_silence:
            # Calculate RMS per frame
            frame_len = 640 # 16000 / 25
            
            # Simple RMS calculation
            # Pad audio if needed
            pad_len = num_frames * frame_len - len(audio_np)
            if pad_len > 0:
                 audio_proc = np.pad(audio_np, (0, pad_len))
            else:
                 audio_proc = audio_np[:num_frames * frame_len]
                 
            # Reshape to (num_frames, frame_len)
            audio_frames = audio_proc.reshape(num_frames, frame_len)
            rms = np.sqrt(np.mean(audio_frames**2, axis=1))
            
            # Create alpha mask (0.0 = Silence/Ref, 1.0 = Speech/Gen)
            # Apply Attack/Release Envelope
            # Attack (Silence -> Speech): Fast (e.g. 0.05s / ~1-2 frames)
            # Release (Speech -> Silence): Slow (e.g. 0.3-0.5s / ~8-12 frames)
            
            target_alpha = np.zeros(num_frames, dtype=np.float32)
            for i in range(num_frames):
                target_alpha[i] = 0.0 if rms[i] < ref_threshold else 1.0
            
            current_alpha = 1.0 # Assume start with speech-ready state or 0.0? 
            # If user starts silent, 0.0 is better, but let's assume 1.0 to avoid initial fade-in lag.
            # Actually, let's start at target[0].
            current_alpha = target_alpha[0]
            
            dataset_alpha = np.zeros(num_frames, dtype=np.float32)
            
            # Coefficients
            # alpha_new = alpha_old * coeff + target * (1 - coeff)
            # coeff = exp(-dt / tau)
            # dt = 1/25 = 0.04s
            # Attack tau ~ 0.05s -> coeff ~ 0.45
            # Release tau ~ 0.4s -> coeff ~ 0.90
            
            # Coefficients
            # alpha_new = alpha_old * coeff + target * (1 - coeff)
            # Attack: Instant (React immediately to voice)
            # Release: Linear/Exponential decay
            
            # rel_coeff = 0.80 -> 20% decay per frame (approx 8-10 frames / 0.4s to close)
            # This is faster than 0.90 (which took ~2s) and feels more responsive.
            rel_coeff = 0.80
            
            for i in range(num_frames):
                target = target_alpha[i]
                if target > current_alpha:
                    # Attack (Rising) - Instant
                    current_alpha = target
                else:
                    # Release (Falling) - Smoothed
                    current_alpha = current_alpha + (target - current_alpha) * (1.0 - rel_coeff)
                
                dataset_alpha[i] = current_alpha
                
            # Log VAD stats for debugging
            non_silence_count = np.count_nonzero(target_alpha)
            logging.info(f"[Ditto] VAD Stats: {non_silence_count}/{num_frames} frames active. RMS Mean: {np.mean(rms):.4f}, Min: {np.min(rms):.4f}, Max: {np.max(rms):.4f}")
            
            # 3. Populate ctrl_info with VAD Alpha and Micro-Motion
            # Micro-Motion: Inject subtle head sway during silence to prevent "dead static" look.
            # Only applied when alpha < 1.0.
            
            idle_amp = 0.5 # Degrees
            
            for i in range(num_frames):
                alpha = float(dataset_alpha[i])
                # Clip just in case
                alpha = max(0.0, min(1.0, alpha))
                
                info_dict = {}
                
                # VAD Alpha (Mouth Control)
                if alpha < 0.999:
                     info_dict["vad_alpha"] = alpha
                
                # Idle Micro-Motion (Head Control)
                # Blend in motion as alpha decreases (silence increases)
                # We use (1.0 - alpha) as the weight for idle motion.
                if alpha < 1.0:
                    idle_weight = (1.0 - alpha)
                    
                    # Periodic sway
                    # t = i / 25.0
                    # Pitch: slower, cos wave
                    # Yaw: slightly faster, sin wave
                    t = i / 25.0
                    d_pitch = math.cos(t * 1.5) * idle_amp * idle_weight
                    d_yaw = math.sin(t * 1.2) * idle_amp * idle_weight
                    
                    # We need to ADD this to the global controls (hd_rot_p, etc.)
                    # But ctrl_info overrides per frame.
                    # Since we want to ADD to the global setting, we must include the global base + offset.
                    # Wait, MotionStitch typically MERGES: default_kwargs filled first, then run_kwargs override.
                    # So if we put 'delta_pitch' here, it OVERRIDES the global one.
                    # So we must add global + offset here.
                    
                    info_dict["delta_pitch"] = hd_rot_p + d_pitch
                    info_dict["delta_yaw"] = hd_rot_y + d_yaw
                    
                if info_dict:
                     ctrl_info[i] = info_dict
        
        # Blink Settings
        delta_eye_open_n = 0 if chk_eye_blink else -1
        blink_min = 60
        blink_max = 100
        
        if blink_mode == "Fast":
             blink_min = 10
             blink_max = 40
        elif blink_mode == "Slow":
             blink_min = 120
             blink_max = 200
        elif blink_mode == "None":
             delta_eye_open_n = -1
        
        # Map emo string to int
        emo_map = {
            "Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3,
            "Neutral": 4, "Sad": 5, "Surprise": 6, "Contempt": 7
        }
        emo_idx = emo_map.get(emo, 4)
        
        # Prepare Controls
        overall_ctrl_info = {
            "delta_pitch": hd_rot_p,
            "delta_yaw": hd_rot_y,
            "delta_roll": hd_rot_r,
            "mouth_amp": mouth_amp,
        }
        
        
        # Calling setup() creates fresh threads and queues
        # Wrap setup() with RestoreLogging to catch any init-time hijacking (e.g. MediaPipe/absl)
        # and restore it IMMEDIATELY before we start the long-running inference.
        with RestoreLogging():
            master_sdk.setup(
                source_image_pil=ref_image_pil, 
                output_path=None, # In-memory
                N_d=num_frames,
                sampling_timesteps=sampling_steps,
                crop_scale=crop_scale,
                emo=emo_idx,
                drive_eye=drive_eye,
                delta_eye_open_n=delta_eye_open_n, # 0=random, -1=none
                blink_interval_min=blink_min,
                blink_interval_max=blink_max,
                smo_k_d=smo_k_d,
                overall_ctrl_info=overall_ctrl_info,
                ctrl_info=ctrl_info,
                total_frames=num_frames # For pbar
            )
            
        # 4. Trigger Audio Feat Extraction & Pipeline
        try:
             aud_feat = master_sdk.wav2feat.wav2feat(audio_np)
             master_sdk.audio2motion_queue.put(aud_feat)
             
             # 5. Wait for completion
             # master_sdk.close() joins threads.
             master_sdk.close() 
        except Exception as e:
             logging.error(f"Error during Ditto inference: {e}")
             # Ensure cleanup
             try: master_sdk.close()
             except: pass
             raise e
             
        # 6. Retrieve frames
        generated = master_sdk.generated_frames
        if not generated:
            raise RuntimeError("Ditto generated 0 frames.")
            
        # Convert List[np.array (H,W,C)] -> Batch Tensor (B,H,W,C)
        # Note: generated frames are RGB (from writer_queue which usually gets RGB).
        import torch
        video_tensor = torch.from_numpy(np.array(generated)).float() / 255.0
        
        return (video_tensor, audio)


NODE_CLASS_MAPPINGS = {
    "AIIA_DittoLoader": AIIA_DittoLoader,
    "AIIA_DittoSampler": AIIA_DittoSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_DittoLoader": "AIIA Ditto Loader",
    "AIIA_DittoSampler": "AIIA Ditto Sampler"
}
