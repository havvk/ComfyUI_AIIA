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

StreamSDK = object # Default fallback to prevent NameError if import fails
DITTO_AVAILABLE = False

try:
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
        
    def setup(self, source_image_pil, **kwargs):
        # Override setup to accept PIL Image instead of path
        
        # ======== Prepare Options ========
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)

        # -- avatar_registrar: template cfg --
        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)
        
        crop_kwargs = {
            "crop_scale": getattr(self, 'crop_scale', 2.3),
            "crop_vx_ratio": getattr(self, 'crop_vx_ratio', 0),
            "crop_vy_ratio": getattr(self, 'crop_vy_ratio', -0.125),
            "crop_flag_do_rot": getattr(self, 'crop_flag_do_rot', True),
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
        self.writer_pbar = MockWriter()
        
        self.generated_frames = []
        
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
            # self.writer_pbar.update()

    def cleanup(self):
        # Close SDK but keep threads alive? 
        # StreamSDK.close() joins threads. We can call close() then create new SDK for next run?
        # Or just clear queues.
        # For safety/simplicity in ComfyUI (stateless nodes), we might want to kill it.
        # But loading models takes time.
        # Models are in self.avatar_registrar, self.audio2motion, etc.
        # If we `close()`, threads die. We cannot restart threads on same object easily because thread.start() only once.
        # So we should probably keep SDK alive but define a 'reset' method?
        # But SDK structure couples Threads with Init.
        # So: Re-instantiate SDK every run?
        # If models are loaded in __init__, re-instantiation re-loads models. BAD.
        # Models are loaded in __init__.
        # Optimizing:
        # We should separate Model Loading from Thread Starting.
        # But that requires modifying StreamSDK code.
        # Allow me to modify StreamSDK via monkeypatch or rewrite.
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
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "generate"
    CATEGORY = "AIIA/Ditto"

    def generate(self, pipe, ref_image, audio, sampling_steps, fps):
        # pipe is the dict we returned in Loader
        # Create a FRESH SDK instance for each run because threads die after run.
        # To reuse models, we need to hack StreamSDK.
        # BUT, Ditto models are small (~few hundred MB?). Maybe acceptable to reload?
        # No, Load time is 5-10s. Usability suffers.
        # Strategy:
        # Load models ONCE (in Loader) and pass them to Sampler?
        # StreamSDK owns the models.
        # We can create a new SDK instance but share the underlying model objects?
        # StreamSDK init: self.lmdm = LMDM(...)
        # We can extract models from 'pipe["sdk"]' (the master instance) and inject them into a new 'RuntimeSDK'.
        
        master_sdk = pipe["sdk"]
        cfg_pkl = pipe["cfg_pkl"]
        data_root = pipe["data_root"]
        
        # Create a new Runtime SDK that shares models with Master SDK
        # We need to modify StreamSDK to support 'preloaded_models' argument.
        # Or we can just spin up new threads on the Master SDK?
        # master_sdk.thread_list are dead if previous run finished.
        # We need to re-create threads.
        
        # Let's try to Restart Threads on master_sdk:
        # Threads cannot be restarted. We must create new Thread objects targeting the SAME worker methods.
        import threading
        
        master_sdk.stop_event.clear()
        master_sdk.generated_frames = []
        master_sdk.worker_exception = None
        
        # Reset Queues
        import queue
        master_sdk.audio2motion_queue = queue.Queue(maxsize=100)
        master_sdk.motion_stitch_queue = queue.Queue(maxsize=100)
        master_sdk.warp_f3d_queue = queue.Queue(maxsize=100)
        master_sdk.decode_f3d_queue = queue.Queue(maxsize=100)
        master_sdk.putback_queue = queue.Queue(maxsize=100)
        master_sdk.writer_queue = queue.Queue(maxsize=100)
        
        # Reset logic states
        master_sdk.clip_idx = 0 # audio2motion
        # But logic state is inside atomic components too? 
        # e.g. self.audio2motion.clip_idx...
        # We should call .setup() again?
        # YES, implementation below calls .setup() which seems to reset components.
        
        # Start new threads
        master_sdk.thread_list = [
            threading.Thread(target=master_sdk.audio2motion_worker),
            threading.Thread(target=master_sdk.motion_stitch_worker),
            threading.Thread(target=master_sdk.warp_f3d_worker),
            threading.Thread(target=master_sdk.decode_f3d_worker),
            threading.Thread(target=master_sdk.putback_worker),
            threading.Thread(target=master_sdk.writer_worker),
        ]
        for t in master_sdk.thread_list:
            t.start()
            
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
        # Calculate Number of Frames based on Audio Duration
        # Ditto standard FPS is 25?
        # math.ceil(len(audio) / 16000 * 25) in inference.py
        import math
        # If user provides FPS, we must adapt.
        # But Ditto model might be trained on 25fps fixed?
        # LMDM seq_frames=80.
        # Let's enforce 25fps for now as Ditto seems tuned for it.
        # If user wants 60fps, we might need to interpolation output.
        # Let's use user fps to calculate N_f, but warn if model is 25-fixed.
        # The paper says 40FPS/Realtime, but generated video FPS depends on how dense the motion keys are.
        # Let's stick to 25 for safe start.
        
        target_fps = 25 # Force 25 for stability first
        num_frames = math.ceil(len(audio_np) / 16000 * target_fps)
        
        master_sdk.setup(
            source_image_pil=ref_image_pil, 
            output_path=None, # In-memory
            N_d=num_frames,
            sampling_timesteps=sampling_steps
        )
        
        # 4. Trigger Audio Feat Extraction & Pipeline
        # In StreamSDK.run_chunk or offline mode logic.
        # We need to inspect how 'run' starts.
        # In inference.py: SDK.wav2feat... then SDK.audio2motion_queue.put(aud_feat)
        # We replicate inference.py logic here.
        
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
