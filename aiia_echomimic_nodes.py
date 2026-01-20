import os
import sys
import torch
import folder_paths
import comfy.model_management as mm
import logging
import numpy as np
import math
import gc
from PIL import Image
import torchaudio

# Configure Logging
logger = logging.getLogger("ComfyUI_AIIA_EchoMimic")
import torchvision.transforms.functional as TF

# --- Path Setup ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
echomimic_v3_root = os.path.join(current_dir, "libs", "EchoMimicV3")

# Check if EchoMimicV3 exists
if not os.path.exists(echomimic_v3_root):
    logger.error(f"EchoMimicV3 root directory not found at: {echomimic_v3_root}")
    ECHOMIMIC_AVAILABLE = False
else:
    # Add to sys.path to allow 'from echomimic_v3_src import ...' to work as expected by the repo code
    if echomimic_v3_root not in sys.path:
        sys.path.insert(0, echomimic_v3_root)
    ECHOMIMIC_AVAILABLE = True

# --- Wrapper for Lazy Import ---
# --- Wrapper for Lazy Import ---
IMPORT_ERROR_MSG = ""
if ECHOMIMIC_AVAILABLE:
    try:
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
        from diffusers import FlowMatchEulerDiscreteScheduler
        from echomimic_v3_src.fm_solvers import FlowDPMSolverMultistepScheduler
        
        # EchoMimic Internal Modules
        from echomimic_v3_src.wan_vae import AutoencoderKLWan
        from echomimic_v3_src.wan_image_encoder import CLIPModel
        from echomimic_v3_src.wan_text_encoder import WanT5EncoderModel
        from echomimic_v3_src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
        from echomimic_v3_src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
        from echomimic_v3_src.utils import get_image_to_video_latent3, filter_kwargs
        from echomimic_v3_src.face_detect import get_mask_coord
        from echomimic_v3_src.cache_utils import get_teacache_coefficients
    except ImportError as e:
        IMPORT_ERROR_MSG = str(e)
        logger.error(f"Failed to import EchoMimicV3 modules: {e}")
        ECHOMIMIC_AVAILABLE = False
    except Exception as e:
        IMPORT_ERROR_MSG = f"Unexpected error during import: {e}"
        logger.error(f"Unexpected error importing EchoMimicV3 modules: {e}")
        ECHOMIMIC_AVAILABLE = False

# --- Constants & Config ---
ECHOMIMIC_MODELS_DIR = "EchoMimicV3" # Expects models under ComfyUI/models/EchoMimicV3

# --- Helper Functions (Adapted from infer.py) ---
def get_sample_size(image, default_size):
    width, height = image.size
    original_area = width * height
    default_area = default_size[0] * default_size[1]
    if default_area < original_area:
        ratio = math.sqrt(original_area / default_area)
        width = width / ratio // 16 * 16
        height = height / ratio // 16 * 16
    else:
        width = width // 16 * 16
        height = height // 16 * 16
    return int(height), int(width)

def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    mask = mask.reshape(-1)
    return mask.float()

# --- Nodes ---

class AIIA_EchoMimicLoader:
    NODE_NAME = "EchoMimic V3 Loader"
    CATEGORY = "AIIA/EchoMimic"
    FUNCTION = "load_model"
    RETURN_TYPES = ("ECHOMIMIC_PIPE",)
    RETURN_NAMES = ("pipe",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_subfolder": ("STRING", {"default": "EchoMimicV3", "tooltip": "Subfolder in models/EchoMimicV3 containing the main models"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
            }
        }

    def load_model(self, model_subfolder, precision, device):
        if not ECHOMIMIC_AVAILABLE:
            raise ImportError(f"EchoMimicV3 modules not loaded. Error: {IMPORT_ERROR_MSG}. Please check dependencies and libs/EchoMimicV3.")

        model_root = os.path.join(folder_paths.models_dir, ECHOMIMIC_MODELS_DIR, model_subfolder)
        if not os.path.exists(model_root):
            # Fallback check standard diffusers path
            model_root_alt = os.path.join(folder_paths.models_dir, "diffusers", model_subfolder)
            if os.path.exists(model_root_alt):
                model_root = model_root_alt
            else:
                raise FileNotFoundError(f"Model directory not found at {model_root} or {model_root_alt}")

        weight_dtype = torch.float32
        if precision == "fp16": weight_dtype = torch.float16
        elif precision == "bf16": weight_dtype = torch.bfloat16

        print(f"[{self.NODE_NAME}] Loading EchoMimicV3 models from {model_root} ({precision})...")
        
        # Load Config
        config_path = os.path.join(echomimic_v3_root, "config", "config.yaml")
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Config file not found at {config_path}")
        
        cfg = OmegaConf.load(config_path)

        # Define fallback base directory (Wan2.1-Fun)
        wan_base_dir = os.path.join(folder_paths.models_dir, ECHOMIMIC_MODELS_DIR, "Wan2.1-Fun-V1.1-1.3B-InP")
        
        def get_component_path(subpath, required=True, allow_fallback=True):
            # 1. Check current model_root (EchoMimicV3 directory)
            p = os.path.join(model_root, subpath)
            if os.path.exists(p):
                return p
            
            # 2. Check Wan2.1 Base directory
            if allow_fallback and os.path.exists(wan_base_dir):
                p_base = os.path.join(wan_base_dir, subpath)
                if os.path.exists(p_base):
                    print(f"[{self.NODE_NAME}] Found {subpath} in Wan2.1 Base: {p_base}")
                    return p_base

            if required:
                raise FileNotFoundError(f"Component '{subpath}' not found in {model_root} or {wan_base_dir}")
            return None

        # Transformer (Must be from EchoMimicV3, strict check preferred, but let's allow fallback if user put it elsewhere)
        # Actually, for Transformer, we want the EchoMimic one. If user is using Base Wan2.1, they get bad results.
        # But we rely on config.yaml 'transformer_subpath' which is 'transformer'.
        print(f"[{self.NODE_NAME}] Loading Transformer...")
        transformer_subpath = cfg['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')
        transformer_path = get_component_path(transformer_subpath, required=True)
        
        transformer = WanTransformerAudioMask3DModel.from_pretrained(
            transformer_path,
            transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True
        ).to("cpu")

        # VAE
        print(f"[{self.NODE_NAME}] Loading VAE...")
        vae_subpath = cfg['vae_kwargs'].get('vae_subpath', 'Wan2.1_VAE.pth')
        vae_path = get_component_path(vae_subpath, required=True)
        vae = AutoencoderKLWan.from_pretrained(
            vae_path,
            additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
        ).to(dtype=torch.float32, device="cpu")

        # Tokenizer
        print(f"[{self.NODE_NAME}] Loading Tokenizer...")
        tokenizer_subpath = cfg['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')
        # Tokenizer is a directory usually
        tokenizer_path = get_component_path(tokenizer_subpath, required=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Text Encoder
        print(f"[{self.NODE_NAME}] Loading Text Encoder...")
        text_encoder_subpath = cfg['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')
        text_encoder_path = get_component_path(text_encoder_subpath, required=True)
        text_encoder = WanT5EncoderModel.from_pretrained(
            text_encoder_path,
            additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
            torch_dtype=weight_dtype,
            low_cpu_mem_usage=True
        ).to(dtype=weight_dtype, device="cpu").eval()

        # Image Encoder
        print(f"[{self.NODE_NAME}] Loading Image Encoder...")
        image_encoder_subpath = cfg['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')
        image_encoder_path = get_component_path(image_encoder_subpath, required=True)
        clip_image_encoder = CLIPModel.from_pretrained(
            image_encoder_path
        ).to(dtype=weight_dtype, device="cpu").eval()

        # Scheduler
        print(f"[{self.NODE_NAME}] Loading Scheduler...")
        scheduler_kwargs = OmegaConf.to_container(cfg['scheduler_kwargs'])
        # infer.py uses a dict mapping for scheduler class selection.
        # "Flow" -> FlowMatchEulerDiscreteScheduler
        # "Flow_Unipc" -> FlowUniPCMultistepScheduler
        # "Flow_DPM++" -> FlowDPMSolverMultistepScheduler
        # We'll default to FlowMatchEulerDiscreteScheduler or allow selection TODO.
        # For now, let's use FlowMatchEulerDiscreteScheduler as base or check config/defaults.
        # infer.py defaults to "Flow" in Config class, but app.py uses "Flow_DPM++".
        # Let's use FlowMatchEulerDiscreteScheduler as safe default, or DPM++ if preferred.
        # Let's try to infer or just use FlowDPMSolverMultistepScheduler as it seems better.
        # Actually, let's look at `libs/EchoMimicV3/echomimic_v3_src/fm_solvers.py` availablity.
        # For now, we stick to FlowMatchEulerDiscreteScheduler as commonly imported, or better:
        from echomimic_v3_src.fm_solvers import FlowDPMSolverMultistepScheduler
        scheduler = FlowDPMSolverMultistepScheduler(**filter_kwargs(FlowDPMSolverMultistepScheduler, scheduler_kwargs))

        # Check Optimization libraries
        try:
            import flash_attn
            print(f"[{self.NODE_NAME}] Flash Attention (v2) is available.")
        except ImportError:
            print(f"[{self.NODE_NAME}] Flash Attention is NOT available. Using SDPA/Math fallback (slower).")


        # Wav2Vec (Assume it's in a separate standard folder or specified)
        # For now, we expect it to be in 'models/EchoMimicV3/wav2vec2-base-960h' or similar
        # But commonly these are downloaded from HF. Let's try to find it.
        wav2vec_path = os.path.join(folder_paths.models_dir, ECHOMIMIC_MODELS_DIR, "wav2vec2-base-960h")
        if not os.path.exists(wav2vec_path):
             wav2vec_path = "facebook/wav2vec2-base-960h" # fallback to HF hub
        
        # Debug: Check where models are actually located
        print(f"[{self.NODE_NAME}] --- Model Devices Info ---")
        print(f"  Transformer: {transformer.device}")
        print(f"  VAE: {vae.device}")
        print(f"  Text Encoder: {text_encoder.device}")
        print(f"  Image Encoder: {clip_image_encoder.device}")
        print(f"------------------------------------------")
        
        print(f"[{self.NODE_NAME}] Loading Audio Encoder from {wav2vec_path}...")
        wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
        wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path).to(dtype=weight_dtype, device="cpu").eval()
        
        print(f"[{self.NODE_NAME}]   Wav2Vec Model: {wav2vec_model.device}")

        # Pipeline Construction
        pipeline = WanFunInpaintAudioPipeline(
            transformer=transformer,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        
        # Manual Device Management: Load everything on CPU.
        # We will manually move models to GPU in the Sampler node to optimize VRAM usage.
        pipeline.to("cpu")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        gc.collect()
        torch.cuda.empty_cache()

        pipe_data = {
            "pipeline": pipeline,
            "device": device,
            "weight_dtype": weight_dtype,
            "wav2vec_processor": wav2vec_processor,
            "wav2vec_model": wav2vec_model, # Keep on CPU, move to device only when needed
        }
        
        print(f"[{self.NODE_NAME}] Models loaded successfully.")
        return (pipe_data,)

class AIIA_EchoMimicSampler:
    NODE_NAME = "EchoMimic V3 Sampler"
    CATEGORY = "AIIA/EchoMimic"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ECHOMIMIC_PIPE",),
                "ref_image": ("IMAGE",), # (1, H, W, 3)
                "ref_audio": ("AUDIO",),
                "prompt": ("STRING", {"multiline": True, "default": "best quality, high quality, 8k, realistic, photorealistic, details, sharp focus"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. 手部快速摆动, 手指频繁抽搐, 夸张手势, 重复机械性动作."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0}),
                "audio_cfg": ("FLOAT", {"default": 2.9, "min": 1.0, "max": 20.0}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 60}),
            },
            "optional": {
                "width": ("INT", {"default": 768}),
                "height": ("INT", {"default": 768}),
                "context_length": ("INT", {"default": 49, "min": 16, "max": 200, "step": 1}),
                "enable_teacache": ("BOOLEAN", {"default": False}),
                "teacache_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process(self, pipe, ref_image, ref_audio, prompt, negative_prompt, seed, steps, cfg, audio_cfg, fps, width=768, height=768, context_length=49, enable_teacache=False, teacache_threshold=0.1):
        if not ECHOMIMIC_AVAILABLE: return (torch.zeros((1, 64, 64, 3)),)

        pipeline = pipe["pipeline"]
        
        # ... (Previous code)

        # TeaCache Configuration
        if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
            if enable_teacache:
                print(f"[{self.NODE_NAME}] Enabling TeaCache (threshold={teacache_threshold})...")
                # Using 1.3B model string to fetch coefficients
                teacache_coeffs = get_teacache_coefficients("wan2.1-fun-v1.1-1.3b")
                pipeline.transformer.enable_teacache(
                    coefficients=teacache_coeffs,
                    num_steps=steps,
                    rel_l1_thresh=teacache_threshold
                )
            else:
                if hasattr(pipeline.transformer, "teacache") and pipeline.transformer.teacache is not None:
                     print(f"[{self.NODE_NAME}] Disabling TeaCache...")
                     pipeline.transformer.disable_teacache()
        device = pipe["device"]
        dtype = pipe["weight_dtype"]
        wav2vec_processor = pipe["wav2vec_processor"]
        wav2vec_model = pipe["wav2vec_model"].to(device, dtype=dtype)

        # 1. Process Image
        # ComfyUI Image is (B, H, W, C) float32 [0,1]. Take first.
        ref_image_np = (ref_image[0].cpu().numpy() * 255).astype(np.uint8)
        ref_img_pil = Image.fromarray(ref_image_np).convert("RGB")

        # 2. Process Audio
        # ComfyUI Audio is {'waveform': (1, C, N), 'sample_rate': sr}
        audio_waveform = ref_audio['waveform']
        sample_rate = ref_audio['sample_rate']
        
        # Resample to 16000 for Wav2Vec if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio_waveform_16k = resampler(audio_waveform)
        else:
            audio_waveform_16k = audio_waveform

        # Mix down to mono if needed
        if audio_waveform_16k.shape[0] > 1: # (Channels, Time)
            audio_waveform_16k = torch.mean(audio_waveform_16k, dim=0, keepdim=True)
            
        audio_input = audio_waveform_16k.squeeze().cpu().numpy()
        
        # Wav2Vec Extraction
        # Move wav2vec to device for inference then back to CPU
        wav2vec_model.to(device)
        try:
            input_values = wav2vec_processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
            # cast to same dtype as model weights (likely bf16)
            input_values = input_values.to(device=device, dtype=wav2vec_model.dtype)
            with torch.no_grad():
                audio_features = wav2vec_model(input_values).last_hidden_state
            audio_embeds = audio_features # (1, T_audio, D)
        finally:
            wav2vec_model.to("cpu")
            torch.cuda.empty_cache()

        # 3. Setup Video Params
        duration_sec = len(audio_input) / 16000
        video_length = int(duration_sec * fps)
        
        # VAE Compression Ratio alignment (from infer.py)
        # Using 4 by default for Wan usually, or read from config if available.
        # infer.py: vae.config.temporal_compression_ratio
        temporal_compression_ratio = pipeline.vae.config.temporal_compression_ratio
        
        # Adjust video_length for 4x compression alignment
        video_length = (int((video_length - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1 if video_length != 1 else 1)
        
        # 4. Face Mask (IP Mask)
        # Here we need face detection. `get_mask_coord` uses 'src.face_detect' which uses retinaface or similar.
        # We need to save the PIL image temporarily if `get_mask_coord` expects a path, or modify it to accept PIL/cv2.
        # Looking at `src.face_detect`: usually expects path or numpy.
        # Let's save temp for compatibility with existing `get_mask_coord` if it takes path.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            ref_img_pil.save(tmp_img.name)
            tmp_img_path = tmp_img.name
        
        try:
            y1, y2, x1, x2, h_, w_ = get_mask_coord(tmp_img_path)
        except Exception as e:
            print(f"[{self.NODE_NAME}] Face detection failed: {e}. Using full image as mask fallback.")
            w, h = ref_img_pil.size
            y1, y2, x1, x2, h_, w_ = 0, h, 0, w, h, w
        finally:
            if os.path.exists(tmp_img_path): os.remove(tmp_img_path)

        # 5. Latent Prep and Inputs
        sample_height, sample_width = get_sample_size(ref_img_pil, [height, width])
        # Downscale ratio calc
        downratio = math.sqrt(sample_height * sample_width / h_ / w_)
        coords = (
            y1 * downratio // 16, y2 * downratio // 16,
            x1 * downratio // 16, x2 * downratio // 16,
            sample_height // 16, sample_width // 16,
        )
        ip_mask = get_ip_mask(coords).unsqueeze(0)
        ip_mask = torch.cat([ip_mask]*3).to(device=device, dtype=dtype)
        
        # Generator on CPU to avoid device mismatch. 
        # Pipeline._execution_device is likely CPU because Text Encoder is on CPU (optimized).
        # Passing a CUDA generator to a pipeline that thinks it's on CPU causes crash.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        # Chunking Configuration
        partial_video_length = context_length
        overlap_video_length = 8
        
        if partial_video_length <= overlap_video_length:
             # failsafe
             partial_video_length = overlap_video_length + 16
        
        # Memory Debug Helper
        def log_vram(tag):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"[{self.NODE_NAME}] [Memory {tag}] Aloc: {allocated:.2f} GB, Res: {reserved:.2f} GB")
        
        def log_model_devices(pipeline):
            print(f"[{self.NODE_NAME}] --- Model Devices ---")
            for name, module in pipeline.components.items():
                if hasattr(module, 'device'):
                     print(f"  {name}: {module.device}")
                elif hasattr(module, 'execution_device'):
                     print(f"  {name}: {module.execution_device}")
            print("-------------------------")

        # Generate video in chunks
        init_frames = 0
        last_frames = init_frames + partial_video_length
        new_sample = None
        
        # Precompute mix_ratio
        mix_ratio = torch.linspace(0, 1, steps=overlap_video_length).view(1, 1, -1, 1, 1).to(device, dtype=dtype)

        print(f"[{self.NODE_NAME}] Generating {video_length} frames in chunks...")
        
        # Keep track of the current reference image(s) for get_image_to_video_latent3
        current_ref_images = ref_img_pil # Initially a single PIL image

        # Initial GC to clear anything loose
        gc.collect()
        torch.cuda.empty_cache()
        log_vram("Before Loop")
        log_model_devices(pipeline)

        # Progress Bar Setup
        from comfy.utils import ProgressBar
        
        # Calculate total chunks to initialize progress bar correctly
        total_chunks = 0
        temp_init_frames = 0
        while temp_init_frames < video_length:
            chunk_len = partial_video_length
            is_last_chunk = False
            
            if temp_init_frames + chunk_len >= video_length:
                chunk_len = video_length - temp_init_frames
                chunk_len = (int((chunk_len - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1 if chunk_len != 1 else 1)
                is_last_chunk = True
                if chunk_len <= 0: break
            
            total_chunks += 1
            if is_last_chunk:
                break
                
            if temp_init_frames == 0:
                temp_init_frames += chunk_len
            else:
                temp_init_frames += (chunk_len - overlap_video_length)
                
        total_steps = total_chunks * steps
        pbar = ProgressBar(total_steps)
        
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            pbar.update(1)
            return {}

        # Pre-encode text prompts (Fix for speed)
        print(f"[{self.NODE_NAME}] Encoding text prompts...")
        
        # Manual Device Management: Move Text Encoder to GPU for encoding
        if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
             pipeline.text_encoder.to(device)
        
        do_classifier_free_guidance = cfg > 1.0
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
        print(f"[{self.NODE_NAME}] Text prompts encoded.")

        # Aggressive memory cleanup: Move Text Encoder BACK TO CPU
        # This frees VRAM but keeps the model available for next run (unlike deleting it)
        if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
             pipeline.text_encoder.to("cpu")
        
        gc.collect()
        torch.cuda.empty_cache()
        log_vram(f"[{self.NODE_NAME}] After Moving T5 to CPU")

        # Move Generation Models to GPU
        print(f"[{self.NODE_NAME}] Moving models to GPU for generation...")
        if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
            pipeline.transformer.to(device)
            
        if hasattr(pipeline, "vae") and pipeline.vae is not None:
            pipeline.vae.to(device)
            
        # CLIP Image Encoder is moved to GPU only when needed, inside the loop
            
        if hasattr(pipeline, "scheduler") and pipeline.scheduler is not None:
             # Scheduler buffers (sigmas) need to be on device
             # The scheduler doesn't inherit from nn.Module, so .to() might not work or might be custom.
             # Diffusers schedulers generally don't have .to(), but some custom ones might.
             # Our custom FlowDPMSolverMultistepScheduler doesn't inherit from nn.Module but has buffers.
             # Ideally set_timesteps handles this, but let's try to set it explicitly if possible or rely on set_timesteps.
             pass 

        # CRITICAL: Re-run set_timesteps with the correct device to ensure sigmas are on GPU
        # The pipeline.scheduler.set_timesteps might have been called during init on CPU.
        # We need to refresh it on GPU.
        pipeline.scheduler.set_timesteps(steps, device=device)
            
        log_vram(f"[{self.NODE_NAME}] Ready for Generation")

        while init_frames < video_length:
            current_partial_video_length = partial_video_length
            if last_frames >= video_length:
                current_partial_video_length = video_length - init_frames
                current_partial_video_length = (
                    int((current_partial_video_length - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1
                    if current_partial_video_length != 1 else 1
                )
                
                if current_partial_video_length <= 0:
                    break
            
            # Prepare inputs for this chunk
            input_video, input_video_mask, clip_image = get_image_to_video_latent3(
                current_ref_images, 
                None, 
                video_length=current_partial_video_length, 
                sample_size=[sample_height, sample_width]
            )
            
            # Pre-compute CLIP Context (Optimized Device Management)
            clip_context = None
            if hasattr(pipeline, "clip_image_encoder") and pipeline.clip_image_encoder is not None:
                print(f"[{self.NODE_NAME}] Computing CLIP Context (Moving Encoder to GPU)...")
                pipeline.clip_image_encoder.to(device)
                
                # Logic copied from pipeline
                if clip_image is not None:
                     # clip_image from get_image_to_video_latent3 is PIL or Tensor?
                     # get_image_to_video_latent3 returns PIL Image usually for clip_image
                     # pipeline expects PIL Image
                     clip_image_t = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, dtype=dtype)
                     clip_context = pipeline.clip_image_encoder([clip_image_t[:, None, :, :]])
                else:
                     clip_image_t = TF.to_tensor(Image.new("RGB", (512, 512))).sub_(0.5).div_(0.5).to(device, dtype=dtype)
                     clip_context = pipeline.clip_image_encoder([clip_image_t[:, None, :, :]])
                     clip_context = torch.zeros_like(clip_context)

                pipeline.clip_image_encoder.to("cpu")
                torch.cuda.empty_cache()
            
            # Slice audio embeds
            partial_audio_embeds = audio_embeds[:, init_frames * 2 : (init_frames + current_partial_video_length) * 2]

            print(f"[{self.NODE_NAME}] Processing chunk: frames {init_frames} to {init_frames + current_partial_video_length}")
            log_vram(f"Start Chunk {init_frames}")

            with torch.no_grad():
                sample = pipeline(
                    prompt=None, # Changed from prompt
                    num_frames=current_partial_video_length,
                    negative_prompt=None, # Changed from negative_prompt
                    prompt_embeds=prompt_embeds, # Added
                    negative_prompt_embeds=negative_prompt_embeds, # Added
                    audio_embeds=partial_audio_embeds,
                    audio_scale=1.0,
                    ip_mask=ip_mask,
                    use_un_ip_mask=False,
                    height=sample_height,
                    width=sample_width,
                    generator=generator,
                    clip_context=clip_context, # Pass pre-computed context
                    neg_scale=1.5,
                    neg_steps=2,
                    use_dynamic_cfg=True,
                    use_dynamic_acfg=True,
                    guidance_scale=cfg,
                    audio_guidance_scale=audio_cfg,
                    num_inference_steps=steps,
                    video=input_video,
                    mask_video=input_video_mask,
                    clip_image=clip_image,
                    cfg_skip_ratio=0, # default from config
                    shift=5.0, # default from config
                    use_longvideo_cfg=False, # default
                    callback_on_step_end=progress_callback,
                ).videos
            
            # Blending Logic
            if init_frames != 0:
                # Ensure mix_ratio is on the same device as the video tensors
                curr_mix_ratio = mix_ratio.to(new_sample.device)
                new_sample[:, :, -overlap_video_length:] = (
                    new_sample[:, :, -overlap_video_length:] * (1 - curr_mix_ratio) +
                    sample[:, :, :overlap_video_length] * curr_mix_ratio
                )
                new_sample = torch.cat([new_sample, sample[:, :, overlap_video_length:]], dim=2)
            else:
                new_sample = sample
            
            if last_frames >= video_length:
                break
                
            # Update Ref Image for next chunk (from last frames of current sample)
            # app.py: ref_img = [ Image.fromarray(...) ] for i in range(-overlap, 0)
            # But get_image_to_video_latent3 takes ref_img_pil (single).
            # If I pass a list now, will it work? I need to verify utils.py.
            # But assuming I should copy app.py:
            # ref_img updates to this list.
            # Next iteration calls get_image_to_video_latent3 with this list.
            
            # Important: `get_image_to_video_latent3` needs to support list.
            # Assuming it does because app.py does it.
            
            current_ref_images = [
                Image.fromarray(
                    (sample[0, :, i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                ) for i in range(current_partial_video_length - overlap_video_length, current_partial_video_length)
            ]
            
            # Safeguard against infinite loops
            if total_chunks > 50 and init_frames == 0:
                 print(f"[{self.NODE_NAME}] Warning: Loop detected with no progress. Force breaking.")
                 break

            # Clean up
            del input_video, input_video_mask, partial_audio_embeds, sample
            torch.cuda.empty_cache()

            # Update loop counters
            if init_frames == 0:
                init_frames += current_partial_video_length
            else:
                init_frames += (current_partial_video_length - overlap_video_length)
            
            last_frames = init_frames + partial_video_length

        # Final Post-processing
        output = new_sample.permute(0, 2, 3, 4, 1).cpu().float() # (B, F, H, W, C)
        output = output.squeeze(0)

        # Cleanup: Move models back to CPU to free VRAM for other nodes
        print(f"[{self.NODE_NAME}] Cleaning up and moving models back to CPU...")
        if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
            pipeline.transformer.to("cpu")
        if hasattr(pipeline, "vae") and pipeline.vae is not None:
            pipeline.vae.to("cpu")
        if hasattr(pipeline, "clip_image_encoder") and pipeline.clip_image_encoder is not None:
            pipeline.clip_image_encoder.to("cpu")
            
        del new_sample, audio_embeds
        gc.collect()
        torch.cuda.empty_cache()
        log_vram(f"[{self.NODE_NAME}] Finished")
        
        return (output,)

NODE_CLASS_MAPPINGS = {
    "AIIA_EchoMimicLoader": AIIA_EchoMimicLoader,
    "AIIA_EchoMimicSampler": AIIA_EchoMimicSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_EchoMimicLoader": "EchoMimic V3 Loader",
    "AIIA_EchoMimicSampler": "EchoMimic V3 Sampler",
}
