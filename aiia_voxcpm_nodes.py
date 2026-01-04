import os
import sys
import torch
import torchaudio
import numpy as np
import folder_paths
from tqdm import tqdm

class AIIA_VoxCPM_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["VoxCPM-1.5-800M"], {"default": "VoxCPM-1.5-800M"}),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "enable_denoiser": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VOXCPM_MODEL",)
    RETURN_NAMES = ("voxcpm_model",)
    FUNCTION = "load_model"
    CATEGORY = "AIIA/VoxCPM"

    def load_model(self, model_name, precision, enable_denoiser):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if precision == "fp16" else torch.float32
        
        # Define paths
        hf_repo_id = "openbmb/VoxCPM1.5"
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Use ComfyUI standard models directory
        # folder_paths.base_path gives the ComfyUI root
        comfy_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = os.path.join(comfy_root, "models", "voxcpm")
        model_path = os.path.join(models_dir, "VoxCPM1.5")
        
        # Download if not exists
        if not os.path.exists(model_path):
            print(f"[AIIA] Downloading VoxCPM 1.5 to {model_path}...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=hf_repo_id, local_dir=model_path, local_dir_use_symlinks=False)
            except ImportError:
                print("[AIIA] huggingface_hub not found, skipping auto-download. Please ensure model exists.")
            except Exception as e:
                print(f"[AIIA WARNING] Auto-download failed: {e}")

        print(f"[AIIA] Loading VoxCPM from {model_path}...")
        
        print(f"[AIIA] Loading VoxCPM from {model_path}...")
        
        # Add local core directory to sys.path to allow importing 'voxcpm' package from source
        voxcpm_core_path = os.path.join(base_path, "voxcpm_core")
        if voxcpm_core_path not in sys.path:
            sys.path.insert(0, voxcpm_core_path)

        try:
            from voxcpm import VoxCPM
        except ImportError as e:
            # Check for specific missing dependencies of the core package
            missing_package = str(e).split("'")[-2] if "'" in str(e) else str(e)
            
            error_msg = f"Failed to import local VoxCPM core. Missing dependency: {missing_package}.\n"
            error_msg += "Please install lightweight dependencies: `pip install wetext inflect`\n"
            error_msg += f"Detailed Error: {e}"
            raise ImportError(error_msg)

        # ZipEnhancer Path handling
        zip_model_id = "iic/speech_zipenhancer_ans_multiloss_16k_base"
        zip_local_path = os.path.join(models_dir, "speech_zipenhancer_ans_multiloss_16k_base")
        
        if enable_denoiser:
             # Check if we need to download
             if not os.path.exists(zip_local_path):
                 print(f"[AIIA] Downloading ZipEnhancer (Denoiser) to {zip_local_path}...")
                 try:
                     # Try importing modelscope for download
                     from modelscope.hub.snapshot_download import snapshot_download as ms_download
                     ms_download(zip_model_id, local_dir=zip_local_path)
                 except ImportError:
                     print("[AIIA WARNING] `modelscope` package not found. Cannot auto-download to custom path.")
                     print("It will likely download to default ~/.cache if the denoiser runs.")
                     # We fall back to the ID string, relying on internal modelscope logic to find/dl it to cache
                     zip_local_path = zip_model_id
                 except Exception as e:
                     print(f"[AIIA WARNING] ZipEnhancer download failed: {e}")
                     zip_local_path = zip_model_id

        try:
            # Initialize VoxCPM using its native class wrapper
            # The signature is: VoxCPM(voxcpm_model_path=..., enable_denoiser=..., zipenhancer_model_path=...)
            
            # Pass zipenhancer_model_path explicitely to avoid default cache path if possible
            # disable optimization (torch.compile) to avoid cudaMallocAsync errors
            model = VoxCPM(
                voxcpm_model_path=model_path, 
                enable_denoiser=enable_denoiser,
                zipenhancer_model_path=zip_local_path,
                optimize=False
            )
            
            tokenizer = None 
            
            return ({"model": model, "tokenizer": tokenizer, "device": device, "dtype": dtype},)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load VoxCPM model: {e}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load VoxCPM model: {e}")

class AIIA_VoxCPM_TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxcpm_model": ("VOXCPM_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, world."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}), 
                "inference_timesteps": ("INT", {"default": 10, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "prompt_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Transcript of the reference audio (Required if audio provided)"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/VoxCPM"

    def generate(self, voxcpm_model, text, speed, cfg_scale, inference_timesteps, seed, reference_audio=None, prompt_text=None):
        model = voxcpm_model["model"]
        tokenizer = voxcpm_model["tokenizer"]
        device = voxcpm_model["device"]
        
        prompt_wav_path = None
        temp_name = None
        
        # Process Reference Audio if provided
        if reference_audio is not None:
            if not prompt_text or prompt_text.strip() == "":
                 raise ValueError("VoxCPM requires 'prompt_text' (transcript) when 'reference_audio' is provided.")
            
            wav = reference_audio["waveform"]
            ref_sr = reference_audio.get("sample_rate", 24000)
            
            if wav.ndim == 3: wav = wav[0]
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            
            audio_prompt_tensor = wav.to(device)
            if ref_sr != 16000:
                 resampler = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=16000).to(device)
                 audio_prompt_tensor = resampler(audio_prompt_tensor)
                 
            # Normalize
            audio_prompt_tensor = audio_prompt_tensor / (torch.max(torch.abs(audio_prompt_tensor)) + 1e-6)
            
            # Save to temp file
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_name = tf.name
                
            sf.write(temp_name, audio_prompt_tensor.squeeze().cpu().numpy(), 16000)
            prompt_wav_path = temp_name

        print(f"[AIIA] Generating VoxCPM TTS for: '{text[:20]}...' with seed={seed}")
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Ensure strict parity between prompt_wav_path and prompt_text
        # VoxCPM requires both to be None, or both to be valid.
        if prompt_wav_path is None:
            prompt_text = None
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_scale,
                    inference_timesteps=inference_timesteps,
                    # temperature/top_p NOT supported by VoxCPM wrapper
                )
                
                # Determine output format
                out_audio = outputs
                out_sr = 44100
                
                if isinstance(outputs, tuple):
                    out_sr, out_audio = outputs
                elif isinstance(outputs, dict):
                    out_audio = outputs.get("audio", outputs.get("waveform"))
                    out_sr = outputs.get("sample_rate", 44100)
                
                print(f"[AIIA Debug] Raw Output Type: {type(out_audio)}")
                if hasattr(out_audio, "shape"):
                    print(f"[AIIA Debug] Raw Output Shape: {out_audio.shape}")

                # Clean up
                if temp_name and os.path.exists(temp_name):
                    os.remove(temp_name)
                    
                # Convert to tensor
                if not isinstance(out_audio, torch.Tensor):
                    out_audio = torch.from_numpy(out_audio)
                
                print(f"[AIIA Debug] Tensor Shape Before Unsqueeze: {out_audio.shape}")

                if out_audio.ndim == 1: 
                    out_audio = out_audio.unsqueeze(0) # [C, T]
                
                # Ensure it is 3D [Batch, Channels, Time] for ComfyUI
                if out_audio.ndim == 2:
                    out_audio = out_audio.unsqueeze(0) # [B, C, T]
                
                print(f"[AIIA Debug] Final Tensor Shape: {out_audio.shape}")

                # Speed adj post-processing
                if speed != 1.0:
                    try:
                        # Note: naive resampling changes duration AND pitch.
                        # Ideally allow disabling this if unwanted. 
                        # Resample expects [..., time]
                        resampler_speed = torchaudio.transforms.Resample(orig_freq=out_sr, new_freq=int(out_sr*speed))
                        # Move to same device
                        resampler_speed = resampler_speed.to(out_audio.device)
                        out_audio = resampler_speed(out_audio)
                    except Exception as e:
                        print(f"[AIIA Warning] Speed adjustment failed: {e}")
                
                return ({"waveform": out_audio.cpu(), "sample_rate": out_sr},)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"VoxCPM Generation Error: {e}")

NODE_CLASS_MAPPINGS = {
    "AIIA_VoxCPM_Loader": AIIA_VoxCPM_Loader,
    "AIIA_VoxCPM_TTS": AIIA_VoxCPM_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_VoxCPM_Loader": "🎤 VoxCPM Loader",
    "AIIA_VoxCPM_TTS": "🗣️ VoxCPM 1.5 TTS"
}
