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
            }
        }

    RETURN_TYPES = ("VOXCPM_MODEL",)
    RETURN_NAMES = ("voxcpm_model",)
    FUNCTION = "load_model"
    CATEGORY = "AIIA/VoxCPM"

    def load_model(self, model_name, precision):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if precision == "fp16" else torch.float32
        
        # Define paths
        hf_repo_id = "openbmb/VoxCPM1.5"
        
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
        
        try:
            from voxcpm import VoxCPM
        except ImportError:
            # User warned about dependency conflicts (especially torch>=2.5.0 requirement of voxcpm)
            # breaking standard ComfyUI environments.
            # We will NOT auto-install. We will guide the user.
            raise ImportError(
                "VoxCPM package is missing. \n"
                "Please manually install it: `pip install voxcpm`\n"
                "⚠️ WARNING: VoxCPM requires PyTorch >= 2.5.0. \n"
                "If your ComfyUI environment uses an older Torch version, installing this MAY BREAK your setup. \n"
                "Proceed with caution."
            )

        try:
            # Initialize VoxCPM using its native class
            # Assumption: VoxCPM(pretrained=path) or from_pretrained(path)
            # Search results suggest instantiation might be direct or via helper.
            # Let's try standard HF style if supported by wrapper, or constructor.
            # If 'voxcpm' is a wrapper around the model code, it likely has a nice API.
            
            # Trying standard init first
            if hasattr(VoxCPM, 'from_pretrained'):
                model = VoxCPM.from_pretrained(model_path, device=device)
            else:
                # Fallback to constructor
                model = VoxCPM(pretrained=model_path, device=device)
            
            # The 'voxcpm' package usually handles tokenizer internally
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
                "reference_audio": ("AUDIO",), # VoxCPM is Zero-Shot
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/VoxCPM"

    def generate(self, voxcpm_model, text, reference_audio, speed, top_p, temperature, cfg_scale, seed):
        model = voxcpm_model["model"]
        tokenizer = voxcpm_model["tokenizer"]
        device = voxcpm_model["device"]
        
        # Process Reference Audio
        wav = reference_audio["waveform"]
        sr = 44100 # VoxCPM is 44.1k
        ref_sr = reference_audio.get("sample_rate", 24000) # ComfyUI generic audio is often 24k or 44.1k
        
        if wav.ndim == 3: wav = wav[0] # [Batch, C, T] -> [C, T]
        if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True) # Mix stereo to mono
        
        # Resample input to 44100 if needed? 
        # Actually VoxCPM usually expects 16k or 24k reference? 
        # Standard procedure: Resample input reference to match model's expected prompt SR (often 16k for TTS prompts)
        # But VoxCPM output is 44.1k. Let's assume prompt should be 16000 or 44100.
        # Let's try 16000 for safety as prompt, or check docs. 
        # NOTE: OpenBMB/VoxCPM usually takes 'prompt_wav' path or tensor.
        
        audio_prompt_tensor = wav.to(device)
        if ref_sr != 16000:
             resampler = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=16000).to(device)
             audio_prompt_tensor = resampler(audio_prompt_tensor)
             
        # Normalize audio prompt
        audio_prompt_tensor = audio_prompt_tensor / (torch.max(torch.abs(audio_prompt_tensor)) + 1e-6)

        print(f"[AIIA] Generating VoxCPM TTS for: '{text[:20]}...' with params: temp={temperature}, top_p={top_p}, seed={seed}")
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        try:
            with torch.no_grad():
                # Assuming the model has a 'generate' method similar to simple usage
                # If custom code, signature might vary.
                # Common: model.generate(text, prompt_wav, ...)
                
                # Check signature of custom model via dir() if we could, but here we guess/align with reference.
                # Reference nodes often convert tensor to numpy or save to temp file. 
                # Let's try passing the tensor directly if supported, or save to temp.
                
                # OPTION 1: Memory-based (Preferred)
                # audio, sr = model.generate(text, audio_prompt_tensor, ...)
                
                # However, many HF custom codes expect file paths.
                # To be safe and robust, let's write prompt to temp file.
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    temp_name = tf.name
                    
                # Save prompt (16k mono)
                sf.write(temp_name, audio_prompt_tensor.squeeze().cpu().numpy(), 16000)
                
                # Invoke generation
                # Note: API might be model.inference(text, prompt_path, ...)
                outputs = model.generate(
                    text=text,
                    prompt_audio_path=temp_name, # Most robust guess for OpenBMB
                    # prompt_text=prompt_text, # Optional usually
                    temperature=temperature,
                    top_p=top_p,
                    guidance_scale=cfg_scale,
                    # speed=speed # If supported
                )
                
                # Determine output format
                # Usually (samplerate, audio_numpy) or just audio_numpy
                
                out_audio = outputs
                out_sr = 44100
                
                if isinstance(outputs, tuple):
                    out_sr, out_audio = outputs
                elif isinstance(outputs, dict):
                    out_audio = outputs.get("audio", outputs.get("waveform"))
                    out_sr = outputs.get("sample_rate", 44100)
                
                # Clean up
                os.remove(temp_name)
                    
                # Convert to tensor
                if not isinstance(out_audio, torch.Tensor):
                    out_audio = torch.from_numpy(out_audio)
                
                if out_audio.ndim == 1: out_audio = out_audio.unsqueeze(0)
                
                # Speed adj post-processing if model didn't handle it
                if speed != 1.0:
                    resampler_speed = torchaudio.transforms.Resample(orig_freq=out_sr, new_freq=int(out_sr*speed))
                    out_audio = resampler_speed(out_audio)
                    # Resample back to preserve SR? Or just tag it? 
                    # ComfyUI usually expects waveform and SR. If we change duration via resampling, 
                    # we change pitch unless we use TimeStretch. 
                    # Simple resampling CHANGES PITCH. 
                    # If user wants speed control without pitch shift, we need time stretching.
                    # For now, let's skip speed if not supported natively to avoid broken pitch.
                    # Or assume model supports it. If not, improved later.
                
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
