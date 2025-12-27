import torch
import os
import tempfile
import sys
import subprocess
import soundfile as sf
import numpy as np

# Auto-install voicefixer if missing
try:
    from voicefixer import VoiceFixer
except ImportError:
    print("[AIIA] VoiceFixer not found. Installing automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "voicefixer>=0.1.3"])
    from voicefixer import VoiceFixer

import shutil
import folder_paths

def setup_voicefixer_path():
    """
    Migrate VoiceFixer models from ~/.cache to ComfyUI/models/voicefixer
    and set up a symlink so the library finds them.
    """
    try:
        # 1. Target Directory (ComfyUI/models/voicefixer)
        comfy_models_dir = folder_paths.models_dir
        target_dir = os.path.join(comfy_models_dir, "voicefixer")
        
        # 2. Source Cache Directory (~/.cache/voicefixer)
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'voicefixer')
        
        # If cache_dir is already a symlink pointing to target_dir, we are good.
        if os.path.islink(cache_dir):
            if os.readlink(cache_dir) == target_dir:
                return
            else:
                # Wrong symlink? Unlink it.
                os.unlink(cache_dir)

        # Prepare target directory
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            print(f"[AIIA] Created VoiceFixer model directory: {target_dir}")

        # If cache dir exists and is a real folder (contains models downloaded by lib)
        if os.path.exists(cache_dir) and not os.path.islink(cache_dir):
            print(f"[AIIA] Migrating VoiceFixer models from {cache_dir} due to user request...")
            # Move content
            for item in os.listdir(cache_dir):
                s = os.path.join(cache_dir, item)
                d = os.path.join(target_dir, item)
                if os.path.exists(d):
                    # Target already exists, assume it's valid or we overwrite?
                    # Let's keep target if exists, only move if missing.
                    if os.path.isdir(s): shutil.rmtree(s)
                    else: os.remove(s)
                else:
                    shutil.move(s, d)
            # Remove empty cache dir
            try:
                os.rmdir(cache_dir)
            except:
                shutil.rmtree(cache_dir) # Force remove if leftovers

        # Now cache_dir should be gone. Create symlink.
        # Ensure parent .cache exists
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        if not os.path.exists(cache_dir):
            print(f"[AIIA] Creating symlink: {cache_dir} -> {target_dir}")
            try:
                os.symlink(target_dir, cache_dir)
            except OSError:
                print(f"[AIIA] Warning: Failed to create symlink. Models will stay in {target_dir} but VoiceFixer might re-download to cache if not manually pointed.")
                # Fallback: We can't easily fallback without changing lib code.
                # But for now, let's assume symlink works on Mac/Linux.
    except Exception as e:
        print(f"[AIIA] Error setting up model path: {e}")

class AIIA_Audio_Denoise:
    _model_cache = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["Denoise + DeReverb + DeClip (Mode 0)", 
                          "Denoise + DeReverb (Mode 1)", 
                          "Denoise Only (Mode 2)"], {"default": "Denoise Only (Mode 2)"}),
                "use_cuda": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "denoise_audio"
    CATEGORY = "AIIA/Audio"

    def denoise_audio(self, audio, mode, use_cuda):
        # 1. Initialize Model (Singleton)
        cuda_available = torch.cuda.is_available() and use_cuda
        device_str = "cuda" if cuda_available else "cpu"
        
        if AIIA_Audio_Denoise._model_cache is None:
            print(f"[AIIA] Loading VoiceFixer model on {device_str}...")
            setup_voicefixer_path()
            # VoiceFixer(verbose=True, emb_vocoder=True) by default
            # We enforce the device during init if possible, or later during restore
            try:
                AIIA_Audio_Denoise._model_cache = VoiceFixer()
            except RuntimeError as e:
                if "PytorchStreamReader" in str(e) or "zip archive" in str(e):
                    print(f"[AIIA] Error: VoiceFixer model seems corrupted. Purging models to force re-download...")
                    # Purge target dir
                    comfy_models_dir = folder_paths.models_dir
                    target_dir = os.path.join(comfy_models_dir, "voicefixer")
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                        os.makedirs(target_dir, exist_ok=True) # Recreate empty dir
                    
                    raise RuntimeError("[AIIA] VoiceFixer model file was corrupted and has been deleted. Please RESTART the workflow/ComfyUI to re-download it correctly.") from e
                else:
                    raise e

        vf = AIIA_Audio_Denoise._model_cache
        
        # 2. Prepare Input Audio
        waveform = audio["waveform"] # [B, C, T]
        sample_rate = audio["sample_rate"]
        
        # Handle batch (process one by one or error? ComfyUI audio is usually Batch 1)
        if waveform.ndim == 3:
            wav_single = waveform[0] # Take first item in batch [C, T]
        else:
            wav_single = waveform # [C, T] or [T]

        # VoiceFixer expects a file path.
        # It handles resampling internally (output is usually 44100Hz).
        
        # Convert to numpy for saving
        wav_np = wav_single.cpu().numpy()
        if wav_np.ndim == 2:
            wav_np = wav_np.T # [T, C] for soundfile

        result_waveform = None
        result_sr = 44100 # VoiceFixer default output

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            
            in_path = tmp_in.name
            out_path = tmp_out.name
            
            # Save input float32
            sf.write(in_path, wav_np, sample_rate, subtype='FLOAT')
            
            # 3. Parse Mode
            # Mode 0: Original (remove noise, reverb, clipping)
            # Mode 1: Remove noise, reverb
            # Mode 2: Remove noise
            vf_mode = 2
            if "Mode 0" in mode: vf_mode = 0
            elif "Mode 1" in mode: vf_mode = 1
            
            print(f"[AIIA] Running VoiceFixer (Mode {vf_mode}) on {device_str}...")
            
            # 4. Run Inference
            # restore(input, output, cuda, mode)
            # cuda: bool
            vf.restore(input=in_path, output=out_path, cuda=cuda_available, mode=vf_mode)
            
            # 5. Load Result
            # VoiceFixer outputs 44100Hz audio
            out_wav, out_sr = sf.read(out_path, dtype='float32')
            result_sr = out_sr
            
            # Convert back to torch [1, C, T] or [1, 1, T]
            # sf.read returns [T, C] or [T]
            out_tensor = torch.from_numpy(out_wav)
            if out_tensor.ndim == 1:
                out_tensor = out_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, T]
            elif out_tensor.ndim == 2:
                out_tensor = out_tensor.T.unsqueeze(0) # [1, C, T]
            
            result_waveform = out_tensor
            
            # Cleanup
            try:
                os.remove(in_path)
                os.remove(out_path)
            except:
                pass

        return ({"waveform": result_waveform, "sample_rate": result_sr},)

NODE_CLASS_MAPPINGS = {
    "AIIA_Audio_Denoise": AIIA_Audio_Denoise
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Audio_Denoise": "Audio AI Denoise (VoiceFixer)"
}
