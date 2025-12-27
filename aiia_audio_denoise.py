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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "voicefixer==0.1.2"])
    from voicefixer import VoiceFixer

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
            # VoiceFixer(verbose=True, emb_vocoder=True) by default
            # We enforce the device during init if possible, or later during restore
            AIIA_Audio_Denoise._model_cache = VoiceFixer()

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
