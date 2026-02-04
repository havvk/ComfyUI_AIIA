import os
import sys
import torch
import torchaudio
import numpy as np
import folder_paths
import subprocess

# --- Robust Package Installation ---
def _install_qwen_tts_if_needed():
    try:
        from qwen_tts import Qwen3TTSModel
        return
    except ImportError:
        print("[AIIA] qwen-tts missing. Attempting installation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "qwen-tts"])
            print("[AIIA] qwen-tts installed successfully.")
        except Exception as e:
            print(f"[AIIA] Failed to install qwen-tts: {e}")

class AIIA_Qwen_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                ], {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}),
                "device": (["cuda", "cpu", "auto", "mps"], {"default": "auto"}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "local_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load_model"
    CATEGORY = "AIIA/Loaders"

    def load_model(self, model_name, device, dtype, local_path=""):
        _install_qwen_tts_if_needed()
        from qwen_tts import Qwen3TTSModel

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available(): device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"
        
        # Resolve dtype
        torch_dtype = torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "fp16" else torch.float32)
        
        # Resolve path
        path = local_path if local_path and os.path.exists(local_path) else model_name
        
        print(f"[AIIA] Loading Qwen3-TTS: {path} on {device} with {dtype}")
        
        # Flash Attention check
        attn_impl = "flash_attention_2" if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else "sdpa"
        
        model = Qwen3TTSModel.from_pretrained(
            path,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl
        )
        
        model_type = "Base"
        if "CustomVoice" in path: model_type = "CustomVoice"
        elif "VoiceDesign" in path: model_type = "VoiceDesign"
        
        return ({"model": model, "type": model_type, "name": path, "device": device},)

class AIIA_Qwen_TTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，这是 Qwen3-TTS 的测试。"}),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {"default": "Chinese"}),
            },
            "optional": {
                "speaker": ("STRING", {"default": "Vivian"}),
                "instruct": ("STRING", {"multiline": True, "default": ""}),
                "reference_audio": ("AUDIO",),
                "reference_text": ("STRING", {"multiline": True, "default": ""}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/Synthesis"

    def generate(self, qwen_model, text, language, speaker="Vivian", instruct="", reference_audio=None, reference_text="", x_vector_only=False, seed=42, speed=1.0):
        model = qwen_model["model"]
        m_type = qwen_model["type"]
        
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        lang_param = language if language != "Auto" else "Auto"
        
        wavs = None
        sr = 24000 # Default if unknown
        
        try:
            if m_type == "CustomVoice":
                print(f"[AIIA] Qwen3-TTS CustomVoice: {speaker} | Instruct: {instruct}")
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=lang_param,
                    speaker=speaker,
                    instruct=instruct if instruct else None
                )
            elif m_type == "VoiceDesign":
                print(f"[AIIA] Qwen3-TTS VoiceDesign: {instruct}")
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=lang_param,
                    instruct=instruct
                )
            else: # Base / Clone
                if reference_audio is not None:
                    # Convert ComfyUI Audio format to (numpy, sr) tuple
                    device = qwen_model["device"]
                    ref_wav = reference_audio["waveform"]
                    ref_sr = reference_audio["sample_rate"]
                    
                    # Convert to mono if needed
                    if ref_wav.ndim == 3: ref_wav = ref_wav[0]
                    if ref_wav.shape[0] > 1: ref_wav = torch.mean(ref_wav, dim=0, keepdim=True)
                    
                    ref_audio_data = (ref_wav.squeeze().cpu().numpy(), ref_sr)
                    
                    print(f"[AIIA] Qwen3-TTS VoiceClone: Using provided reference.")
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=lang_param,
                        ref_audio=ref_audio_data,
                        ref_text=reference_text if reference_text else None,
                        x_vector_only_mode=x_vector_only
                    )
                else:
                    # Fallback if no reference provided for Base model
                    # Typically Base model MUST have reference.
                    # We might want to provide a default one or error out.
                    raise ValueError("Qwen3-TTS Base model requires 'reference_audio' and 'reference_text' for cloning.")

            # Process output
            if wavs is not None and len(wavs) > 0:
                audio_out = torch.from_numpy(wavs[0]).float()
                if audio_out.ndim == 1: audio_out = audio_out.unsqueeze(0)
                
                # Speed adj (Qwen3-TTS might not have native speed param in generate_* yet, so we use torchaudio if needed)
                if speed != 1.0:
                    # Simple speed change via resampling (pitch change) - matches CosyVoice fallback
                    resampler = torchaudio.transforms.Resample(orig_freq=int(sr*speed), new_freq=sr)
                    audio_out = resampler(audio_out)
                
                return ({"waveform": audio_out.unsqueeze(0), "sample_rate": sr},)
            
        except Exception as e:
            print(f"[AIIA] Qwen3-TTS Generation Error: {e}")
            import traceback
            traceback.print_exc()
            raise e

NODE_CLASS_MAPPINGS = {
    "AIIA_Qwen_Loader": AIIA_Qwen_Loader,
    "AIIA_Qwen_TTS": AIIA_Qwen_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Qwen_Loader": "🤖 Qwen3-TTS Loader",
    "AIIA_Qwen_TTS": "🗣️ Qwen3-TTS Synthesis"
}
