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
        hf_repo_id = "openbmb/VoxCPM-1.5"
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_path, "models", "voxcpm")
        model_path = os.path.join(models_dir, "VoxCPM-1.5")
        
        # Download if not exists
        if not os.path.exists(model_path):
            print(f"[AIIA] Downloading VoxCPM 1.5 to {model_path}...")
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=hf_repo_id, local_dir=model_path, local_dir_use_symlinks=False)
            except ImportError:
                raise ImportError("Please install huggingface_hub to download models automatically.")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

        print(f"[AIIA] Loading VoxCPM from {model_path}...")
        
        # TODO: Actual Model Loading Logic
        # We need to determine if we use 'transformers' or a custom 'voxcpm' package.
        # For now, we'll placeholder the object.
        model = {"path": model_path, "device": device, "dtype": dtype}
        
        return (model,)

class AIIA_VoxCPM_TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxcpm_model": ("VOXCPM_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, world."}),
                "reference_audio": ("AUDIO",), # VoxCPM is Zero-Shot, so this is required for cloning
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/VoxCPM"

    def generate(self, voxcpm_model, text, reference_audio, speed, top_p, temperature):
        # Placeholder generation logic
        print(f"[AIIA] Generating VoxCPM TTS for text: {text[:20]}...")
        
        # Mock output for now
        sr = 44100
        duration_sec = 2
        waveform = torch.zeros((1, int(sr * duration_sec)))
        
        return ({"waveform": waveform, "sample_rate": sr},)

NODE_CLASS_MAPPINGS = {
    "AIIA_VoxCPM_Loader": AIIA_VoxCPM_Loader,
    "AIIA_VoxCPM_TTS": AIIA_VoxCPM_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_VoxCPM_Loader": "🎤 VoxCPM Loader",
    "AIIA_VoxCPM_TTS": "🗣️ VoxCPM 1.5 TTS"
}
