
import sys
import os
import torch
# Add ComfyUI_AIIA to path to import voxcpm
sys.path.append("/app/ComfyUI/custom_nodes/ComfyUI_AIIA")
sys.path.append("/app/ComfyUI/custom_nodes/ComfyUI_AIIA/voxcpm_core")

from voxcpm.core import VoxCPM

try:
    path = "/app/ComfyUI/models/voxcpm/VoxCPM1.5"
    print(f"Loading VoxCPM from {path}")
    model = VoxCPM(path, optimize=False, enable_denoiser=False)
    
    print(f"--- DIAGNOSTICS ---")
    print(f"Model VAE Sample Rate: {model.tts_model.sample_rate}")
    print(f"Model VAE Config Object: {model.tts_model.audio_vae.config}")
    print(f"-------------------")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
