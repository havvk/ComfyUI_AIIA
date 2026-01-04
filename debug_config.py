
import sys
import os
import json
import torch

# Setup paths to mimic the node environment
sys.path.append("/app/ComfyUI/custom_nodes/ComfyUI_AIIA")
sys.path.append("/app/ComfyUI/custom_nodes/ComfyUI_AIIA/voxcpm_core")

from voxcpm.model.voxcpm import VoxCPMConfig, VoxCPMModel
from voxcpm.modules.audiovae import AudioVAE, AudioVAEConfig

path = "/app/ComfyUI/models/voxcpm/VoxCPM1.5"
config_path = os.path.join(path, "config.json")

print(f"--- DEBUGGING CONFIG LOADING ---")
try:
    with open(config_path, 'r') as f:
        data = json.load(f)
    print(f"JSON 'audio_vae_config' sample_rate: {data.get('audio_vae_config', {}).get('sample_rate')}")
    
    # Try loading Config Object
    config = VoxCPMConfig.model_validate(data)
    print(f"Loaded VoxCPMConfig.device: {config.device}")
    
    if config.audio_vae_config:
        print(f"Loaded VoxCPMConfig.audio_vae_config.sample_rate: {config.audio_vae_config.sample_rate}")
    else:
        print(f"Loaded VoxCPMConfig.audio_vae_config is NONE (Used Default?)")

    # Try recreating the Model Load logic manually
    audio_vae = AudioVAE(config.audio_vae_config)
    print(f"Initialized AudioVAE.sample_rate: {audio_vae.sample_rate}")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
