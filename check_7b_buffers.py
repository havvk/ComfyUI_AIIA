import torch
import os
from safetensors.torch import load_file

def check_model_buffers(model_path):
    print(f"Checking model at: {model_path}")
    
    # Check for index or tensors
    tensors = []
    if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
        import json
        with open(os.path.join(model_path, "model.safetensors.index.json"), "r") as f:
            index = json.load(f)
            tensors = list(index["weight_map"].values())
            # Deduplicate
            tensors = list(set(tensors))
    else:
        # Try finding all safetensors
        tensors = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    
    found_scaling = False
    found_bias = False
    
    for tensor_file in tensors:
        file_path = os.path.join(model_path, tensor_file)
        print(f"Reading {tensor_file}...")
        try:
            weights = load_file(file_path, device="cpu")
            if "model.speech_scaling_factor" in weights:
                val = weights["model.speech_scaling_factor"]
                print(f"✅ model.speech_scaling_factor: {val}")
                found_scaling = True
            if "model.speech_bias_factor" in weights:
                val = weights["model.speech_bias_factor"]
                print(f"✅ model.speech_bias_factor: {val}")
                found_bias = True
            
            if found_scaling and found_bias:
                break
        except Exception as e:
            print(f"Error reading {tensor_file}: {e}")
            
    if not found_scaling:
        print("❌ model.speech_scaling_factor NOT found!")
    if not found_bias:
        print("❌ model.speech_bias_factor NOT found!")

# We assume models are in models/vibevoice/vibevoice/VibeVoice-7B/
base_path = "/Users/l.ylive.cn/Library/CloudStorage/OneDrive-个人/ComfyUI节点/ComfyUI_AIIA/../../models/vibevoice/vibevoice/VibeVoice-7B"
if os.path.exists(base_path):
    check_model_buffers(base_path)
else:
    print(f"Path not found: {base_path}")
    # Try microsoft path
    alt_base_path = "/Users/l.ylive.cn/Library/CloudStorage/OneDrive-个人/ComfyUI节点/ComfyUI_AIIA/../../models/vibevoice/microsoft/VibeVoice-7B"
    if os.path.exists(alt_base_path):
        check_model_buffers(alt_base_path)
    else:
        print("Model paths not found.")
