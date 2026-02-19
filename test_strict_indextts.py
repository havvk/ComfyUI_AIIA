
import sys
import os
import torch

# Base paths
COMFY_ROOT = "/app/ComfyUI"
AIIA_ROOT = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(AIIA_ROOT, "libs", "index-tts")
MODEL_DIR = os.path.join(COMFY_ROOT, "models", "indextts2")
ASSETS_DIR = os.path.join(AIIA_ROOT, "assets")

# Add libs to sys.path
sys.path.insert(0, LIB_PATH)

print(f"--- IndexTTS Strict Test ---")
print(f"Lib path: {LIB_PATH}")
print(f"Model dir: {MODEL_DIR}")

# Import
try:
    from indextts.infer_v2 import IndexTTS2
    print("Successfully imported IndexTTS2")
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)

# Config
cfg_path = os.path.join(MODEL_DIR, "config.yaml")
use_fp16 = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Initializing IndexTTS2 on {device}...")

# Initialize without any patches
tts = IndexTTS2(
    cfg_path=cfg_path,
    model_dir=MODEL_DIR,
    use_fp16=use_fp16,
    device=device,
    use_cuda_kernel=True if device.startswith("cuda") else False,
    use_deepspeed=False
)

print("Model initialized.")

# Inference
text = "你好，这是一段 IndexTTS-2 语音合成测试。如果声音失真，说明模型或环境有问题。"
ref_audio = os.path.join(ASSETS_DIR, "seed_female_hq.wav")
output_path = "test_out_official.wav"

if not os.path.exists(ref_audio):
    print(f"Reference audio not found at {ref_audio}")
    sys.exit(1)

print(f"Generating audio...")
print(f"Text: {text}")
print(f"Ref: {ref_audio}")

try:
    with torch.no_grad():
        tts.infer(
            text=text,
            spk_audio_prompt=ref_audio,
            output_path=output_path,
            verbose=True
        )
    print(f"Successfully generated: {output_path}")
except Exception as e:
    print(f"Inference failed: {e}")
    import traceback
    traceback.print_exc()

print("--- End Test ---")
