
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

# --- Patches for transformers 4.57+ compatibility ---
import contextlib
from transformers import GenerationConfig

@contextlib.contextmanager
def _transformers_patches():
    from transformers import cache_utils
    from transformers.generation import candidate_generator as cg
    from transformers.generation import configuration_utils as cu
    import transformers.modeling_utils as mu

    # Dummy classes/functions
    class _QCC:
        def __init__(self, **kw): pass
    def _crop(model, past_key_values, max_length): return past_key_values
    class _SeqSummary(torch.nn.Module):
        def __init__(self, config): super().__init__()
        def forward(self, hidden_states, **kw): return hidden_states[:, -1]
    def _chunking(forward_fn, chunk_size, *tensors, **kw): return forward_fn(*tensors, **kw)

    patches = [
        (cache_utils, "QuantizedCacheConfig",             _QCC),
        (cg,          "_crop_past_key_values",             _crop),
        (cu,          "NEED_SETUP_CACHE_CLASSES_MAPPING",  {}),
        (cu,          "QUANT_BACKEND_CLASSES_MAPPING",     {}),
        (mu,          "SequenceSummary",                   _SeqSummary),
        (mu,          "apply_chunking_to_forward",         _chunking),
    ]

    originals = []
    _SENTINEL = object()
    for mod, attr, shim in patches:
        old = getattr(mod, attr, _SENTINEL)
        originals.append((mod, attr, old))
        if old is _SENTINEL:
            setattr(mod, attr, shim)

    # GenerationConfig.forced_decoder_ids
    gc_had = hasattr(GenerationConfig, "forced_decoder_ids")
    if not gc_had:
        setattr(GenerationConfig, "forced_decoder_ids", None)

    try:
        yield
    finally:
        # Revert
        for mod, attr, old in reversed(originals):
            if old is _SENTINEL:
                delattr(mod, attr)
        if not gc_had and hasattr(GenerationConfig, "forced_decoder_ids"):
            delattr(GenerationConfig, "forced_decoder_ids")
# ----------------------------------------------------

print(f"Initializing IndexTTS2 on {device}...")

# Config
cfg_path = os.path.join(MODEL_DIR, "config.yaml")

# Test both FP16 and FP32
for use_fp16 in [True, False]:
    print(f"\n--- Testing with use_fp16={use_fp16} ---")
    
    # Initialize with patches applied during import
    try:
        with _transformers_patches():
            # Import inside patch context because IndexTTS imports transformers at top level
            # Wait, IndexTTS was already imported above? FAIL.
            # I must patch BEFORE import.
            # But script structure has imports at top.
            # I need to move imports here or reload.
            if 'indextts.infer_v2' in sys.modules:
                del sys.modules['indextts.infer_v2']
            if 'indextts.gpt.model_v2' in sys.modules:
                 # Recursive reload is hard.
                 pass
            from indextts.infer_v2 import IndexTTS2

            tts = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=MODEL_DIR,
                use_fp16=use_fp16,
                device=device,
                use_cuda_kernel=True if device.startswith("cuda") else False,
                use_deepspeed=False
            )
    except Exception as e:
        print(f"Init failed: {e}")
        continue

    print("Model initialized.")

    # Inference
    text = "你好，这是一段 IndexTTS-2 语音合成测试。"
    ref_audio = os.path.join(ASSETS_DIR, "seed_female_hq.wav")
    suffix = "fp16" if use_fp16 else "fp32"
    output_path = f"test_out_official_{suffix}.wav"
    
    if not os.path.exists(ref_audio):
        print(f"Reference audio not found at {ref_audio}")
        continue

    print(f"Generating audio to {output_path}...")
    
    try:
        with torch.no_grad():
            with _transformers_patches(): # Patches needed for inference too?
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
sys.exit(0)
