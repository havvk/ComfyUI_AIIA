
import sys
import os
import torch
import contextlib

# Base paths
COMFY_ROOT = "/app/ComfyUI"
AIIA_ROOT = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(AIIA_ROOT, "libs", "index-tts")
MODEL_DIR = os.path.join(COMFY_ROOT, "models", "indextts2")
ASSETS_DIR = os.path.join(AIIA_ROOT, "assets")

# Add libs to sys.path
sys.path.insert(0, LIB_PATH)

print(f"--- IndexTTS Strict Test (Patched) ---")
print(f"Lib path: {LIB_PATH}")
print(f"Model dir: {MODEL_DIR}")

# --- Patches for transformers 4.57+ compatibility ---
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

# Config
cfg_path = os.path.join(MODEL_DIR, "config.yaml")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

# Test both FP16 and FP32
for use_fp16 in [True, False]:
    print(f"\n--- Testing with use_fp16={use_fp16} ---")
    
    # Initialize with patches applied during import
    try:
        with _transformers_patches():
            # Clear modules to ensure patches apply if imported before (unlikely here but safe)
            if 'indextts.infer_v2' in sys.modules:
                del sys.modules['indextts.infer_v2']
            
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
        import traceback
        traceback.print_exc()
        continue

    print("Model initialized.")

    # Inference
    text = f"你好，这是一段 IndexTTS-2 语音合成测试。FP16={use_fp16}"
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
                # We need to capture the raw wav before it's saved/converted ideally.
                # But tts.infer() saves it.
                # However, tts.infer() returns the path.
                # Let's try to monkeypatch torch.clamp or torchaudio.save inside the context?
                # Or just use the model components directly? No, too complex.
                
                # We can use tts.infer(..., stream_return=True) to get the generator?
                # tts.infer signature: if stream_return is True, it yields output.
                
                # Let's try stream_return=True to get raw audio data.
                # output = next(tts.infer(..., stream_return=True))
                # infer() yields partial wavs? No, logic says:
                # if stream_return: yield wav.cpu() (per segment)
                # then yields silence.
                # then yields output_path (if output_path set)?
                # Wait, code says:
                # if output_path: ... save ... if stream_return: return None ... yield output_path
                # So if output_path is set AND stream_return is True, it might not yield wavs?
                
                # Let's look at lines 669:
                # if stream_return: yield wav.cpu()
                
                # So we can collect segments.
                generator = tts.infer(
                    text=text,
                    spk_audio_prompt=ref_audio,
                    output_path=None, # Don't let it save
                    stream_return=True,
                    verbose=True
                )
                
                wavs = []
                for chunk in generator:
                    if isinstance(chunk, tuple):
                        # (sampling_rate, wav_data_numpy) - final yield if output_path=None
                        pass
                    elif isinstance(chunk, torch.Tensor):
                        wavs.append(chunk)

                # Combine
                if not wavs:
                    print("No wavs generated via stream!")
                    continue
                    
                full_wav = torch.cat(wavs, dim=1) if len(wavs) > 1 else wavs[0]
                
                # Save as float (normalized)
                # The model output seems to be already scaled to 32767 in `infer_v2.py` line 664?
                # wav = torch.clamp(32767 * wav, ...)
                # Wait, `infer_v2.py` does that inside the loop.
                # So `wavs` collected here are ALREADY multiplied by 32767.
                
                # So we should divide by 32768.0 to get back to float [-1, 1]
                full_wav_float = full_wav.float() / 32768.0
                
                import torchaudio
                torchaudio.save(output_path.replace(".wav", "_norm.wav"), full_wav_float, 24000)
                print(f"Saved normalized float wav: {output_path.replace('.wav', '_norm.wav')}")
                
                # Also save as int16 as original
                torchaudio.save(output_path, full_wav.to(torch.int16), 24000)
                print(f"Saved int16 wav: {output_path}")

    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

print("--- End Test ---")

