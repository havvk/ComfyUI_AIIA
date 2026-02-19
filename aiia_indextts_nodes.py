"""
AIIA IndexTTS-2 ComfyUI Nodes
Loader + TTS nodes for Bilibili's IndexTTS-2 zero-shot voice cloning model.
"""

import os
import sys
import torch
import torchaudio
import tempfile
import numpy as np
import contextlib
import shutil

# ---------------------------------------------------------------------------
#  Lazy setup: add libs/index-tts to sys.path so `from indextts...` works
# ---------------------------------------------------------------------------
_INDEXTTS_READY = False
_INDEXTTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs", "index-tts")


# Global cache to prevent reloading the model on every execution
_INDEXTTS_MODEL_CACHE = {}


def _ensure_indextts():
    """Clone repo if needed and add to sys.path (idempotent)."""
    global _INDEXTTS_READY
    if _INDEXTTS_READY:
        return

    if not os.path.exists(_INDEXTTS_DIR):
        print("[AIIA] IndexTTS-2 not found in libs/. Please clone it:")
        print(f"  cd {os.path.dirname(_INDEXTTS_DIR)}")
        print("  git clone https://github.com/index-tts/index-tts.git")
        raise ImportError(
            "IndexTTS-2 not found. Clone https://github.com/index-tts/index-tts.git "
            f"into {_INDEXTTS_DIR}"
        )

    # Add to sys.path with priority
    if _INDEXTTS_DIR not in sys.path:
        sys.path.insert(0, _INDEXTTS_DIR)
    else:
        sys.path.remove(_INDEXTTS_DIR)
        sys.path.insert(0, _INDEXTTS_DIR)

    # Install missing pip dependencies that ComfyUI env may lack
    _install_missing_deps()


    # Patch transformers compatibility before any indextts import
    # _patch_transformers_compat() -> Now using context manager in load/infer


    _INDEXTTS_READY = True
    print(f"[AIIA] IndexTTS-2 loaded from: {_INDEXTTS_DIR}")





@contextlib.contextmanager
def _active_transformers_patches():
    """
    Context manager that temporarily applies transformers >= 4.57 compatibility patches
    and reverts them on exit. This prevents breaking other nodes like NeMo that might
    rely on the original transformers behavior or their own version checks.
    """
    _patches_applied = []
    
    try:
        # 1. QuantizedCacheConfig (removed from cache_utils)
        from transformers import cache_utils
        if not hasattr(cache_utils, "QuantizedCacheConfig"):
            class QuantizedCacheConfig:
                def __init__(self, **kwargs): pass
            cache_utils.QuantizedCacheConfig = QuantizedCacheConfig
            _patches_applied.append((cache_utils, "QuantizedCacheConfig"))
            # print("[AIIA] Applied patch: transformers.cache_utils.QuantizedCacheConfig")

        # 2. _crop_past_key_values (removed from candidate_generator)
        from transformers.generation import candidate_generator as cg
        if not hasattr(cg, "_crop_past_key_values"):
            def _crop_past_key_values(model, past_key_values, max_length):
                return past_key_values
            cg._crop_past_key_values = _crop_past_key_values
            _patches_applied.append((cg, "_crop_past_key_values"))
            # print("[AIIA] Applied patch: transformers.generation.candidate_generator._crop_past_key_values")

        # 3. NEED_SETUP_CACHE_CLASSES_MAPPING & QUANT_BACKEND_CLASSES_MAPPING
        from transformers.generation import configuration_utils as cu
        if not hasattr(cu, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
            cu.NEED_SETUP_CACHE_CLASSES_MAPPING = {}
            _patches_applied.append((cu, "NEED_SETUP_CACHE_CLASSES_MAPPING"))
        if not hasattr(cu, "QUANT_BACKEND_CLASSES_MAPPING"):
            cu.QUANT_BACKEND_CLASSES_MAPPING = {}
            _patches_applied.append((cu, "QUANT_BACKEND_CLASSES_MAPPING"))

        # 4. SequenceSummary (removed from modeling_utils)
        import transformers.modeling_utils as mu
        if not hasattr(mu, "SequenceSummary"):
            class SequenceSummary(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                def forward(self, hidden_states, **kwargs):
                    return hidden_states[:, -1]
            mu.SequenceSummary = SequenceSummary
            _patches_applied.append((mu, "SequenceSummary"))
            # print("[AIIA] Applied patch: transformers.modeling_utils.SequenceSummary")

        # 5. GenerationConfig.forced_decoder_ids (removed in 4.39)
        from transformers import GenerationConfig
        if not hasattr(GenerationConfig, "forced_decoder_ids"):
            setattr(GenerationConfig, "forced_decoder_ids", None)
            _patches_applied.append((GenerationConfig, "forced_decoder_ids"))
            # print("[AIIA] Applied patch: transformers.GenerationConfig.forced_decoder_ids")

        # 6. apply_chunking_to_forward (removed in 4.37)
        if not hasattr(mu, "apply_chunking_to_forward"):
            def apply_chunking_to_forward(forward_chunk_fn, chunk_size, *input_tensors, **kwargs):
                # Basic pass-through implementation
                return forward_chunk_fn(*input_tensors, **kwargs)
            
            mu.apply_chunking_to_forward = apply_chunking_to_forward
            _patches_applied.append((mu, "apply_chunking_to_forward"))
            # print("[AIIA] Applied patch: transformers.modeling_utils.apply_chunking_to_forward")
        
        yield

    except Exception as e:
        print(f"[AIIA] Warning: transformers compat patch failed: {e}")
        yield

    finally:
        # Revert patches
        for target, attr_name in reversed(_patches_applied):
            if hasattr(target, attr_name):
                delattr(target, attr_name)
        # print(f"[AIIA] Reverted {len(_patches_applied)} transformers patches.")



def _install_missing_deps():
    """Install lightweight pip packages that IndexTTS-2 needs but ComfyUI might lack."""
    import subprocess
    required = [
        "omegaconf",
        "cn2an",
        "g2p_en",
        "jieba",
        "munch",
        "librosa",
        "json5",
    ]
    # Check which ones are missing
    missing = []
    for pkg in required:
        # Skip core packages that should verify by ComfyUI
        if pkg in ["torch", "torchaudio", "transformers"]:
            continue
            
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # wetext / WeTextProcessing
    try:
        __import__("tn")  # wetext exposes 'tn' module
    except ImportError:
        try:
            __import__("wetext")
        except ImportError:
            if sys.platform == "linux":
                missing.append("WeTextProcessing")
            else:
                missing.append("wetext")

    if missing:
        print(f"[AIIA] Installing missing IndexTTS-2 dependencies: {missing}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )


# ============================================================================
#  LOADER NODE
# ============================================================================

@contextlib.contextmanager
def _patch_indextts_loading(model_dir):
    """
    Monkeypatch IndexTTS-2's internal loading calls to support local sub-models.
    Redirects hf_hub_download/from_pretrained to local folders if they exist in model_dir.
    """
    from indextts import infer_v2
    from transformers import SeamlessM4TFeatureExtractor
    from indextts.s2mel.modules.bigvgan import bigvgan

    # Save originals
    orig_hf_download = infer_v2.hf_hub_download
    orig_bigvgan_from_pretrained = bigvgan.BigVGAN.from_pretrained
    orig_seamless_from_pretrained = SeamlessM4TFeatureExtractor.from_pretrained

    # --- 1. Patch hf_hub_download (for MaskGCT & Campplus) ---
    def patched_hf_download(repo_id, filename=None, **kwargs):
        # amphion/MaskGCT -> model_dir/semantic_codec/model.safetensors
        if repo_id == "amphion/MaskGCT" and filename == "semantic_codec/model.safetensors":
            local_path = os.path.join(model_dir, "semantic_codec", "model.safetensors")
            if os.path.exists(local_path):
                print(f"[AIIA] Found local MaskGCT: {local_path}")
                return local_path
        
        # funasr/campplus -> model_dir/campplus_cn_common.bin
        if repo_id == "funasr/campplus" and filename == "campplus_cn_common.bin":
            local_path = os.path.join(model_dir, "campplus_cn_common.bin")
            if os.path.exists(local_path):
                print(f"[AIIA] Found local Campplus: {local_path}")
                return local_path
        
        return orig_hf_download(repo_id, filename=filename, **kwargs)

    # --- 2. Patch BigVGAN.from_pretrained ---
    @classmethod
    def patched_bigvgan(cls, pretrained_model_name_or_path, **kwargs):
        # Check model_dir/bigvgan_v2...
        basename = os.path.basename(pretrained_model_name_or_path)
        local_path = os.path.join(model_dir, basename)
        if os.path.exists(local_path):
            print(f"[AIIA] Found local BigVGAN: {local_path}")
            return orig_bigvgan_from_pretrained(local_path, **kwargs)
        return orig_bigvgan_from_pretrained(pretrained_model_name_or_path, **kwargs)

    # --- 3. Patch SeamlessM4TFeatureExtractor.from_pretrained (W2V-BERT) ---
    @classmethod
    def patched_seamless(cls, pretrained_model_name_or_path, **kwargs):
        if pretrained_model_name_or_path == "facebook/w2v-bert-2.0":
            local_path = os.path.join(model_dir, "w2v-bert-2.0")
            if os.path.exists(local_path):
                print(f"[AIIA] Found local W2V-BERT: {local_path}")
                return orig_seamless_from_pretrained(local_path, **kwargs)
        return orig_seamless_from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Apply patches
    infer_v2.hf_hub_download = patched_hf_download
    bigvgan.BigVGAN.from_pretrained = patched_bigvgan
    # infer_v2 imports this class, so patching the class globally works for infer_v2 usage
    SeamlessM4TFeatureExtractor.from_pretrained = patched_seamless

    try:
        yield
    finally:
        # Restore originals
        infer_v2.hf_hub_download = orig_hf_download
        bigvgan.BigVGAN.from_pretrained = orig_bigvgan_from_pretrained
        SeamlessM4TFeatureExtractor.from_pretrained = orig_seamless_from_pretrained

class AIIA_IndexTTS2_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_fp16": ("BOOLEAN", {"default": True, "tooltip": "Use half-precision for lower VRAM and faster inference."}),
                "use_cuda_kernel": ("BOOLEAN", {"default": True, "tooltip": "Use BigVGAN custom CUDA kernel (faster, CUDA only)."}),
            },
            "optional": {
                "model_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Override model directory. Leave empty to use ComfyUI/models/indextts2/."
                }),
            }
        }

    RETURN_TYPES = ("INDEXTTS_MODEL",)
    RETURN_NAMES = ("indextts_model",)
    FUNCTION = "load_model"
    CATEGORY = "AIIA/Loaders"

    def load_model(self, use_fp16=True, use_cuda_kernel=True, model_dir=""):
        _ensure_indextts()

        import folder_paths

        # Determine model dir: user override → ComfyUI/models/indextts2/
        if not model_dir or not model_dir.strip():
            model_dir = os.path.join(folder_paths.models_dir, "indextts2")

        os.makedirs(model_dir, exist_ok=True)

        # config.yaml must be in model_dir
        cfg_path = os.path.join(model_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            # Try to copy from libs/index-tts/checkpoints/ if available there
            lib_cfg = os.path.join(_INDEXTTS_DIR, "checkpoints", "config.yaml")
            if os.path.exists(lib_cfg):
                import shutil
                shutil.copy2(lib_cfg, cfg_path)
                print(f"[AIIA] Copied config.yaml from {lib_cfg} → {cfg_path}")

        # Check for essential model files
        essential_files = ["config.yaml", "gpt.pth", "s2mel.pth", "bpe.model"]
        missing = [f for f in essential_files if not os.path.exists(os.path.join(model_dir, f))]

        if missing:
            raise FileNotFoundError(
                f"IndexTTS-2 model files missing in {model_dir}: {missing}\n\n"
                f"Please download the model:\n"
                f"  # Using HF Mirror (recommended for China):\n"
                f"  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download IndexTeam/IndexTTS-2 --local-dir {model_dir}\n\n"
                f"  # Or using ModelScope:\n"
                f"  modelscope download --model IndexTeam/IndexTTS-2 --local_dir {model_dir}\n"
            )

        # Check cache
        cache_key = (model_dir, use_fp16, use_cuda_kernel)
        if cache_key in _INDEXTTS_MODEL_CACHE:
            print(f"[AIIA] IndexTTS-2 found in cache for {model_dir}")
            return (_INDEXTTS_MODEL_CACHE[cache_key],)

        print(f"[AIIA] Loading IndexTTS-2 from {model_dir} (fp16={use_fp16}, cuda_kernel={use_cuda_kernel})")
        
        # Apply transformers patches BEFORE importing indextts, because the import chain
        # (infer_v2 → model_v2 → transformers_gpt2 → transformers_generation_utils) does
        # top-level `from transformers.cache_utils import QuantizedCacheConfig` which needs
        # the patch to exist first.
        #
        # IMPORTANT: indextts's infer_v2.py does `from modelscope import AutoModelForCausalLM`
        # at module level, which globally monkey-patches transformers.PreTrainedModel.from_pretrained.
        # This breaks NeMo's SortformerEncLabelModel.restore_from() (hangs indefinitely).
        # We save and restore the original method to neutralize modelscope's side-effect.
        import transformers
        _orig_from_pretrained = transformers.PreTrainedModel.from_pretrained

        with _active_transformers_patches():
            from indextts.infer_v2 import IndexTTS2
            with _patch_indextts_loading(model_dir):
                tts = IndexTTS2(
                    cfg_path=cfg_path,
                    model_dir=model_dir,
                    use_fp16=use_fp16,
                    use_cuda_kernel=use_cuda_kernel,
                    use_deepspeed=False,
                )

        # Restore original from_pretrained after modelscope monkey-patched it
        transformers.PreTrainedModel.from_pretrained = _orig_from_pretrained
        
        _INDEXTTS_MODEL_CACHE[cache_key] = tts
        print("[AIIA] IndexTTS-2 loaded successfully.")

        return (tts,)


# ============================================================================
#  TTS NODE
# ============================================================================

class AIIA_IndexTTS2_TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "indextts_model": ("INDEXTTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，这是一段 IndexTTS-2 语音合成测试。"}),
                "voice_preset": (["Female_HQ", "Male_HQ", "Female", "Male"], {"default": "Female_HQ"}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Speaker voice reference. Leave empty to use voice_preset."}),
                "emotion_audio": ("AUDIO", {"tooltip": "Optional emotion reference audio (separate from speaker voice)."}),
                "emo_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Emotion blending strength (0=no emotion, 1=full emotion)."
                }),
                "happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sad":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "afraid": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "disgusted": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "melancholic": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "surprised": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "calm":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "use_random": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable random sampling (reduces voice cloning fidelity)."
                }),
                "seed": ("INT", {
                    "default": 0, "min": -1, "max": 2147483647,
                    "tooltip": "Random seed. -1 = random."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/Synthesis"

    def _load_fallback_audio(self, target_name="Female_HQ"):
        """Load a built-in seed audio as fallback when no reference is provided."""
        nodes_path = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(nodes_path, "assets")

        filename_map = {
            "Female_HQ": "seed_female_hq.wav",
            "Male_HQ": "seed_male_hq.wav",
            "Female": "seed_female.wav",
            "Male": "seed_male.wav",
        }
        filename = filename_map.get(target_name, "seed_female_hq.wav")
        path = os.path.join(assets_dir, filename)

        if not os.path.exists(path):
            print(f"[AIIA Warning] Fallback seed not found at {path}")
            return None

        try:
            waveform, sample_rate = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[AIIA Error] Failed to load fallback audio: {e}")
            return None

    def _audio_to_wav_path(self, audio_tensor, prefix="indextts"):
        """Convert ComfyUI AUDIO dict to a temporary wav file path."""
        waveform = audio_tensor["waveform"]  # (batch, channels, samples)
        sr = audio_tensor["sample_rate"]

        # Take first batch item, ensure 2D (channels, samples)
        wav = waveform[0] if waveform.dim() == 3 else waveform
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix=prefix, delete=False)
        torchaudio.save(tmp.name, wav.cpu(), sr)
        tmp.close()
        return tmp.name

    def generate(self, indextts_model, text, voice_preset="Female_HQ",
                 reference_audio=None,
                 emotion_audio=None,
                 emo_alpha=1.0,
                 happy=0.0, angry=0.0, sad=0.0, afraid=0.0,
                 disgusted=0.0, melancholic=0.0, surprised=0.0, calm=0.0,
                 use_random=False, seed=0):
        _ensure_indextts()

        import comfy.utils

        # Fallback: user reference_audio > voice_preset dropdown
        if reference_audio is None:
            print(f"[AIIA IndexTTS-2] No reference audio, using voice preset '{voice_preset}'.")
            reference_audio = self._load_fallback_audio(voice_preset)
            if reference_audio is None:
                raise ValueError(
                    f"Could not load voice preset '{voice_preset}'! "
                    "Please connect a reference audio or check assets/ directory."
                )
        else:
            print(f"[AIIA IndexTTS-2] Using user-provided reference audio (preset '{voice_preset}' ignored).")

        tts = indextts_model

        # --- Seed ---
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # --- Reference audio → temp wav ---
        ref_path = self._audio_to_wav_path(reference_audio, prefix="indextts_ref_")

        # --- Emotion audio → temp wav (optional) ---
        emo_path = None
        if emotion_audio is not None:
            emo_path = self._audio_to_wav_path(emotion_audio, prefix="indextts_emo_")

        # --- Emotion vector (if any slider > 0) ---
        emo_vector = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        has_emo_vector = any(v > 0.001 for v in emo_vector)
        if not has_emo_vector:
            emo_vector = None

        # --- Output temp file ---
        out_fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="indextts_out_")
        os.close(out_fd)

        # --- Generate ---
        try:
            print(f"[AIIA IndexTTS-2] Generating: text='{text[:50]}...', ref={ref_path}")
            if emo_vector:
                print(f"  Emotion vector: {emo_vector}, alpha={emo_alpha}")
            if emo_path:
                print(f"  Emotion audio: {emo_path}")

            # Run inference (with temporary transformers patches)
            with _active_transformers_patches(), torch.no_grad():
                tts.infer(
                    spk_audio_prompt=ref_path,
                    text=text,
                    output_path=out_path,
                    emo_audio_prompt=emo_path,
                    emo_alpha=emo_alpha,
                    emo_vector=emo_vector,
                    use_random=use_random,
                    verbose=True,
                )

            # --- Read output wav → ComfyUI AUDIO format ---
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                print("[AIIA IndexTTS-2] WARNING: Generation produced no output. Returning silence.")
                silence = torch.zeros(1, 1, 22050)  # 1 second silence
                return ({"waveform": silence, "sample_rate": 22050},)

            wav, sr = torchaudio.load(out_path)
            # wav shape: (channels, samples) → (1, channels, samples) for ComfyUI batch dim
            wav = wav.unsqueeze(0)

            print(f"[AIIA IndexTTS-2] Generated {wav.shape[-1] / sr:.2f}s audio at {sr}Hz")

            return ({"waveform": wav, "sample_rate": sr},)

        finally:
            # Cleanup temp files
            for p in [ref_path, emo_path, out_path]:
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


# ============================================================================
#  NODE REGISTRATION
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "AIIA_IndexTTS2_Loader": AIIA_IndexTTS2_Loader,
    "AIIA_IndexTTS2_TTS": AIIA_IndexTTS2_TTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_IndexTTS2_Loader": "IndexTTS-2 Loader",
    "AIIA_IndexTTS2_TTS": "IndexTTS-2 TTS",
}
