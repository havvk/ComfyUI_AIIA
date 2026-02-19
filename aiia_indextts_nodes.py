"""
AIIA IndexTTS-2 ComfyUI Nodes
Loader + TTS nodes for Bilibili's IndexTTS-2 zero-shot voice cloning model.
"""

import os
import sys
import re
import contextlib
import tempfile
import warnings

import torch
import torchaudio
import numpy as np

from pathlib import Path
from typing import Optional, List, Tuple

# --- Preload kaldifst to prevent hang when used with NeMo Diarization ---
# IndexTTS-2 seems to put the process (OpenMP/MKL/dlopen lock) in a state
# that causes kaldifst initialization to hang if imported LATER.
# By importing it here at startup (before IndexTTS runs), we ensure it
# initializes safely.
try:
    import kaldifst
except ImportError:
    pass


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





    _INDEXTTS_READY = True
    print(f"[AIIA] IndexTTS-2 loaded from: {_INDEXTTS_DIR}")





_SENTINEL = object()  # unique marker for "attribute didn't exist before"





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
                "use_cuda_kernel": ("BOOLEAN", {"default": True, "tooltip": "Use BigVGAN custom CUDA kernel for faster vocoder inference (NVIDIA GPU only)."}),
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
        
        from indextts.infer_v2 import IndexTTS2
        with _patch_indextts_loading(model_dir):
            tts = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=model_dir,
                use_fp16=use_fp16,
                use_cuda_kernel=use_cuda_kernel,
                use_deepspeed=False,
            )
        
        _INDEXTTS_MODEL_CACHE[cache_key] = tts
        print("[AIIA] IndexTTS-2 loaded successfully.")

        return (tts,)


# ============================================================================
#  EMOTION TAG MAPPING
# ============================================================================

# Emotion tag → IndexTTS-2 8-dim vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
_EMOTION_TAG_TO_VECTOR = {
    "happy":        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "excited":      [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
    "enthusiastic": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
    "proud":        [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    "angry":        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad":          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "disappointed": [0.0, 0.0, 0.7, 0.0, 0.0, 0.3, 0.0, 0.0],
    "nostalgic":    [0.0, 0.0, 0.4, 0.0, 0.0, 0.5, 0.0, 0.1],
    "afraid":       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "fearful":      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "anxious":      [0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.2, 0.0],
    "nervous":      [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0],
    "disgusted":    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "sarcastic":    [0.0, 0.3, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
    "melancholic":  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "surprised":    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "confused":     [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.6, 0.0],
    "mysterious":   [0.0, 0.0, 0.0, 0.2, 0.0, 0.3, 0.0, 0.3],
    "calm":         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "gentle":       [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
    "neutral":      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "serious":      [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "romantic":     [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
    "lazy":         [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.5],
    "gossip":       [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
    "innocent":     [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3],
}


# ============================================================================
#  TTS NODE
# ============================================================================

class AIIA_IndexTTS2_TTS:
    # Match emotion tags injected by Splitter/Annotator, e.g. [Happy], [Calm]
    _EMOTION_TAG_RE = re.compile(r'^\[(\w[\w\s]*?)\]\s*', re.MULTILINE)

    @staticmethod
    def _split_by_emotion_tags(text: str) -> List[Tuple[Optional[str], str]]:
        """
        Split text by lines. 
        - Lines starting with `[Tag]` get that emotion.
        - Lines without a tag get None (No Emotion). I.e. NO Inheritance.
        - Consecutive lines with the same emotion are merged.
        
        Input: 
            [Happy] Line 1
            Line 2 (No tag -> None)
            [Happy] Line 3
            [Happy] Line 4
            
        Output: 
            [("Happy", "Line 1"), (None, "Line 2"), ("Happy", "Line 3\\nLine 4")]
        """
        # Regex to match leading tag: [Happy] ...
        tag_pattern = re.compile(r'^\s*\[([\w\s]+?)\]\s*(.*)$')
        
        lines = text.split('\n')
        raw_segments = []
        
        for line in lines:
            m = tag_pattern.match(line)
            if m:
                tag = m.group(1)
                content = m.group(2)
                raw_segments.append((tag, content))
            else:
                raw_segments.append((None, line))
        
        if not raw_segments:
            return [(None, text)]

        # Merge consecutive segments with the same emotion (case-insensitive)
        merged = [raw_segments[0]]
        for tag, txt in raw_segments[1:]:
            prev_tag, prev_txt = merged[-1]
            
            same_emo = False
            if prev_tag is None and tag is None:
                same_emo = True
            elif prev_tag is not None and tag is not None:
                if prev_tag.strip().lower() == tag.strip().lower():
                    same_emo = True
            
            if same_emo:
                merged[-1] = (prev_tag, prev_txt + "\n" + txt)
            else:
                merged.append((tag, txt))
        
        # Filter out purely empty segments if they have a tag (avoid generating silence for just tags)
        # But keep None segments (for pauses)
        final_segments = []
        for tag, txt in merged:
            if txt.strip():
                final_segments.append((tag, txt))
                
        if not final_segments:
             return [(None, text)]

        return final_segments

    @staticmethod
    def _crossfade_segments(waveforms: list, sample_rate: int, crossfade_ms: int = 50, silence_ms: int = 100) -> torch.Tensor:
        """
        Join waveform segments with cosine fade-out / silence gap / fade-in.
        Each waveform is (C, N).
        """
        if not waveforms:
            return torch.zeros(1, 0)
        if len(waveforms) == 1:
            return waveforms[0]

        xfade_samples = int(sample_rate * crossfade_ms / 1000)
        silence_samples = int(sample_rate * silence_ms / 1000)

        result = waveforms[0]
        for i in range(1, len(waveforms)):
            curr = waveforms[i]

            # Fade out the tail of previous segment
            fo_len = min(xfade_samples, result.shape[-1])
            if fo_len >= 2:
                t_fo = torch.linspace(0, np.pi, fo_len, device=result.device)
                fade_out = 0.5 * (1.0 + torch.cos(t_fo))  # 1 → 0
                result = torch.cat([
                    result[..., :-fo_len],
                    result[..., -fo_len:] * fade_out,
                ], dim=-1)

            # Insert silence gap
            channels = result.shape[0] if result.dim() >= 2 else 1
            silence = torch.zeros(channels, silence_samples, device=result.device)

            # Fade in the head of next segment
            fi_len = min(xfade_samples, curr.shape[-1])
            if fi_len >= 2:
                t_fi = torch.linspace(0, np.pi, fi_len, device=curr.device)
                fade_in = 1.0 - 0.5 * (1.0 + torch.cos(t_fi))  # 0 → 1
                curr = torch.cat([
                    curr[..., :fi_len] * fade_in,
                    curr[..., fi_len:],
                ], dim=-1)

            result = torch.cat([result, silence, curr], dim=-1)

        return result

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
                "use_emo_text": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-detect emotion from text using built-in Qwen emotion model. Overrides emotion sliders."
                }),
                "emo_text": ("STRING", {
                    "default": "",
                    "tooltip": "Custom emotion text prompt (used with use_emo_text). Leave empty to use main text."
                }),
                "interval_silence": ("INT", {
                    "default": 200, "min": 0, "max": 2000, "step": 50,
                    "tooltip": "Silence duration (ms) inserted between text segments for long text."
                }),
                "max_text_tokens_per_segment": ("INT", {
                    "default": 120, "min": 30, "max": 500, "step": 10,
                    "tooltip": "Max tokens per text segment. Lower = more segments, higher = longer per-segment generation."
                }),
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

    def _infer_single_segment(self, tts, text, ref_path, out_path,
                              emo_path=None, emo_alpha=1.0, emo_vector=None,
                              use_emo_text=False, emo_text=None,
                              interval_silence=200, max_text_tokens_per_segment=120,
                              use_random=False):
        """Generate a single segment and return (waveform, sample_rate)."""
        with torch.no_grad():
            tts.infer(
                spk_audio_prompt=ref_path,
                text=text,
                output_path=out_path,
                emo_audio_prompt=emo_path,
                emo_alpha=emo_alpha,
                emo_vector=emo_vector,
                use_emo_text=use_emo_text,
                emo_text=emo_text if (emo_text and emo_text.strip()) else None,
                interval_silence=interval_silence,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                use_random=use_random,
                verbose=True,
            )

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return None, None

        wav, sr = torchaudio.load(out_path)
        return wav, sr

    def generate(self, indextts_model, text, voice_preset="Female_HQ",
                 reference_audio=None,
                 emotion_audio=None,
                 emo_alpha=1.0,
                 happy=0.0, angry=0.0, sad=0.0, afraid=0.0,
                 disgusted=0.0, melancholic=0.0, surprised=0.0, calm=0.0,
                 use_emo_text=False, emo_text="",
                 interval_silence=200, max_text_tokens_per_segment=120,
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

        # --- Emotion vector from sliders (if any slider > 0) ---
        slider_emo_vector = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        has_slider_emo = any(v > 0.001 for v in slider_emo_vector)
        if not has_slider_emo:
            slider_emo_vector = None

        # --- Detect emotion tags in text ---
        emotion_segments = self._split_by_emotion_tags(text)
        has_multi_emotion = len(emotion_segments) > 1 or (len(emotion_segments) == 1 and emotion_segments[0][0] is not None)

        if has_multi_emotion:
            print(f"[AIIA IndexTTS-2] Detected {len(emotion_segments)} emotion-tagged segment(s):")
            for i, (emo, seg) in enumerate(emotion_segments):
                print(f"  Segment {i}: [{emo}] {seg[:50]}...")

        # --- Output temp file ---
        out_fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="indextts_out_")
        os.close(out_fd)

        # --- Generate ---
        try:
            print(f"[AIIA IndexTTS-2] Generating: text='{text[:50]}...', ref={ref_path}")

            if has_multi_emotion:
                # ===== Per-segment emotion generation =====
                all_waveforms = []
                final_sr = 22050  # will be overwritten by actual sr

                for seg_idx, (seg_emotion, seg_text) in enumerate(emotion_segments):
                    # Determine emotion vector for this segment
                    seg_emo_vector = None
                    seg_use_emo_text = False
                    seg_emo_text = None
                    seg_emo_path = emo_path  # default: use global emotion audio

                    if seg_emotion:
                        tag_lower = seg_emotion.lower().strip()
                        if tag_lower in _EMOTION_TAG_TO_VECTOR:
                            # Known tag → use mapped vector
                            seg_emo_vector = list(_EMOTION_TAG_TO_VECTOR[tag_lower])
                            seg_emo_path = None  # tag overrides emotion audio
                            print(f"  [{seg_emotion}] → mapped vector: {seg_emo_vector}")
                        else:
                            # Unknown tag → fallback to QwenEmotion inference
                            seg_use_emo_text = True
                            seg_emo_text = seg_emotion
                            seg_emo_path = None
                            print(f"  [{seg_emotion}] → unknown tag, falling back to QwenEmotion inference")
                    else:
                        # No tag on this segment → use slider vector or global settings
                        seg_emo_vector = slider_emo_vector
                        seg_use_emo_text = use_emo_text
                        seg_emo_text = emo_text

                    print(f"  Generating segment {seg_idx+1}/{len(emotion_segments)}: '{seg_text[:40]}...'")

                    wav, sr = self._infer_single_segment(
                        tts, seg_text, ref_path, out_path,
                        emo_path=seg_emo_path,
                        emo_alpha=emo_alpha,
                        emo_vector=seg_emo_vector,
                        use_emo_text=seg_use_emo_text,
                        emo_text=seg_emo_text,
                        interval_silence=interval_silence,
                        max_text_tokens_per_segment=max_text_tokens_per_segment,
                        use_random=use_random,
                    )

                    if wav is not None:
                        all_waveforms.append(wav)
                        final_sr = sr
                    else:
                        print(f"  WARNING: Segment {seg_idx+1} produced no output, skipping.")

                if not all_waveforms:
                    print("[AIIA IndexTTS-2] WARNING: All segments produced no output. Returning silence.")
                    silence = torch.zeros(1, 1, 22050)
                    return ({"waveform": silence, "sample_rate": 22050},)

                # Crossfade join all segments
                final_wav = self._crossfade_segments(all_waveforms, final_sr)
                final_wav = final_wav.unsqueeze(0)  # (C, N) → (1, C, N)

                total_duration = final_wav.shape[-1] / final_sr
                print(f"[AIIA IndexTTS-2] Generated {len(all_waveforms)} segments, total {total_duration:.2f}s at {final_sr}Hz")

                return ({"waveform": final_wav, "sample_rate": final_sr},)

            else:
                # ===== Single segment (original path) =====
                if slider_emo_vector:
                    print(f"  Emotion vector: {slider_emo_vector}, alpha={emo_alpha}")
                if emo_path:
                    print(f"  Emotion audio: {emo_path}")

                wav, sr = self._infer_single_segment(
                    tts, text, ref_path, out_path,
                    emo_path=emo_path,
                    emo_alpha=emo_alpha,
                    emo_vector=slider_emo_vector,
                    use_emo_text=use_emo_text,
                    emo_text=emo_text,
                    interval_silence=interval_silence,
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    use_random=use_random,
                )

                if wav is None:
                    print("[AIIA IndexTTS-2] WARNING: Generation produced no output. Returning silence.")
                    silence = torch.zeros(1, 1, 22050)
                    return ({"waveform": silence, "sample_rate": 22050},)

                wav = wav.unsqueeze(0)  # (C, N) → (1, C, N)
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
