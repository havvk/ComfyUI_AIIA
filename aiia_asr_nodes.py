import torch
import os
import json
import tempfile
import numpy as np
import soundfile as sf
import folder_paths

# --- 模型路径初始化 ---
_FUNASR_MODELS_DIR = None
_AVAILABLE_MODELS = {}

try:
    _models_base = os.path.join(folder_paths.base_path, "models", "funasr")
    if os.path.isdir(_models_base):
        _FUNASR_MODELS_DIR = _models_base
        for entry in os.listdir(_models_base):
            full_path = os.path.join(_models_base, entry)
            if os.path.isdir(full_path):
                _AVAILABLE_MODELS[entry] = full_path
                print(f"[AIIA ASR] 发现模型: {entry} -> {full_path}")
    else:
        print(f"[AIIA ASR] 警告: funasr 模型目录不存在: {_models_base}")
except Exception as e:
    print(f"[AIIA ASR] 模型路径初始化错误: {e}")


class AIIA_ASR:
    """通用 ASR 语音识别节点，基于 FunASR，支持字级时间戳输出。"""

    NODE_NAME = "AIIA ASR"
    _model_cache = {}  # 类级别模型缓存: {model_key: model_instance}

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = list(_AVAILABLE_MODELS.keys()) if _AVAILABLE_MODELS else ["NO_MODELS_FOUND"]
        default_model = "paraformer-zh" if "paraformer-zh" in _AVAILABLE_MODELS else model_choices[0]

        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (model_choices, {"default": default_model}),
            },
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "batch_size_s": ("INT", {
                    "default": 300, "min": 1, "max": 3600, "step": 10,
                    "tooltip": "以秒为单位的动态 batch 大小。越大越快但占用更多显存。"
                }),
                "hotword": ("STRING", {
                    "default": "",
                    "tooltip": "热词列表，每行一个词。提高这些词的识别准确率。"
                }),
            }
        }

    RETURN_TYPES = ("ASR_RESULT", "STRING",)
    RETURN_NAMES = ("asr_result", "text",)
    FUNCTION = "recognize"
    CATEGORY = "AIIA/Audio"

    def _ensure_model(self, model_name: str, device: str):
        """加载或从缓存获取模型实例。"""
        cache_key = f"{model_name}_{device}"
        if cache_key in self._model_cache:
            print(f"[{self.NODE_NAME}] 使用缓存模型: {model_name} on {device}")
            return self._model_cache[cache_key]

        model_path = _AVAILABLE_MODELS.get(model_name)
        if not model_path:
            raise RuntimeError(f"模型 '{model_name}' 未找到。可用模型: {list(_AVAILABLE_MODELS.keys())}")

        from funasr import AutoModel

        # 检测是否为 SenseVoice 系列（需要 trust_remote_code）
        is_sensevoice = "sensevoice" in model_name.lower()

        print(f"[{self.NODE_NAME}] 加载模型: {model_path} on {device}...")
        model = AutoModel(
            model=model_path,
            device=device,
            disable_update=True,
            trust_remote_code=is_sensevoice,
        )
        print(f"[{self.NODE_NAME}] 模型加载完成。")

        self._model_cache[cache_key] = model
        return model

    def _audio_to_numpy(self, audio: dict) -> tuple:
        """将 ComfyUI AUDIO 格式转换为 16kHz mono numpy 数组。"""
        waveform = audio["waveform"]  # (batch, channels, samples)
        sample_rate = audio["sample_rate"]

        # 取第一个 batch
        if waveform.ndim == 3:
            wav = waveform[0]
        else:
            wav = waveform

        # 转 mono
        if wav.ndim == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        elif wav.ndim == 2:
            wav = wav.squeeze(0)

        wav_np = wav.cpu().numpy().astype(np.float32)

        # 重采样到 16kHz（FunASR 要求）
        if sample_rate != 16000:
            try:
                import torchaudio.functional as F
                wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)
                wav_resampled = F.resample(wav_tensor, sample_rate, 16000)
                wav_np = wav_resampled.squeeze(0).numpy()
                print(f"[{self.NODE_NAME}] 重采样: {sample_rate}Hz -> 16000Hz")
            except ImportError:
                # 如果 torchaudio 不可用，写临时文件让 FunASR 自行处理
                print(f"[{self.NODE_NAME}] 警告: torchaudio 不可用，尝试直接传入音频")
            sample_rate = 16000

        return wav_np, sample_rate

    def recognize(self, audio, model, device="cuda", batch_size_s=300, hotword=""):
        log = f"[{self.NODE_NAME}]"

        if model == "NO_MODELS_FOUND":
            error_result = {
                "text": "",
                "words": [],
                "error": "未找到 FunASR 模型。请将模型放在 ComfyUI/models/funasr/ 目录下。"
            }
            return (error_result, "")

        # 验证音频
        if audio is None or "waveform" not in audio:
            error_result = {"text": "", "words": [], "error": "输入音频无效"}
            return (error_result, "")

        wav_np, sr = self._audio_to_numpy(audio)
        duration = len(wav_np) / sr
        print(f"{log} 音频时长: {duration:.2f}s, 采样率: {sr}Hz")

        if duration < 0.1:
            print(f"{log} 音频太短 ({duration:.3f}s)，跳过识别")
            return ({"text": "", "words": []}, "")

        # 加载模型
        asr_model = self._ensure_model(model, device)

        # 构建生成参数
        generate_kwargs = {
            "input": wav_np,
            "batch_size_s": batch_size_s,
        }

        # 热词支持（仅 Paraformer 支持）
        if hotword and hotword.strip() and "paraformer" in model.lower():
            generate_kwargs["hotword"] = hotword.strip()
            print(f"{log} 使用热词: {hotword.strip()[:50]}...")

        # SenseVoice 特殊参数
        if "sensevoice" in model.lower():
            generate_kwargs["language"] = "auto"
            generate_kwargs["use_itn"] = True

        print(f"{log} 开始识别...")
        results = asr_model.generate(**generate_kwargs)

        if not results or len(results) == 0:
            print(f"{log} 识别结果为空")
            return ({"text": "", "words": []}, "")

        result = results[0]
        raw_text = result.get("text", "")
        raw_timestamps = result.get("timestamp", [])

        # 构建 words 列表
        words = []
        if raw_timestamps and raw_text:
            # FunASR paraformer: text 是空格分隔的词, timestamp 是 [[start_ms, end_ms], ...]
            text_tokens = raw_text.split()
            if len(text_tokens) == len(raw_timestamps):
                for token, ts in zip(text_tokens, raw_timestamps):
                    words.append({
                        "word": token,
                        "start": round(ts[0] / 1000.0, 3),  # ms -> s
                        "end": round(ts[1] / 1000.0, 3),
                    })
            else:
                print(f"{log} 警告: 词数 ({len(text_tokens)}) 与时间戳数 ({len(raw_timestamps)}) 不匹配")
                # 尽力匹配
                for i, ts in enumerate(raw_timestamps):
                    token = text_tokens[i] if i < len(text_tokens) else "?"
                    words.append({
                        "word": token,
                        "start": round(ts[0] / 1000.0, 3),
                        "end": round(ts[1] / 1000.0, 3),
                    })

        # 去掉空格，构建完整文本
        clean_text = raw_text.replace(" ", "") if raw_text else ""

        asr_result = {
            "text": clean_text,
            "words": words,
        }

        print(f"{log} 识别完成: {len(words)} 个词, 文本: {clean_text[:80]}...")
        return (asr_result, clean_text)


# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_ASR": AIIA_ASR,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_ASR": "🎙️ AIIA ASR (Word Timestamps)",
}
