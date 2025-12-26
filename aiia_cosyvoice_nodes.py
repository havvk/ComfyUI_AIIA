import torch
import numpy as np
import os
import random
import tempfile
import soundfile as sf
from typing import Dict, Any, Tuple, List
import folder_paths

class AIIA_CosyVoice_VoiceConversion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "source_audio": ("AUDIO",),
                "target_audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "chunk_size": ("INT", {"default": 25, "min": 10, "max": 29, "step": 1, "tooltip": "每次转换的音频秒数。建议不超过29秒以保证稳定性。"}),
                "overlap_size": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1, "tooltip": "切片间的重叠秒数，用于平滑拼接。"}),
            },
            "optional": {
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert_voice_unlimited"
    CATEGORY = "AIIA/Synthesis"

    def convert_voice_unlimited(self, model, source_audio, target_audio, speed, chunk_size, overlap_size, seed=42):
        cosyvoice_model = model["model"]
        sample_rate = cosyvoice_model.sample_rate # 通常是 24000
        
        # 1. 准备参考音频 (target_audio) - 限制在 30s 内
        target_waveform = target_audio["waveform"]
        target_sr = target_audio["sample_rate"]
        # 转换到模型采样率
        if target_sr != sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(target_sr, sample_rate)
            target_waveform = resampler(target_waveform)
        
        # 截取前 30s 作为参考
        max_target_samples = 30 * sample_rate
        if target_waveform.shape[-1] > max_target_samples:
            target_waveform = target_waveform[..., :max_target_samples]
        
        # 保存参考音频到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_target:
            target_path = tmp_target.name
            target_np = target_waveform.squeeze().cpu().numpy()
            if target_np.ndim == 2: target_np = target_np.T
            sf.write(target_path, target_np, sample_rate)

        # 2. 准备源音频 (source_audio)
        source_waveform = source_audio["waveform"]
        source_sr = source_audio["sample_rate"]
        if source_sr != sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(source_sr, sample_rate)
            source_waveform = resampler(source_waveform)
        
        source_waveform = source_waveform.squeeze() # [Channels, Samples] 或 [Samples]
        if source_waveform.ndim == 1:
            source_waveform = source_waveform.unsqueeze(0)
        
        total_samples = source_waveform.shape[-1]
        chunk_samples = chunk_size * sample_rate
        overlap_samples = overlap_size * sample_rate
        
        if total_samples <= 30 * sample_rate:
            # 短音频直接处理
            print(f"[AIIA CosyVoice] Short audio detected, processing normally...")
            result_waveform = self._inference_single_chunk(cosyvoice_model, source_waveform, target_path, speed, sample_rate, seed)
            return ({"waveform": result_waveform.unsqueeze(0), "sample_rate": sample_rate},)

        # 3. 长音频切片处理
        print(f"[AIIA CosyVoice] Long audio detected ({total_samples/sample_rate:.2f}s). Chunking...")
        
        final_segments = []
        step = chunk_samples - overlap_samples
        
        for start in range(0, total_samples, step):
            end = min(start + chunk_samples, total_samples)
            if start >= total_samples: break
            
            chunk = source_waveform[:, start:end]
            print(f"[AIIA CosyVoice] Processing chunk: {start/sample_rate:.1f}s - {end/sample_rate:.1f}s")
            
            converted_chunk = self._inference_single_chunk(cosyvoice_model, chunk, target_path, speed, sample_rate, seed)
            
            # 处理拼接
            if not final_segments:
                final_segments.append(converted_chunk)
            else:
                # 简单的线性淡入淡出拼接
                prev_chunk = final_segments[-1]
                # 计算实际重叠长度 (考虑到 speed 影响，CosyVoice 的输出长度是 source_len / speed)
                # 但 CosyVoice VC 模式下通常保持 1:1 时间轴（除非设置了 speed）
                # 我们假设输出与输入长度成比例
                actual_overlap = int(overlap_samples / speed)
                
                if actual_overlap > 0 and prev_chunk.shape[-1] > actual_overlap:
                    fade_out = torch.linspace(1.0, 0.0, actual_overlap, device=converted_chunk.device)
                    fade_in = torch.linspace(0.0, 1.0, actual_overlap, device=converted_chunk.device)
                    
                    # 混合重叠部分
                    overlap_part = prev_chunk[:, -actual_overlap:] * fade_out + converted_chunk[:, :actual_overlap] * fade_in
                    
                    final_segments[-1] = prev_chunk[:, :-actual_overlap]
                    final_segments.append(overlap_part)
                    final_segments.append(converted_chunk[:, actual_overlap:])
                else:
                    final_segments.append(converted_chunk)
            
            # 清理缓存
            torch.cuda.empty_cache()

        merged_waveform = torch.cat(final_segments, dim=-1)
        
        # 清理临时文件
        if os.path.exists(target_path): os.unlink(target_path)
        
        return ({"waveform": merged_waveform.unsqueeze(0).cpu(), "sample_rate": sample_rate},)

    def _inference_single_chunk(self, model, waveform, target_path, speed, sample_rate, seed):
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_source:
            source_path = tmp_source.name
            source_np = waveform.cpu().numpy()
            if source_np.ndim == 2: source_np = source_np.T
            sf.write(source_path, source_np, sample_rate)
        
        try:
            output = model.inference_vc(
                source_wav=source_path,
                prompt_wav=target_path,
                stream=False,
                speed=speed
            )
            
            all_speech = []
            for chunk in output:
                all_speech.append(chunk['tts_speech'])
            
            res = torch.cat(all_speech, dim=-1)
            return res
        finally:
            if os.path.exists(source_path): os.unlink(source_path)

NODE_CLASS_MAPPINGS = {
    "AIIA_CosyVoice_VoiceConversion": AIIA_CosyVoice_VoiceConversion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_CosyVoice_VoiceConversion": "Voice Conversion (AIIA Unlimited)"
}
