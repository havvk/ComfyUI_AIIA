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
                "chunk_size": ("INT", {"default": 25, "min": 10, "max": 28, "step": 1, "tooltip": "建议保持在25秒左右"}),
                "overlap_size": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1, "tooltip": "重叠部分也会计入30秒限制"}),
            },
            "optional": {
                "whisper_chunks": ("WHISPER_CHUNKS",),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("AUDIO", "SPLICE_INFO")
    RETURN_NAMES = ("audio", "splice_info")
    FUNCTION = "convert_voice_unlimited"
    CATEGORY = "AIIA/Synthesis"

    def _find_best_split_point(self, waveform, target_idx, search_range_samples):
        start = max(0, target_idx - search_range_samples)
        end = min(waveform.shape[-1], target_idx + search_range_samples)
        if start >= end: return target_idx
        search_region = waveform[:, start:end]
        energy = torch.abs(search_region).mean(dim=0)
        window_size = 100
        if energy.shape[0] > window_size:
            energy = torch.conv1d(energy.unsqueeze(0).unsqueeze(0), 
                                 torch.ones(1, 1, window_size, device=energy.device) / window_size,
                                 padding=window_size//2).squeeze()
        min_idx = torch.argmin(energy).item()
        return start + min_idx

    def convert_voice_unlimited(self, model, source_audio, target_audio, speed, chunk_size, overlap_size, whisper_chunks=None, seed=42):
        cosyvoice_model = model["model"]
        sample_rate = cosyvoice_model.sample_rate
        
        target_waveform = target_audio["waveform"]
        if target_audio["sample_rate"] != sample_rate:
            import torchaudio
            target_waveform = torchaudio.transforms.Resample(target_audio["sample_rate"], sample_rate)(target_waveform)
        
        max_target_samples = 30 * sample_rate
        if target_waveform.shape[-1] > max_target_samples:
            target_waveform = target_waveform[..., :max_target_samples]
        
        # Normalize and Save Target
        if target_waveform.abs().max() > 1.0:
            target_waveform = target_waveform / target_waveform.abs().max()
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_target:
            target_path = tmp_target.name
            target_np = target_waveform.squeeze().cpu().numpy()
            if target_np.ndim == 2: target_np = target_np.T
            sf.write(target_path, target_np, sample_rate, subtype='FLOAT')

        source_waveform = source_audio["waveform"]
        if source_audio["sample_rate"] != sample_rate:
            import torchaudio
            source_waveform = torchaudio.transforms.Resample(source_audio["sample_rate"], sample_rate)(source_waveform)
        
        # Normalize Source
        if source_waveform.abs().max() > 1.0:
            source_waveform = source_waveform / source_waveform.abs().max()

        source_waveform = source_waveform.squeeze() 
        if source_waveform.ndim == 1: source_waveform = source_waveform.unsqueeze(0)
        total_samples = source_waveform.shape[-1]
        
        MAX_TOTAL_SEC = 29.8
        chunk_samples = chunk_size * sample_rate
        overlap_samples = overlap_size * sample_rate
        max_total_samples = int(MAX_TOTAL_SEC * sample_rate)
        
        # 记录切片信息用于调试
        splice_info = {
            "sample_rate": sample_rate,
            "total_samples": total_samples,
            "splice_points": []
        }

        if total_samples <= max_total_samples:
            result_waveform = self._inference_single_chunk(cosyvoice_model, source_waveform, target_path, speed, sample_rate, seed)
            if os.path.exists(target_path): os.unlink(target_path)
            return ({"waveform": result_waveform.unsqueeze(0), "sample_rate": sample_rate}, splice_info)

        # 3. 智能分块逻辑
        chunks_to_process = []
        is_pre_chunked = whisper_chunks and any(c.get("speaker") == "AIIA_SMART_CHUNK" for c in whisper_chunks.get("chunks", []))
        
        if is_pre_chunked:
            print(f"[AIIA CosyVoice] Detected Smart Chunker input. Following pre-defined segments strictly.")
            for c in whisper_chunks["chunks"]:
                s_time, e_time = c["timestamp"]
                s_idx = int(s_time * sample_rate)
                e_idx = int(e_time * sample_rate)
                e_idx_with_overlap = min(e_idx + overlap_samples, total_samples)
                chunks_to_process.append(source_waveform[:, s_idx:e_idx_with_overlap])
        else:
            current_start = 0
            search_range = 2 * sample_rate
            while current_start < total_samples:
                target_end_time = (current_start + chunk_samples) / sample_rate
                best_split_time = target_end_time
                if whisper_chunks and "chunks" in whisper_chunks:
                    closest_gap_dist = float('inf')
                    for i in range(len(whisper_chunks["chunks"]) - 1):
                        gap_mid = (whisper_chunks["chunks"][i]["timestamp"][1] + whisper_chunks["chunks"][i+1]["timestamp"][0]) / 2
                        if (gap_mid * sample_rate + overlap_samples) - current_start > max_total_samples: continue
                        dist = abs(gap_mid - target_end_time)
                        if dist < 5 and dist < closest_gap_dist:
                            closest_gap_dist = dist
                            best_split_time = gap_mid
                split_point = int(best_split_time * sample_rate)
                split_point = self._find_best_split_point(source_waveform, split_point, search_range)
                split_point = min(split_point, total_samples)
                if (split_point + overlap_samples) - current_start > max_total_samples:
                    split_point = current_start + max_total_samples - overlap_samples - 100
                actual_end = min(split_point + overlap_samples, total_samples)
                if actual_end >= total_samples:
                    chunks_to_process.append(source_waveform[:, current_start:])
                    break
                else:
                    chunks_to_process.append(source_waveform[:, current_start:actual_end])
                current_start = split_point
                if total_samples - current_start < sample_rate:
                    break

        # 4. 拼接逻辑 (Sacrificial Context Cross-fade)
        final_segments = []
        MAX_CROSS_FADE_DURATION = 0.05 # 50ms 强制最大淡入淡出
        max_xfade_samples = int(MAX_CROSS_FADE_DURATION * sample_rate)

        for i, chunk in enumerate(chunks_to_process):
            print(f"[AIIA CosyVoice] Processing chunk {i+1}/{len(chunks_to_process)}, actual input len: {chunk.shape[-1]/sample_rate:.2f}s")
            converted_chunk = self._inference_single_chunk(cosyvoice_model, chunk, target_path, speed, sample_rate, seed)
            
            if not final_segments:
                final_segments.append(converted_chunk)
            else:
                prev_chunk = final_segments[-1]
                # 计算重叠
                # overlap_samples 是输入时的重叠，对应输出需要除以 speed (虽然 CosyVoice 不一定会严格且线性地遵循 speed，但这是一个估算)
                estimated_overlap_samples = int(overlap_samples / speed)
                
                # 我们只使用仅仅够消除爆音的短 crossfade，剩下的重叠部分作为“牺牲上下文”被丢弃
                xfade_len = min(estimated_overlap_samples, max_xfade_samples)
                sacrificial_len = 0
                
                if estimated_overlap_samples > xfade_len:
                    sacrificial_len = estimated_overlap_samples - xfade_len

                # 确保 tensor 够长
                if prev_chunk.shape[-1] > (sacrificial_len + xfade_len):
                     # 1. 裁剪前一段的尾部 (Sacrificial Context)
                    trim_idx = prev_chunk.shape[-1] - sacrificial_len
                    prev_chunk_trimmed = prev_chunk[:, :trim_idx]
                    
                    # 2. 准备 Cross-fade
                    t = torch.linspace(0, np.pi, xfade_len, device=converted_chunk.device)
                    fade_out = 0.5 * (1.0 + torch.cos(t))
                    fade_in = 1.0 - fade_out
                    
                    prev_tail = prev_chunk_trimmed[:, -xfade_len:]
                    curr_head = converted_chunk[:, :xfade_len]
                    
                    # 只有当维度匹配时才做 xfade，否则直接拼接
                    if prev_tail.shape[-1] == xfade_len and curr_head.shape[-1] == xfade_len:
                        overlap_part = prev_tail * fade_out + curr_head * fade_in
                        
                        final_segments[-1] = prev_chunk_trimmed[:, :-xfade_len] # 移除 xfade 部分
                        final_segments.append(overlap_part)
                        final_segments.append(converted_chunk[:, xfade_len:])   # 添加 xfade 之后的部分
                        
                        # 记录拼接点 (相对于最终音频的采样点)
                        current_total_len = sum([seg.shape[-1] for seg in final_segments]) - converted_chunk.shape[-1] + xfade_len # 指向 overlap 中心
                        splice_info["splice_points"].append(current_total_len)

                    else:
                         # 维度不够，直接硬拼接
                        final_segments.append(converted_chunk)
                else:
                    final_segments.append(converted_chunk)
            
            torch.cuda.empty_cache()

        merged_waveform = torch.cat(final_segments, dim=-1)
        if os.path.exists(target_path): os.unlink(target_path)
        return ({"waveform": merged_waveform.unsqueeze(0).cpu(), "sample_rate": sample_rate}, splice_info)

    def _inference_single_chunk(self, model, waveform, target_path, speed, sample_rate, seed):
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_source:
            source_path = tmp_source.name
            source_np = waveform.cpu().numpy()
            if source_np.ndim == 2: source_np = source_np.T
            sf.write(source_path, source_np, sample_rate, subtype='FLOAT')
        try:
            output = model.inference_vc(source_wav=source_path, prompt_wav=target_path, stream=False, speed=speed)
            all_speech = [chunk['tts_speech'] for chunk in output]
            return torch.cat(all_speech, dim=-1)
        finally:
            if os.path.exists(source_path): os.unlink(source_path)

NODE_CLASS_MAPPINGS = {"AIIA_CosyVoice_VoiceConversion": AIIA_CosyVoice_VoiceConversion}
NODE_DISPLAY_NAME_MAPPINGS = {"AIIA_CosyVoice_VoiceConversion": "Voice Conversion (AIIA Unlimited)"}
