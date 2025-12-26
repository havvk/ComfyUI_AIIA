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
                "chunk_size": ("INT", {"default": 25, "min": 10, "max": 29, "step": 1}),
                "overlap_size": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
            },
            "optional": {
                "whisper_chunks": ("WHISPER_CHUNKS",),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert_voice_unlimited"
    CATEGORY = "AIIA/Synthesis"

    def _find_best_split_point(self, waveform, target_idx, search_range_samples):
        """
        在目标索引附近寻找能量最低（静音）的点。
        """
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
        
        # 1. 准备参考音频
        target_waveform = target_audio["waveform"]
        if target_audio["sample_rate"] != sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(target_audio["sample_rate"], sample_rate)
            target_waveform = resampler(target_waveform)
        
        max_target_samples = 30 * sample_rate
        if target_waveform.shape[-1] > max_target_samples:
            target_waveform = target_waveform[..., :max_target_samples]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_target:
            target_path = tmp_target.name
            target_np = target_waveform.squeeze().cpu().numpy()
            if target_np.ndim == 2: target_np = target_np.T
            sf.write(target_path, target_np, sample_rate)

        # 2. 准备源音频
        source_waveform = source_audio["waveform"]
        if source_audio["sample_rate"] != sample_rate:
            import torchaudio
            resampler = torchaudio.transforms.Resample(source_audio["sample_rate"], sample_rate)
            source_waveform = resampler(source_waveform)
        
        source_waveform = source_waveform.squeeze() 
        if source_waveform.ndim == 1: source_waveform = source_waveform.unsqueeze(0)
        
        total_samples = source_waveform.shape[-1]
        chunk_samples = chunk_size * sample_rate
        overlap_samples = overlap_size * sample_rate
        search_range = 2 * sample_rate 
        
        if total_samples <= 30 * sample_rate:
            result_waveform = self._inference_single_chunk(cosyvoice_model, source_waveform, target_path, speed, sample_rate, seed)
            if os.path.exists(target_path): os.unlink(target_path)
            return ({"waveform": result_waveform.unsqueeze(0), "sample_rate": sample_rate},)

        print(f"[AIIA CosyVoice] Long audio detected. Whisper chunks provided: {whisper_chunks is not None}")
        
        # 3. 智能分块逻辑
        chunks_to_process = []
        current_start = 0
        
        while current_start < total_samples:
            target_time = (current_start + chunk_samples) / sample_rate
            best_split_time = target_time
            
            # 如果有 whisper_chunks，尝试寻找最佳语义切点
            if whisper_chunks and "chunks" in whisper_chunks:
                # 寻找最接近 target_time 的缝隙
                closest_gap_dist = float('inf')
                for i in range(len(whisper_chunks["chunks"]) - 1):
                    gap_start = whisper_chunks["chunks"][i]["timestamp"][1]
                    gap_end = whisper_chunks["chunks"][i+1]["timestamp"][0]
                    gap_mid = (gap_start + gap_end) / 2
                    
                    # 缝隙不能让 chunk 超过 30s 限制
                    if gap_mid * sample_rate - current_start > 29.5 * sample_rate:
                        continue
                    
                    dist = abs(gap_mid - target_time)
                    if dist < closest_gap_dist and dist < 5: # 5秒内的缝隙都考虑
                        closest_gap_dist = dist
                        best_split_time = gap_mid
            
            split_point = int(best_split_time * sample_rate)
            # 应用物理层静音微调
            split_point = self._find_best_split_point(source_waveform, split_point, search_range)
            
            # 确保不越界且不违反 30s 限制
            split_point = min(split_point, total_samples)
            if split_point - current_start > 29.5 * sample_rate:
                split_point = current_start + int(29 * sample_rate)

            actual_end = min(split_point + overlap_samples, total_samples)
            chunks_to_process.append(source_waveform[:, current_start:actual_end])
            current_start = split_point
            if current_start >= total_samples - sample_rate: # 剩不到1秒就结束
                if current_start < total_samples:
                    chunks_to_process[-1] = source_waveform[:, current_start - (split_point - chunks_to_process[-1].shape[-1]):]
                break

        final_segments = []
        for i, chunk in enumerate(chunks_to_process):
            print(f"[AIIA CosyVoice] Processing chunk {i+1}/{len(chunks_to_process)}, len: {chunk.shape[-1]/sample_rate:.1f}s")
            converted_chunk = self._inference_single_chunk(cosyvoice_model, chunk, target_path, speed, sample_rate, seed)
            
            if not final_segments:
                final_segments.append(converted_chunk)
            else:
                prev_chunk = final_segments[-1]
                chunk_overlap_samples = int(overlap_samples / speed)
                if chunk_overlap_samples > 0 and prev_chunk.shape[-1] > chunk_overlap_samples:
                    t = torch.linspace(0, np.pi, chunk_overlap_samples, device=converted_chunk.device)
                    fade_out = 0.5 * (1.0 + torch.cos(t))
                    fade_in = 1.0 - fade_out
                    overlap_part = prev_chunk[:, -chunk_overlap_samples:] * fade_out + converted_chunk[:, :chunk_overlap_samples] * fade_in
                    final_segments[-1] = prev_chunk[:, :-chunk_overlap_samples]
                    final_segments.append(overlap_part)
                    final_segments.append(converted_chunk[:, chunk_overlap_samples:])
                else:
                    final_segments.append(converted_chunk)
            torch.cuda.empty_cache()

        merged_waveform = torch.cat(final_segments, dim=-1)
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
            output = model.inference_vc(source_wav=source_path, prompt_wav=target_path, stream=False, speed=speed)
            all_speech = [chunk['tts_speech'] for chunk in output]
            return torch.cat(all_speech, dim=-1)
        finally:
            if os.path.exists(source_path): os.unlink(source_path)

NODE_CLASS_MAPPINGS = {"AIIA_CosyVoice_VoiceConversion": AIIA_CosyVoice_VoiceConversion}
NODE_DISPLAY_NAME_MAPPINGS = {"AIIA_CosyVoice_VoiceConversion": "Voice Conversion (AIIA Unlimited)"}
