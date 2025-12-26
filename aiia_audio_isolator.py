import torch
import numpy as np

class AIIA_Audio_Speaker_Isolator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "whisper_chunks": ("WHISPER_CHUNKS",),
                "speaker_label": ("STRING", {"default": "SPEAKER_00"}),
                "isolation_mode": (["Maintain Duration", "Concatenate"], {"default": "Maintain Duration"}),
            },
            "optional": {
                "fade_duration_ms": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "在声音开始和结束处添加淡入淡出，防止爆音"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("isolated_audio", "segment_count")
    FUNCTION = "isolate_speaker"
    CATEGORY = "AIIA/audio"

    def isolate_speaker(self, audio, whisper_chunks, speaker_label, isolation_mode, fade_duration_ms=10.0):
        if not isinstance(whisper_chunks, dict) or "chunks" not in whisper_chunks:
            print(f"警告: [AIIA Audio Isolator] 输入的 whisper_chunks 格式不正确。")
            return (audio, 0)

        waveform = audio["waveform"] # Shape: [Batch, Channels, Samples]
        sample_rate = audio["sample_rate"]
        fade_samples = int((fade_duration_ms / 1000.0) * sample_rate)
        
        # 准备输出容器
        if isolation_mode == "Maintain Duration":
            # 创建等长的静音张量
            final_waveform = torch.zeros_like(waveform)
        else:
            processed_segments = []

        matched_count = 0
        for chunk in whisper_chunks["chunks"]:
            if chunk.get("speaker") == speaker_label:
                start_time, end_time = chunk["timestamp"]
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # 边界检查
                if start_sample < waveform.shape[-1]:
                    end_sample = min(end_sample, waveform.shape[-1])
                    seg_len = end_sample - start_sample
                    if seg_len <= 0: continue
                    
                    segment = waveform[:, :, start_sample:end_sample].clone()
                    
                    # 应用淡入淡出处理
                    if fade_samples > 0 and seg_len > fade_samples * 2:
                        fade_in = torch.linspace(0.0, 1.0, fade_samples, device=segment.device)
                        fade_out = torch.linspace(1.0, 0.0, fade_samples, device=segment.device)
                        segment[:, :, :fade_samples] *= fade_in
                        segment[:, :, -fade_samples:] *= fade_out
                    
                    if isolation_mode == "Maintain Duration":
                        # 将片段放回原始位置
                        final_waveform[:, :, start_sample:end_sample] = segment
                    else:
                        processed_segments.append(segment)
                    
                    matched_count += 1

        if matched_count == 0:
            print(f"警告: [AIIA Audio Isolator] 未找到说话人 {speaker_label} 的片段。")
            if isolation_mode == "Maintain Duration":
                return ({"waveform": torch.zeros_like(waveform), "sample_rate": sample_rate}, 0)
            else:
                silent = torch.zeros((waveform.shape[0], waveform.shape[1], sample_rate))
                return ({"waveform": silent, "sample_rate": sample_rate}, 0)

        if isolation_mode == "Concatenate":
            final_waveform = torch.cat(processed_segments, dim=-1)
        
        print(f"--- [AIIA Audio Isolator] 模式: {isolation_mode}, 说话人: {speaker_label}, 片段数: {matched_count}, 最终时长: {final_waveform.shape[-1]/sample_rate:.2f}秒 ---")

        return ({"waveform": final_waveform, "sample_rate": sample_rate}, matched_count)

NODE_CLASS_MAPPINGS = {
    "AIIA_Audio_Speaker_Isolator": AIIA_Audio_Speaker_Isolator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Audio_Speaker_Isolator": "Audio Speaker Isolator (AIIA)"
}