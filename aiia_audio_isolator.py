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
            },
            "optional": {
                "fade_duration_ms": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0, "tooltip": "拼接处的淡入淡出时长，防止出现爆音"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("isolated_audio", "segment_count")
    FUNCTION = "isolate_speaker"
    CATEGORY = "AIIA/audio"

    def isolate_speaker(self, audio, whisper_chunks, speaker_label, fade_duration_ms=10.0):
        if not isinstance(whisper_chunks, dict) or "chunks" not in whisper_chunks:
            print(f"警告: [AIIA Audio Isolator] 输入的 whisper_chunks 格式不正确。")
            return (audio, 0)

        waveform = audio["waveform"] # Shape: [Batch, Channels, Samples]
        sample_rate = audio["sample_rate"]
        
        # 提取匹配的片段
        matched_segments = []
        for chunk in whisper_chunks["chunks"]:
            if chunk.get("speaker") == speaker_label:
                start_time, end_time = chunk["timestamp"]
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # 边界检查
                if start_sample < waveform.shape[-1]:
                    end_sample = min(end_sample, waveform.shape[-1])
                    segment = waveform[:, :, start_sample:end_sample]
                    
                    if segment.shape[-1] > 0:
                        matched_segments.append(segment)

        if not matched_segments:
            print(f"警告: [AIIA Audio Isolator] 未找到说话人 {speaker_label} 的任何音频片段。返回空音频。")
            # 返回 1 秒静音以防止下游节点崩溃
            silent_waveform = torch.zeros((waveform.shape[0], waveform.shape[1], sample_rate))
            return ({"waveform": silent_waveform, "sample_rate": sample_rate}, 0)

        # 拼接处理
        # 如果设置了淡入淡出
        processed_segments = []
        fade_samples = int((fade_duration_ms / 1000.0) * sample_rate)
        
        for i, seg in enumerate(matched_segments):
            if fade_samples > 0 and seg.shape[-1] > fade_samples * 2:
                # 简单的线性淡入淡出
                fade_in = torch.linspace(0.0, 1.0, fade_samples, device=seg.device)
                fade_out = torch.linspace(1.0, 0.0, fade_samples, device=seg.device)
                
                seg_copy = seg.clone()
                seg_copy[:, :, :fade_samples] *= fade_in
                seg_copy[:, :, -fade_samples:] *= fade_out
                processed_segments.append(seg_copy)
            else:
                processed_segments.append(seg)

        # 合并张量
        final_waveform = torch.cat(processed_segments, dim=-1)
        
        print(f"--- [AIIA Audio Isolator] 已提取说话人 {speaker_label} 的 {len(matched_segments)} 个片段，总时长: {final_waveform.shape[-1]/sample_rate:.2f}秒 ---")

        return ({"waveform": final_waveform, "sample_rate": sample_rate}, len(matched_segments))

NODE_CLASS_MAPPINGS = {
    "AIIA_Audio_Speaker_Isolator": AIIA_Audio_Speaker_Isolator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Audio_Speaker_Isolator": "Audio Speaker Isolator (AIIA)"
}
