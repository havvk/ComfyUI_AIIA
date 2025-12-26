import torch
import numpy as np

class AIIA_Audio_Speaker_Merge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "duration_mode": (["Longest", "Shortest", "Audio 1", "Audio 2", "Specified"], {"default": "Longest"}),
                "specified_duration": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 10000.0, "step": 0.1, "tooltip": "当模式为 Specified 时生效"}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "是否自动归一化音量以防止叠加导致的爆音"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("merged_audio",)
    FUNCTION = "merge_audio"
    CATEGORY = "AIIA/audio"

    def merge_audio(self, audio_1, audio_2, duration_mode, specified_duration, normalize):
        waveform_1 = audio_1["waveform"] # [B, C, T]
        waveform_2 = audio_2["waveform"]
        sr_1 = audio_1["sample_rate"]
        sr_2 = audio_2["sample_rate"]

        if sr_1 != sr_2:
            print(f"警告: [AIIA Audio Merger] 两段音频采样率不一致 ({sr_1} vs {sr_2})。将以第一段为准。")
        
        # 确定目标采样点数
        len_1 = waveform_1.shape[-1]
        len_2 = waveform_2.shape[-1]
        
        if duration_mode == "Longest":
            target_len = max(len_1, len_2)
        elif duration_mode == "Shortest":
            target_len = min(len_1, len_2)
        elif duration_mode == "Audio 1":
            target_len = len_1
        elif duration_mode == "Audio 2":
            target_len = len_2
        else: # Specified
            target_len = int(specified_duration * sr_1)

        # 统一 Batch 和 Channel 数 (取最大值)
        max_b = max(waveform_1.shape[0], waveform_2.shape[0])
        max_c = max(waveform_1.shape[1], waveform_2.shape[1])

        def prepare_waveform(wf, target_t, b, c):
            # 扩展 B 和 C
            new_wf = wf.repeat(b // wf.shape[0], c // wf.shape[1], 1)
            # 处理 T (截断或填充)
            if new_wf.shape[-1] > target_t:
                return new_wf[:, :, :target_t]
            elif new_wf.shape[-1] < target_t:
                padding = torch.zeros((b, c, target_t - new_wf.shape[-1]), device=wf.device)
                return torch.cat([new_wf, padding], dim=-1)
            return new_wf

        wf1_final = prepare_waveform(waveform_1, target_len, max_b, max_c)
        wf2_final = prepare_waveform(waveform_2, target_len, max_b, max_c)

        # 合并
        merged_wf = wf1_final + wf2_final

        # 归一化处理
        if normalize:
            max_val = torch.max(torch.abs(merged_wf))
            if max_val > 1.0:
                merged_wf /= max_val
                print(f"信息: [AIIA Audio Merger] 检测到电平超限，已自动归一化。")

        return ({"waveform": merged_wf, "sample_rate": sr_1},)

NODE_CLASS_MAPPINGS = {
    "AIIA_Audio_Speaker_Merge": AIIA_Audio_Speaker_Merge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Audio_Speaker_Merge": "Audio Speaker Merger (AIIA)"
}
