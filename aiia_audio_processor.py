import torch
import numpy as np

class AIIA_Audio_Silence_Splitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "max_duration": ("FLOAT", {"default": 27.0, "min": 5.0, "max": 29.0, "step": 0.1}),
                "silence_threshold": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0001}),
                "min_silence_duration": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("WHISPER_CHUNKS", "INT")
    RETURN_NAMES = ("whisper_chunks", "chunk_count")
    FUNCTION = "split_audio"
    CATEGORY = "AIIA/audio"

    def split_audio(self, audio, max_duration, silence_threshold, min_silence_duration):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # 强制在 CPU 处理以防 OOM
        waveform_cpu = waveform.cpu()
        
        # 计算能量
        # waveform shape: [B, C, T]
        energy = torch.abs(waveform_cpu).mean(dim=1).squeeze()
        if energy.ndim > 1: energy = energy.mean(dim=0)
        
        # 1. 识别静音区间
        is_silence = energy < silence_threshold
        sil_np = is_silence.numpy()
        
        # 寻找变化点
        diff = np.diff(sil_np.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if sil_np[0]: starts = np.insert(starts, 0, 0)
        if sil_np[-1]: ends = np.append(ends, len(sil_np) - 1)
        
        # 保证成对
        if len(starts) > len(ends): starts = starts[:len(ends)]
        elif len(ends) > len(starts): ends = ends[:len(starts)]
        
        # 过滤过短的静音
        # 修正变量名：使用参数传入的 min_silence_duration
        min_sil_samples = int(min_silence_duration * sample_rate)
        valid_silence_gaps = []
        for s, e in zip(starts, ends):
            if e - s >= min_sil_samples:
                valid_silence_gaps.append((s / sample_rate, e / sample_rate))
        
        # 2. 贪心构建
        total_duration = waveform_cpu.shape[-1] / sample_rate
        chunks = []
        current_start = 0.0
        
        while current_start < total_duration:
            target_end = current_start + max_duration
            if target_end >= total_duration:
                chunks.append([current_start, total_duration])
                break
            
            best_gap_mid = -1
            for g_start, g_end in valid_silence_gaps:
                g_mid = (g_start + g_end) / 2
                if current_start < g_mid <= target_end:
                    best_gap_mid = g_mid
                elif g_mid > target_end:
                    break
            
            if best_gap_mid != -1:
                chunks.append([current_start, best_gap_mid])
                current_start = best_gap_mid
            else:
                chunks.append([current_start, target_end])
                current_start = target_end

        whisper_chunks_data = {
            "text": "",
            "chunks": [
                {"timestamp": [round(c[0], 3), round(c[1], 3)], "text": f"Chunk {i}", "speaker": "AIIA_SMART_CHUNK"} 
                for i, c in enumerate(chunks)
            ],
            "language": ""
        }
        
        return (whisper_chunks_data, len(chunks))

class AIIA_Audio_PostProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_rate": (["44100", "48000", "24000", "22050", "Original"], {"default": "44100"}),
                "fade_length": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Fade in/out duration in seconds"}),
                "normalize": ("BOOLEAN", {"default": True, "tooltip": "Normalize to -1dB"}),
                "resampling_alg": (["sinc_best", "sinc_fast", "kaiser_best"], {"default": "sinc_best"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "AIIA/Audio"

    def process_audio(self, audio, target_rate, fade_length, normalize, resampling_alg):
        import torchaudio
        
        waveform = audio["waveform"] # [B, C, T] usually
        original_rate = audio["sample_rate"]
        
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
            
        # 1. Resample
        if target_rate != "Original":
            new_rate = int(target_rate)
            if new_rate != original_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_rate,
                    new_freq=new_rate,
                    resampling_method=resampling_alg,
                    dtype=waveform.dtype
                )
                waveform = resampler(waveform)
                original_rate = new_rate
        
        # 2. Fade In/Out
        if fade_length > 0:
            fade_samples = int(fade_length * original_rate)
            if fade_samples * 2 < waveform.shape[-1]:
                fade_in = torch.linspace(0, 1, fade_samples, device=waveform.device)
                fade_out = torch.linspace(1, 0, fade_samples, device=waveform.device)
                
                # Apply to last dim (time)
                waveform[..., :fade_samples] *= fade_in
                waveform[..., -fade_samples:] *= fade_out
        
        # 3. Normalize (to -1dB = 0.891)
        if normalize:
            peak = waveform.abs().max()
            if peak > 0:
                target_peak = 0.891 # -1.0 dB
                waveform = waveform * (target_peak / peak)
            
        return ({"waveform": waveform, "sample_rate": original_rate},)

NODE_CLASS_MAPPINGS = {
    "AIIA_Audio_Silence_Splitter": AIIA_Audio_Silence_Splitter,
    "AIIA_Audio_PostProcess": AIIA_Audio_PostProcess
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Audio_Silence_Splitter": "Audio Smart Chunker (Silence-based)",
    "AIIA_Audio_PostProcess": "Audio Post-Process (Resample/Fade/Norm)"
}
