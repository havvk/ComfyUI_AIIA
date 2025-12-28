import torch

class AIIA_Audio_Info:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("info_text", "sample_rate", "duration_seconds", "channels", "shape_info")
    FUNCTION = "analyze_audio"
    CATEGORY = "AIIA/Audio"

    def analyze_audio(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        # Waveform shape: [Batch, Channels, Time] or [Channels, Time]
        # Usually ComfyUI standard is [Batch, Channels, Time]
        
        shape = waveform.shape
        
        if waveform.ndim == 3:
            batch_size = shape[0]
            channels = shape[1]
            frames = shape[2]
        elif waveform.ndim == 2:
            # Maybe [Channels, Time]
            batch_size = 1
            channels = shape[0]
            frames = shape[1]
        else:
            # 1D
            batch_size = 1
            channels = 1
            frames = shape[0]
            
        duration = frames / sample_rate
        
        info = (
            f"🔍 Audio Analysis Report:\n"
            f"------------------------\n"
            f"• Sample Rate  : {sample_rate} Hz\n"
            f"• Duration     : {duration:.2f} s\n"
            f"• Channels     : {channels}\n"
            f"• Batch Size   : {batch_size}\n"
            f"• Total Samples: {frames}\n"
            f"• Shape        : {list(shape)}"
        )
        
        print(f"[AIIA Audio Info]\n{info}")
        
        return (info, sample_rate, duration, channels, str(list(shape)))
