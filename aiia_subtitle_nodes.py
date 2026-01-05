import json
import datetime
import os
import random
import torchaudio
import folder_paths

class AIIA_Subtitle_Gen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments_info": ("STRING", {"forceInput": True}),
                "format": (["SRT", "ASS", "Both"], {"default": "SRT"}),
            },
            "optional": {
                "ass_style": ("STRING", {"default": "Default", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("srt_content", "ass_content")
    FUNCTION = "generate_subtitle"
    CATEGORY = "AIIA/Subtitle"

    def generate_subtitle(self, segments_info, format="SRT", ass_style="Default"):
        try:
            segments = json.loads(segments_info)
        except Exception as e:
            print(f"[AIIA Subtitle] Error parsing segments JSON: {e}")
            return ("", "")

        if not isinstance(segments, list):
            print("[AIIA Subtitle] Segments info must be a list of dicts.")
            return ("", "")

        srt_out = ""
        ass_out = ""

        if format in ["SRT", "Both"]:
            srt_out = self._generate_srt(segments)
        
        if format in ["ASS", "Both"]:
            ass_out = self._generate_ass(segments, ass_style)

        return (srt_out, ass_out)

    def _generate_srt(self, segments):
        output = []
        for i, seg in enumerate(segments):
            start = self._format_srt_time(seg["start"])
            end = self._format_srt_time(seg["end"])
            text = seg["text"]
            
            output.append(f"{i+1}")
            output.append(f"{start} --> {end}")
            output.append(f"{text}\n")
        
        return "\n".join(output)

    def _generate_ass(self, segments, style_name="Default"):
        # Basic ASS Header
        header = [
            "[Script Info]",
            "Title: AIIA Generated Subtitles",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: None",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: {style_name},Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,10,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        
        events = []
        for seg in segments:
            start = self._format_ass_time(seg["start"])
            end = self._format_ass_time(seg["end"])
            text = seg["text"].replace("\n", "\\N")
            speaker = seg.get("speaker", "Unknown")
            
            events.append(f"Dialogue: 0,{start},{end},{style_name},{speaker},0,0,0,,{text}")

        return "\n".join(header + events)

    def _format_srt_time(self, seconds):
        # HH:MM:SS,mmm
        td = datetime.timedelta(seconds=seconds)
        # timedeltas are days, seconds, microseconds
        # We need to manually format to handle hours > 24 if needed, but for subtitles within a request likely not.
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def _format_ass_time(self, seconds):
        # H:MM:SS.cs (centiseconds)
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        centis = int(td.microseconds / 10000)
        
        return f"{hours}:{minutes:02}:{secs:02}.{centis:02}"

class AIIA_Subtitle_Preview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subtitle_content": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "preview_subtitle"
    CATEGORY = "AIIA/Subtitle"

    def preview_subtitle(self, subtitle_content, unique_id, audio=None):
        audio_info = None
        if audio is not None:
            # Save audio to temp
            output_dir = folder_paths.get_temp_directory()
            filename = f"preview_{unique_id}_{random.randint(1000, 9999)}.wav"
            file_path = os.path.join(output_dir, filename)
            
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Ensure waveform is correct shape for save
            # torchaudio.save expects [channels, time]
            if waveform.ndim == 3: # [batch, channels, time]
                waveform = waveform.squeeze(0)
            
            torchaudio.save(file_path, waveform, sample_rate)
            
            audio_info = {
                "filename": filename,
                "type": "temp",
                "subfolder": ""
            }

        return {"ui": {"text": [subtitle_content], "audio": [audio_info] if audio_info else []}}

NODE_CLASS_MAPPINGS = {
    "AIIA_Subtitle_Gen": AIIA_Subtitle_Gen,
    "AIIA_Subtitle_Preview": AIIA_Subtitle_Preview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Subtitle_Gen": "📝 AIIA Subtitle Generation",
    "AIIA_Subtitle_Preview": "🎬 AIIA Subtitle Preview"
}
