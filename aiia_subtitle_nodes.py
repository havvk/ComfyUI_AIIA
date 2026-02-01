import json
import datetime
import os
import random
import re
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
                "save_file": ("BOOLEAN", {"default": False, "label_on": "Save to Disk", "label_off": "Memory Only"}),
            },
            "optional": {
                "calibration_info": ("WHISPER_CHUNKS",),
                "ass_style": ("STRING", {"default": "Default", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "aiia_subtitle"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("srt_content", "ass_content")
    FUNCTION = "generate_subtitle"
    CATEGORY = "AIIA/Subtitle"
    OUTPUT_NODE = True

    def generate_subtitle(self, segments_info, format="SRT", save_file=False, ass_style="Default", filename_prefix="aiia_subtitle", calibration_info=None):
        try:
            segments = json.loads(segments_info)
        except Exception as e:
            print(f"[AIIA Subtitle] Error parsing segments JSON: {e}")
            return ("", "")

        if not isinstance(segments, list):
            print("[AIIA Subtitle] Segments info must be a list of dicts.")
            return ("", "")

        # --- Subtitle Calibration (v1.10.2) ---
        if calibration_info and "chunks" in calibration_info:
            print(f"[AIIA Subtitle] Calibrating {len(segments)} segments using {len(calibration_info['chunks'])} high-precision chunks.")
            segments = self._calibrate_segments(segments, calibration_info["chunks"])

        if not isinstance(segments, list):
            print("[AIIA Subtitle] Segments info must be a list of dicts.")
            return ("", "")

        srt_out = ""
        ass_out = ""

        if format in ["SRT", "Both"]:
            srt_out = self._generate_srt(segments)
        
        if format in ["ASS", "Both"]:
            ass_out = self._generate_ass(segments, ass_style)
            
        # File Saving Logic
        if save_file:
            output_dir = folder_paths.get_output_directory()
            
            # Timestamp (uniques)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format in ["SRT", "Both"]:
                srt_name = f"{filename_prefix}_{timestamp}.srt"
                srt_path = os.path.join(output_dir, srt_name)
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_out)
                print(f"[AIIA Subtitle] Saved SRT to: {srt_path}")
                
            if format in ["ASS", "Both"]:
                ass_name = f"{filename_prefix}_{timestamp}.ass"
                ass_path = os.path.join(output_dir, ass_name)
                with open(ass_path, "w", encoding="utf-8") as f:
                    f.write(ass_out)
                print(f"[AIIA Subtitle] Saved ASS to: {ass_path}")

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
        # 1. Collect unique speakers
        speakers = set()
        for seg in segments:
            speakers.add(seg.get("speaker", "Unknown"))
        
        # 2. Assign colors to speakers
        # Simple palette: White, Yellow, Cyan, Green, Orange, Pink, LightBlue
        palette = [
            "&H00FFFFFF", # White (Pure)
            "&H0000D7FF", # Gold/Yellow (Cinematic)
            "&H00FFFF00", # Cyan (Standard)
            "&H0000FF00", # Green (Lime)
            "&H000080FF", # Orange
            "&H00FF80FF", # Pink
            "&H00FFC0C0"  # LightBlue
        ]
        
        styles = []
        speaker_map = {}
        
        # Base Style String Template
        # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, ...
        # Base Style String Template
        # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, ...
        # Changes: Bold=1, Outline=2, Shadow=1 for professional look
        base_style = "Arial,40,{primary_color},&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1"
        
        sorted_speakers = sorted(list(speakers))
        for i, spk in enumerate(sorted_speakers):
            color = palette[i % len(palette)]
            # Create a style name safely (no spaces ideally, but ASS allows spaces)
            safe_spk = spk.replace(",", "").strip()
            # If user provided a custom style base name, maybe prefix it? 
            # But simpler to just use Speaker Name as Style Name for direct mapping.
            final_style_name = safe_spk if safe_spk else "Unknown"
            
            style_line = f"Style: {final_style_name},{base_style.format(primary_color=color)}"
            styles.append(style_line)
            speaker_map[spk] = final_style_name

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
        ] + styles + [
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
        ]
        
        events = []
        for seg in segments:
            start = self._format_ass_time(seg["start"])
            end = self._format_ass_time(seg["end"])
            text = seg["text"].replace("\n", "\\N")
            # [v1.10.8] Support Markdown Bold (**text**) -> ASS Bold ({\b1}text{\b0})
            text = re.sub(r'\*\*(.*?)\*\*', r'{\\b1}\1{\\b0}', text)
            speaker = seg.get("speaker", "Unknown")
            # Use the mapped style name
            style_for_event = speaker_map.get(speaker, "Default")
            
            events.append(f"Dialogue: 0,{start},{end},{style_for_event},{speaker},0,0,0,,{text}")

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

    def _calibrate_segments(self, segments, chunks):
        """
        Calibrate estimated segments using high-precision VAD chunks.
        Algorithm: Iterative sequence matching with speaker-centric isolation (v1.10.5).
        """
        if not chunks:
            return segments

        # 1. Ensure chunks are sorted chronologically
        sorted_chunks = sorted(chunks, key=lambda x: x["timestamp"][0])
        
        calibrated = []
        chunk_idx = 0
        num_chunks = len(sorted_chunks)
        
        def normalize_spk(s):
            if not s: return ""
            return str(s).lower().replace("speaker_", "").replace("speaker ", "").strip()

        for i, seg in enumerate(segments):
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_dur = seg_end - seg_start
            
            # --- Speaker-Centric Magic (v1.10.5) ---
            # 1. Find the "Winner Speaker" for this segment based on maximum overlap duration
            speaker_overlaps = {}
            # Large window for initial scan to be robust
            scan_idx = chunk_idx
            while scan_idx < num_chunks:
                c = sorted_chunks[scan_idx]
                c_start, c_end = c["timestamp"]
                # Hard break if the chunk is way past our segment
                if c_start > seg_end + 3.0: break
                
                # Calculate overlap duration
                overlap = min(seg_end, c_end) - max(seg_start, c_start)
                if overlap > 0:
                    spk = normalize_spk(c.get("speaker", "unknown"))
                    speaker_overlaps[spk] = speaker_overlaps.get(spk, 0.0) + overlap
                scan_idx += 1
            
            winner_spk = None
            if speaker_overlaps:
                # Get speaker with most accumulated overlap duration
                winner_spk = max(speaker_overlaps, key=speaker_overlaps.get)
            
            # 2. Find chunks belonging to the winner spk to use for snapping
            matched_chunks = []
            find_idx = chunk_idx
            lookahead_count = 0
            while find_idx < num_chunks and lookahead_count < 15:
                chunk = sorted_chunks[find_idx]
                c_start, c_end = chunk["timestamp"]
                c_spk = normalize_spk(chunk.get("speaker", "unknown"))
                
                overlap = min(seg_end, c_end) - max(seg_start, c_start)
                is_overlap = overlap > 0.05
                # Special case: tiny gap exactly at boundaries or start of video
                # [v1.10.6 Fix] Added c_end check to ensure we don't snap to a chunk that ends before the segment starts
                if not is_overlap and i == 0 and find_idx == 0:
                    if abs(c_start - seg_start) < 0.5 and c_end > seg_start - 0.1:
                        is_overlap = True
                
                if is_overlap and (winner_spk is None or c_spk == winner_spk):
                     matched_chunks.append(find_idx)
                
                if c_start > seg_end + 1.5: break
                find_idx += 1
                lookahead_count += 1
            
            if matched_chunks:
                # Use min/max over all matched chunks
                actual_starts = [sorted_chunks[idx]["timestamp"][0] for idx in matched_chunks]
                actual_ends = [sorted_chunks[idx]["timestamp"][1] for idx in matched_chunks]
                
                min_s = min(actual_starts)
                max_e = max(actual_ends)
                
                # [v1.10.7 Fix] Handle Multi-Segment Chunks (Shared Chunk Logic)
                # If we are reusing a chunk from previous segment, we must start AFTER previous segment
                new_start = min_s
                if i > 0:
                    prev_end = calibrated[-1]["end"]
                    if prev_end > new_start and prev_end < max_e:
                        new_start = prev_end

                # Determine if we should consume the chunk or share it
                # Check if next segment also wants this chunk (overlaps with the tail of this chunk)
                is_shared = False
                last_matched_idx = max(matched_chunks)
                chunk_end_time = sorted_chunks[last_matched_idx]["timestamp"][1]
                
                # Predicted end for this segment
                predicted_end = new_start + seg_dur
                
                # Only check for sharing if there is significant leftover time in the chunk
                if chunk_end_time - predicted_end > 0.5 and i + 1 < len(segments):
                    next_seg = segments[i+1]
                    # If next segment effectively overlaps the remainder of this chunk
                    if next_seg["start"] < chunk_end_time:
                         is_shared = True
                
                if is_shared:
                    # If shared, we limit our end to our duration (trust TTS relative duration)
                    new_end = predicted_end
                    # And we DO NOT advance past this chunk, so next segment can pick it up
                    chunk_idx = last_matched_idx 
                else:
                    # If not shared, we consume the full VAD chunk (snap to VAD end)
                    new_end = max_e
                    chunk_idx = last_matched_idx + 1
                
                seg["start"] = round(new_start, 3)
                seg["end"] = round(new_end, 3)
            else:
                # No match found, use fallback logic
                if i > 0:
                    prev_end = calibrated[-1]["end"]
                    if seg["start"] < prev_end:
                         diff = prev_end - seg["start"]
                         seg["start"] += diff
                         seg["end"] += diff
            
            calibrated.append(seg)
            
        return calibrated

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

class AIIA_Subtitle_To_Segments:
    """Convert SRT/ASS text or files into segments_info format."""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "subtitle_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "subtitle_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("segments_info",)
    FUNCTION = "convert"
    CATEGORY = "AIIA/Subtitle"

    def convert(self, subtitle_text, subtitle_path=""):
        import re
        content = subtitle_text.strip()
        
        # If path provided and exists, read it
        if subtitle_path and os.path.exists(subtitle_path):
            try:
                with open(subtitle_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read().strip()
            except Exception as e:
                print(f"[AIIA Subtitle Convert] Error reading file: {e}")

        if not content:
            return (json.dumps([]),)

        segments = []
        
        # Detect Format
        if "Dialogue:" in content:
            segments = self._parse_ass(content)
        elif " --> " in content:
            segments = self._parse_srt(content)
        else:
            print("[AIIA Subtitle Convert] Unknown format or empty content.")

        return (json.dumps(segments, ensure_ascii=False, indent=2),)

    def _parse_srt(self, text):
        import re
        segments = []
        # Pattern: Index, Time, Text
        # Handles \n and \r\n
        blocks = re.split(r'\n\s*\n', text.strip())
        for block in blocks:
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            if len(lines) < 2: continue
            
            # Find time line
            time_match = re.search(r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)', lines[0] if "-->" in lines[0] else lines[1])
            if not time_match: continue
            
            start_s = self._time_to_seconds(time_match.group(1), "srt")
            end_s = self._time_to_seconds(time_match.group(2), "srt")
            
            # Content is everything after the time line
            idx = 1 if "-->" in lines[0] else 2
            content = " ".join(lines[idx:])
            
            segments.append({
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "text": content,
                "speaker": "Unknown"
            })
        return segments

    def _parse_ass(self, text):
        import re
        segments = []
        # Look for Dialogue: lines
        for line in text.split('\n'):
            if line.startswith("Dialogue:"):
                # Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
                parts = line.split(',', 9)
                if len(parts) < 10: continue
                
                start_s = self._time_to_seconds(parts[1].strip(), "ass")
                end_s = self._time_to_seconds(parts[2].strip(), "ass")
                speaker = parts[4].strip() or "Unknown"
                content = parts[9].strip().replace('\\N', ' ').replace('\\n', ' ')
                # Clean ASS tags like {\pos(x,y)}
                content = re.sub(r'\{.*?\}', '', content)
                
                segments.append({
                    "start": round(start_s, 3),
                    "end": round(end_s, 3),
                    "text": content,
                    "speaker": speaker
                })
        return segments

    def _time_to_seconds(self, t_str, fmt):
        try:
            if fmt == "srt":
                # HH:MM:SS,mmm
                h, m, s_ms = t_str.split(':')
                s, ms = s_ms.split(',')
                return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
            else:
                # H:MM:SS.cc
                h, m, s_cs = t_str.split(':')
                s, cs = s_cs.split('.')
                return int(h)*3600 + int(m)*60 + int(s) + int(cs)/100.0
        except:
            return 0.0

NODE_CLASS_MAPPINGS = {
    "AIIA_Subtitle_Gen": AIIA_Subtitle_Gen,
    "AIIA_Subtitle_Preview": AIIA_Subtitle_Preview,
    "AIIA_Subtitle_To_Segments": AIIA_Subtitle_To_Segments
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Subtitle_Gen": "📝 AIIA Subtitle Generation",
    "AIIA_Subtitle_Preview": "🎬 AIIA Subtitle Preview",
    "AIIA_Subtitle_To_Segments": "🔄 AIIA Subtitle to Segments"
}
