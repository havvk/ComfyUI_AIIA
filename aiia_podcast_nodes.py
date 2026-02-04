
import json
import re

AIIA_EMOTION_LIST = [
    "None", "开心 (Happy)", "悲伤 (Sad)", "生气 (Angry)", "兴奋 (Excited)", 
    "温柔 (Gentle)", "严肃 (Serious)", "恐惧 (Fearful)", "惊讶 (Surprised)", 
    "低语 (Whispering)", "呐喊 (Shouting)", "羞涩 (Shy)", "诱惑 (Seductive)", 
    "哭腔 (Crying)", "笑声 (Laughter)", "尴尬 (Embarrassed)", "失望 (Disappointed)", 
    "自豪 (Proud)", "疑惑 (Doubtful)", "焦虑 (Anxious)", "平静 (Calm)"
]

class AIIA_Podcast_Script_Parser:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "script_text": ("STRING", {
                    "multiline": True, 
                    "default": "A: 大家好，欢迎收听我们的播客。\nB: 是的，今天我们要聊一个很有趣的话题。\n(Pause 0.5)\nA: [开心] 没错，就是关于 AI 的未来！",
                    "placeholder": "输入剧本，格式：\n角色名: 台词內容\n(Pause 秒数)\n[情感] 台词"
                }),
            },
            "optional": {
                "speaker_mapping": ("STRING", {
                    "multiline": True, 
                    "default": "A=Speaker_A\nB=Speaker_B",
                    "placeholder": "角色映射 (可选):\nA=Speaker_01\nB=Speaker_02"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("dialogue_json", "speaker_list", "full_script_json")
    FUNCTION = "parse_script"
    CATEGORY = "AIIA/Podcast"

    def parse_script(self, script_text, speaker_mapping=""):
        lines = script_text.strip().split('\n')
        dialogue = []
        speakers = set()
        
        # 解析映射
        mapping = {}
        if speaker_mapping:
            for line in speaker_mapping.strip().split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    mapping[k.strip()] = v.strip()

        current_speaker = None
        current_visual = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Filter comments
            if line.startswith("#"):
                continue

            # 检测暂停 (Pause 0.5)
            pause_match = re.match(r'^\((Pause|Wait)\s*(\d*\.?\d+)\s*s?\)$', line, re.IGNORECASE)
            if pause_match:
                duration = float(pause_match.group(2))
                dialogue.append({
                    "type": "pause",
                    "duration": duration
                })
                continue

            # 检测 Visual 标签 (Visual: https://...)
            # 支持: (Visual: ./img.png) or (Visual: https://example.com)
            visual_match = re.match(r'^\(Visual:\s*(.+?)\)$', line, re.IGNORECASE)
            if visual_match:
                current_visual = visual_match.group(1).strip()
                continue
            
            # 检测普通台词 Speaker: Text
            # 支持中文冒号和英文冒号
            parts = re.split(r'[:：]', line, 1)
            if len(parts) == 2:
                speaker_raw = parts[0].strip()
                content = parts[1].strip()
                
                # 应用映射
                speaker_id = mapping.get(speaker_raw, speaker_raw)
                speakers.add(speaker_id)
                current_speaker = speaker_id
                
                # 解析情感 [Happy] Text
                emotion = None
                emotion_match = re.match(r'^\[(.*?)\]\s*(.*)', content)
                if emotion_match:
                    emotion = emotion_match.group(1)
                    content = emotion_match.group(2)
                
                item = {
                    "type": "speech",
                    "speaker": speaker_id,
                    "text": content,
                    "emotion": emotion
                }
                
                # 如果有暂存的 Visual 标签，附加到这句话
                if current_visual:
                    item["visual"] = current_visual
                    current_visual = None
                
                dialogue.append(item)
            else:
                # 可能是延续上一句的内容，或者无法解析
                # 简单起见，如果延续上一句，我们追加到上一句
                # 但更安全的做法是忽略或作为新的一句（但这需要speaker）
                # 这里假设格式必须严格
                pass

        speaker_list = sorted(list(speakers))
        
        # Build clean dialogue for TTS (without Visual tags) to enable caching
        clean_dialogue = []
        for item in dialogue:
            clean_item = item.copy()
            if "visual" in clean_item:
                del clean_item["visual"]
            clean_dialogue.append(clean_item)

        return (json.dumps(clean_dialogue, ensure_ascii=False, indent=2), ",".join(speaker_list), json.dumps(dialogue, ensure_ascii=False, indent=2))

class AIIA_Segment_Merge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segments_info": ("STRING", {"forceInput": True}),
                "full_script": ("STRING", {"forceInput": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_segments",)
    FUNCTION = "merge_visuals"
    CATEGORY = "AIIA/Utils"

    def merge_visuals(self, segments_info, full_script):
        try:
            segments = json.loads(segments_info)
            script_items = json.loads(full_script)
            
            # Filter script items to only speech to match segments 1:1
            speech_items = [item for item in script_items if item.get("type") == "speech"]
            
            if len(segments) != len(speech_items):
                print(f"[AIIA Merge] Warning: Segment count ({len(segments)}) does not match script speech count ({len(speech_items)}). Visual tags might be misaligned.")
                limit = min(len(segments), len(speech_items))
            else:
                limit = len(segments)
            
            for i in range(limit):
                seg = segments[i]
                script = speech_items[i]
                
                # Check consistency
                # if seg["text"] != script["text"]: ... (Optional check)

                # Copy visual tag if present
                if "visual" in script:
                    seg["visual"] = script["visual"]
            
            return (json.dumps(segments, ensure_ascii=False, indent=2),)

        except Exception as e:
            print(f"[AIIA Merge] Error: {e}")
            return (segments_info,) # Fallback to original


class AIIA_Dialogue_TTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dialogue_json": ("STRING", {"forceInput": True}),
                "tts_engine": (["CosyVoice", "VibeVoice", "Qwen3-TTS"], {"default": "CosyVoice"}),
                "pause_duration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "speed_global": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "batch_mode": (["Natural (Hybrid)", "Strict (Per-Speaker)", "Whole (Single Batch)"], {"default": "Natural (Hybrid)"}),
                "qwen_model": ("QWEN_MODEL",), # Primary Qwen model
                
                # VibeVoice Specific Params
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "cosyvoice_model": ("COSYVOICE_MODEL",),
                "vibevoice_model": ("VIBEVOICE_MODEL",),
                "qwen_base_model": ("QWEN_MODEL",),      # Optional specialized Base
                "qwen_custom_model": ("QWEN_MODEL",),    # Optional specialized CustomVoice
                "qwen_design_model": ("QWEN_MODEL",),    # Optional specialized VoiceDesign
                
                # Speaker A
                "speaker_A_ref": ("AUDIO",),
                "speaker_A_id": ("STRING", {"default": "", "placeholder": "CosyVoice 内部音色ID (可选)"}),
                "speaker_A_emotion": (AIIA_EMOTION_LIST, {"default": "None"}),
                
                # Speaker B
                "speaker_B_ref": ("AUDIO",),
                "speaker_B_id": ("STRING", {"default": "", "placeholder": "CosyVoice 内部音色ID (可选)"}),
                "speaker_B_emotion": (AIIA_EMOTION_LIST, {"default": "None"}),

                # Speaker C
                "speaker_C_ref": ("AUDIO",),
                "speaker_C_id": ("STRING", {"default": "", "placeholder": "CosyVoice 内部音色ID (可选)"}),
                "speaker_C_emotion": (AIIA_EMOTION_LIST, {"default": "None"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("full_audio", "segments_info")
    FUNCTION = "process_dialogue"
    CATEGORY = "AIIA/Podcast"

    def _load_fallback_audio(self, target="Male"):
        import os
        import torchaudio
        
        # 定位 assets 目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(current_dir, "assets")
        
        # 映射
        filename_map = {
            "Male_HQ": "seed_male_hq.wav",
            "Female_HQ": "seed_female_hq.wav",
            "Male": "seed_male.wav",
            "Female": "seed_female.wav"
        }
        
        filename = filename_map.get(target, "seed_female_hq.wav")
        path = os.path.join(assets_dir, filename)
            
        if not os.path.exists(path):
            print(f"[AIIA Warning] Fallback seed not found at {path}")
            return None
            
        try:
            waveform, sample_rate = torchaudio.load(path)
            # 统一转 mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # AIIA Fix: Attenuate volume to prevent clipping/breaking in generation
            waveform = waveform * 0.8
            
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[AIIA Error] Failed to load fallback audio: {e}")
            return None

    def process_dialogue(self, dialogue_json, tts_engine, pause_duration, speed_global, 
                         qwen_model, cosyvoice_model=None, vibevoice_model=None, 
                         qwen_base_model=None, qwen_custom_model=None, qwen_design_model=None,
                         cfg_scale=1.5, temperature=0.8, top_k=20, top_p=0.95, **kwargs):
        import json
        import torch
        import os
        import torchaudio
        
        # 0. 验证输入
        if tts_engine == "CosyVoice" and cosyvoice_model is None:
            raise ValueError("选择 CosyVoice 引擎时，必须连接 'cosyvoice_model'！")
        if tts_engine == "VibeVoice" and vibevoice_model is None:
            raise ValueError("选择 VibeVoice 引擎时，必须连接 'vibevoice_model'！")
        if tts_engine == "Qwen3-TTS" and qwen_model is None:
            raise ValueError("选择 Qwen3-TTS 引擎时，必须连接 'qwen_model'！")

        dialogue = json.loads(dialogue_json)
        full_waveform = []
        sample_rate = 22050 
        
        from .aiia_cosyvoice_nodes import AIIA_CosyVoice_TTS
        from .aiia_vibevoice_nodes import AIIA_VibeVoice_TTS
        from .aiia_qwen_nodes import AIIA_Qwen_TTS
        
        cosy_gen = AIIA_CosyVoice_TTS()
        vibe_gen = AIIA_VibeVoice_TTS()
        qwen_gen = AIIA_Qwen_TTS()

        print(f"[AIIA Podcast] 开始处理对话，共 {len(dialogue)} 个片段。引擎: {tts_engine}")

        # --- Helper: 解析真实 Speaker Key ---
        def get_speaker_key(speaker_name):
            spk_key = speaker_name.strip()
            if spk_key.upper() in ["A", "B", "C"]: return spk_key.upper()
            clean = re.sub(r'speaker[ _-]*', '', spk_key, flags=re.IGNORECASE).strip()
            if clean and clean[0].upper() in ["A", "B", "C"]: return clean[0].upper()
            if spk_key[-1].upper() in ["A", "B", "C"]: return spk_key[-1].upper()
            return spk_key[0].upper()

        # --- Helper: 获取参考音频 ---
        def get_ref_audio(spk_key):
            ref = kwargs.get(f"speaker_{spk_key}_ref")
            if ref is not None: return ref
            
            # Fallback logic
            fallback_target = "Male"
            if "A" in spk_key: fallback_target = "Male_HQ"
            elif "B" in spk_key: fallback_target = "Female_HQ"
            elif "C" in spk_key: fallback_target = "Male"
            else: fallback_target = "Female"
            
            print(f"  [Auto-Fallback] Speaker {spk_key} using {fallback_target}")
            return self._load_fallback_audio(fallback_target)

        # --- Segment Tracker ---
        segments_info = [] # List of {"start": float, "end": float, "text": str, "speaker": str}
        time_ptr = [0.0] # Mutable time pointer

        # --- Batch Flusher ---
        def flush_batch(batch_items, current_full_wav, sr_ptr):
            if not batch_items: return
            
            if tts_engine == "VibeVoice":
                # VibeVoice Hybrid Batching
                unique_speakers = {} 
                ref_audio_list = []
                final_text_lines = []
                
                # Pre-calculate total text length for interpolation
                total_char_len = 0
                item_lengths = []

                for item in batch_items:
                    spk_name = item["speaker"]
                    spk_key = get_speaker_key(spk_name)
                    
                    if spk_key not in unique_speakers:
                        unique_speakers[spk_key] = len(unique_speakers)
                        ref_audio_list.append(get_ref_audio(spk_key))
                    
                    internal_id = unique_speakers[spk_key]
                    text = item["text"]
                    emotion = item.get("emotion", "")
                    
                    # Merge preset emotion if any
                    preset_emo_full = kwargs.get(f"speaker_{spk_key}_emotion", "None")
                    if preset_emo_full and preset_emo_full != "None":
                        emo_label = preset_emo_full.split(" (")[0] if " (" in preset_emo_full else preset_emo_full
                        if emotion: text = f"[{emotion}, {emo_label}] {text}"
                        else: text = f"[{emo_label}] {text}"
                    elif emotion:
                        text = f"[{emotion}] {text}"

                    # Clean text for length calc (approx)
                    clean_text = re.sub(r'\[.*?\]', '', text).strip()
                    char_len = len(clean_text) if clean_text else 1
                    total_char_len += char_len
                    item_lengths.append(char_len)

                    final_text_lines.append(f"Speaker {internal_id}: {text}")
                
                full_text = "\n".join(final_text_lines)
                print(f"  [Batch Process] Processing {len(batch_items)} segments using {len(unique_speakers)} speakers.")
                
                try:
                    res = vibe_gen.generate(
                        vibevoice_model=vibevoice_model,
                        text=full_text,
                        reference_audio=ref_audio_list,
                        cfg_scale=cfg_scale,
                        ddpm_steps=20,
                        speed=speed_global,
                        normalize_text=True,
                        do_sample="auto",
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    
                    if res and res[0]:
                        wav = res[0]["waveform"]
                        sr = res[0]["sample_rate"]
                        
                        if sr_ptr[0] != sr:
                            if current_full_wav:
                                wav = torchaudio.transforms.Resample(sr, sr_ptr[0])(wav)
                            else:
                                sr_ptr[0] = sr
                        
                        if wav.ndim == 3: wav = wav.squeeze(0)
                        if wav.ndim == 1: wav = wav.unsqueeze(0)
                        current_full_wav.append(wav)

                        # --- Timestamp Interpolation ---
                        total_duration = wav.shape[-1] / sr
                        current_batch_start = time_ptr[0]
                        
                        accumulated_time = 0.0
                        for idx, item in enumerate(batch_items):
                            item_len = item_lengths[idx]
                            fraction = item_len / max(total_char_len, 1)
                            est_duration = fraction * total_duration
                            
                            seg_start = current_batch_start + accumulated_time
                            seg_end = seg_start + est_duration
                            
                            segments_info.append({
                                "start": round(seg_start, 3),
                                "end": round(seg_end, 3),
                                "text": item["text"],
                                "speaker": item["speaker"],
                                "visual": item.get("visual")
                            })
                            accumulated_time += est_duration
                        
                        time_ptr[0] += total_duration

                except Exception as e:
                    print(f"[Error] Batch generation failed: {e}")
                    current_full_wav.append(torch.zeros(1, 24000))
                    # Fallback timestamps
                    start_t = time_ptr[0]
                    end_t = start_t + 1.0
                    for item in batch_items:
                        segments_info.append({
                            "start": start_t, "end": end_t, "text": item["text"] + " (Gen Failed)", "speaker": item["speaker"]
                        })
                    time_ptr[0] += 1.0
                    
            elif tts_engine == "Qwen3-TTS":
                # Qwen3-TTS (Iterative)
                for i, item in enumerate(batch_items):
                    spk_name = item["speaker"]
                    spk_key = get_speaker_key(spk_name)
                    text = item["text"]
                    emotion = item.get("emotion", "None")
                    
                    # Mapping logic for Qwen
                    spk_id = kwargs.get(f"speaker_{spk_key}_id", "Vivian") # Default to Vivian if empty
                    if not spk_id.strip(): spk_id = "Vivian"
                    
                    ref_audio = get_ref_audio(spk_key)
                    
                    # Merge preset emotion
                    preset_emo_full = kwargs.get(f"speaker_{spk_key}_emotion", "None")
                    merged_emo = emotion if emotion and emotion != "None" else ""
                    if preset_emo_full and preset_emo_full != "None":
                        emo_label = preset_emo_full.split(" (")[0] if " (" in preset_emo_full else preset_emo_full
                        if merged_emo: merged_emo = f"{merged_emo}，{emo_label}"
                        else: merged_emo = emo_label
                    
                    instruct = f"{merged_emo}。" if merged_emo else ""
                    
                    # --- Qwen Smart Routing ---
                    # Logic: 
                    # 1. If ref_audio exists -> Prefer qwen_base_model
                    # 2. If instruct exists but no ref_audio -> Might be design intent, prefer qwen_design_model
                    # 3. If spk_id exists -> Prefer qwen_custom_model
                    # 4. Fallback to primary qwen_model
                    
                    target_model = qwen_model
                    if ref_audio is not None and qwen_base_model is not None:
                        target_model = qwen_base_model
                    elif instruct and qwen_design_model is not None and ref_audio is None:
                        target_model = qwen_design_model
                    elif qwen_custom_model is not None:
                        target_model = qwen_custom_model
                    
                    print(f"  [Qwen Routing] {spk_name} -> Using {target_model['type']} model ({target_model['name']})")
                    
                    try:
                        # Call Qwen TTS with routed model
                        res = qwen_gen.generate(
                            qwen_model=target_model,
                            text=text,
                            language="Auto",
                            speaker=spk_id,
                            instruct=instruct,
                            reference_audio=ref_audio,
                            seed=seed if seed >= 0 else -1,
                            speed=speed_global,
                            cfg_scale=cfg_scale,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p
                        )
                        
                        generated = res[0]
                        wav = generated["waveform"]
                        sr = generated["sample_rate"]
                        
                        if sr_ptr[0] != sr:
                            if current_full_wav:
                                wav = torchaudio.transforms.Resample(sr, sr_ptr[0])(wav)
                            else:
                                sr_ptr[0] = sr
                        
                        if wav.ndim == 3: wav = wav.squeeze(0)
                        if wav.ndim == 1: wav = wav.unsqueeze(0)
                        current_full_wav.append(wav)

                        # --- Timestamp Tracking ---
                        seg_duration = wav.shape[-1] / sr
                        seg_start = time_ptr[0]
                        seg_end = seg_start + seg_duration
                        
                        segments_info.append({
                            "start": round(seg_start, 3),
                            "end": round(seg_end, 3),
                            "text": text,
                            "speaker": spk_name,
                            "visual": item.get("visual")
                        })
                        time_ptr[0] += seg_duration
                        
                        # Add a small gap between segments
                        gap = 0.2
                        gap_samples = int(gap * sr_ptr[0])
                        current_full_wav.append(torch.zeros(1, gap_samples))
                        time_ptr[0] += gap
                            
                    except Exception as e:
                        print(f"[Error] Qwen item generation failed: {e}")
                        current_full_wav.append(torch.zeros(1, 24000))
                        time_ptr[0] += 1.0

            else:
                # CosyVoice (Iterative)
                for i, item in enumerate(batch_items):
                    spk_name = item["speaker"]
                    spk_key = get_speaker_key(spk_name)
                    text = item["text"]
                    emotion = item.get("emotion") # In CosyVoice, we put it in [] in text
                    
                    # Merge preset emotion
                    preset_emo_full = kwargs.get(f"speaker_{spk_key}_emotion", "None")
                    if preset_emo_full and preset_emo_full != "None":
                        emo_label = preset_emo_full.split(" (")[0] if " (" in preset_emo_full else preset_emo_full
                        if emotion: text = f"[{emotion}, {emo_label}] {text}"
                        else: text = f"[{emo_label}] {text}"
                    elif emotion:
                        text = f"[{emotion}] {text}"

                    ref_audio = get_ref_audio(spk_key)
                    
                    # CosyVoice uses instruct_text for emotion, so we use the merged emotion for it
                    # If emotion is None (from script) and preset_emo_full is "None", then instruct will be empty.
                    # Otherwise, it will use the merged emotion.
                    merged_emo_for_instruct = ""
                    if preset_emo_full and preset_emo_full != "None":
                        merged_emo_for_instruct = preset_emo_full.split(" (")[0] if " (" in preset_emo_full else preset_emo_full
                    elif item.get("emotion") and item.get("emotion") != "None": # Use script emotion if no preset
                        merged_emo_for_instruct = item.get("emotion")

                    instruct = f"{merged_emo_for_instruct}." if merged_emo_for_instruct else ""
                    
                    print(f"  [Processing] {spk_name}: {text[:15]}...")
                    try:
                        res = cosy_gen.generate(
                            model=cosyvoice_model,
                            tts_text=text,
                            instruct_text=instruct,
                            spk_id=spk_id,
                            speed=speed_global,
                            seed=42+i,
                            dialect="None (Auto)",
                            emotion="None (Neutral)",
                            reference_audio=ref_audio
                        )
                        generated = res[0]
                        wav = generated["waveform"]
                        sr = generated["sample_rate"]
                        
                        if sr_ptr[0] != sr:
                            if current_full_wav:
                                wav = torchaudio.transforms.Resample(sr, sr_ptr[0])(wav)
                            else:
                                sr_ptr[0] = sr
                        
                        if wav.ndim == 3: wav = wav.squeeze(0)
                        if wav.ndim == 1: wav = wav.unsqueeze(0)
                        current_full_wav.append(wav)

                        # --- Timestamp Tracking ---
                        seg_duration = wav.shape[-1] / sr
                        seg_start = time_ptr[0]
                        seg_end = seg_start + seg_duration
                        
                        segments_info.append({
                            "start": round(seg_start, 3),
                            "end": round(seg_end, 3),
                            "text": text,
                            "speaker": spk_name,
                            "visual": item.get("visual")
                        })
                        time_ptr[0] += seg_duration
                        
                        if tts_engine == "CosyVoice" and i < len(batch_items) - 1:
                            gap = 0.2
                            gap_samples = int(gap * sr_ptr[0])
                            current_full_wav.append(torch.zeros(1, gap_samples))
                            time_ptr[0] += gap
                            
                    except Exception as e:
                        print(f"[Error] Item generation failed: {e}")
                        current_full_wav.append(torch.zeros(1, 16000))
                        time_ptr[0] += 1.0

                # --- Main Loop ---
        batch_buffer = []
        sr_wrapper = [22050 if tts_engine == "CosyVoice" else 24000] # Mutable ref
        
        for i, item in enumerate(dialogue):
            # Check mode
            batch_mode = kwargs.get("batch_mode", "Natural (Hybrid)")

            if item["type"] == "pause":
                # In Whole Batch mode, we IGNORE pauses to maintain single-batch integrity
                if batch_mode == "Whole (Single Batch)":
                    print("  [Whole Batch] Ignoring explicit pause to maintain flow.")
                    continue

                # Flush current speech batch
                flush_batch(batch_buffer, full_waveform, sr_wrapper)
                batch_buffer = []
                
                # Append Pause
                duration = item.get("duration", pause_duration)
                print(f"  [Pause] {duration}s")
                silence_samples = int(duration * sr_wrapper[0])
                if silence_samples > 0:
                    full_waveform.append(torch.zeros(1, silence_samples))
                    time_ptr[0] += duration # Track pause time

            
            elif item["type"] == "speech":
                # Check for strict mode
                batch_mode = kwargs.get("batch_mode", "Natural (Hybrid)")
                
                # If strict mode, flush on speaker change
                if batch_mode == "Strict (Per-Speaker)" and batch_buffer:
                    last_spk = batch_buffer[-1]["speaker"]
                    curr_spk = item["speaker"]
                    # Simple name check or mapped key check? 
                    # Let's use name check for simplicity as key derivation is same
                    if last_spk != curr_spk:
                        flush_batch(batch_buffer, full_waveform, sr_wrapper)
                        batch_buffer = []

                batch_buffer.append(item)
        
        # Final Flush
        flush_batch(batch_buffer, full_waveform, sr_wrapper)

        # Merge
        sample_rate = sr_wrapper[0]
        if not full_waveform:
            final_tensor = torch.zeros(1, 16000)
            sample_rate = 16000
        else:
            final_tensor = torch.cat(full_waveform, dim=1)
        
        # Output Segments Info JSON
        segments_json = json.dumps(segments_info, ensure_ascii=False, indent=2)

        return ({"waveform": final_tensor.unsqueeze(0), "sample_rate": sample_rate}, segments_json)


# 节点映射导出
NODE_CLASS_MAPPINGS = {
    "AIIA_Podcast_Script_Parser": AIIA_Podcast_Script_Parser,
    "AIIA_Dialogue_TTS": AIIA_Dialogue_TTS,
    "AIIA_Segment_Merge": AIIA_Segment_Merge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Podcast_Script_Parser": "📜 AIIA Podcast Script Parser",
    "AIIA_Dialogue_TTS": "🎧 AIIA Dialogue TTS (Multi-Role)",
    "AIIA_Segment_Merge": "🔗 AIIA Segment Merge (Visual)"
}
