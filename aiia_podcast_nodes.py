
import json
import re

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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("dialogue_json", "speaker_list")
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
                
                dialogue.append({
                    "type": "speech",
                    "speaker": speaker_id,
                    "text": content,
                    "emotion": emotion
                })
            else:
                # 可能是延续上一句的内容，或者无法解析
                # 简单起见，如果延续上一句，我们追加到上一句
                # 但更安全的做法是忽略或作为新的一句（但这需要speaker）
                # 这里假设格式必须严格
                pass

        speaker_list = sorted(list(speakers))
        
        return (json.dumps(dialogue, ensure_ascii=False, indent=2), ",".join(speaker_list))

class AIIA_Dialogue_TTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dialogue_json": ("STRING", {"forceInput": True}),
                "tts_engine": (["CosyVoice", "VibeVoice"], {"default": "CosyVoice"}),
                "pause_duration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "speed_global": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "batch_mode": (["Natural (Hybrid)", "Strict (Per-Speaker)", "Whole (Single Batch)"], {"default": "Natural (Hybrid)"}),
                
                # VibeVoice Specific Params
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "cosyvoice_model": ("COSYVOICE_MODEL",),
                "vibevoice_model": ("VIBEVOICE_MODEL",),
                
                # Speaker A
                "speaker_A_ref": ("AUDIO",),
                "speaker_A_id": ("STRING", {"default": "", "placeholder": "CosyVoice 内部音色ID (可选)"}),
                
                # Speaker B
                "speaker_B_ref": ("AUDIO",),
                "speaker_B_id": ("STRING", {"default": "", "placeholder": "CosyVoice 内部音色ID (可选)"}),

                # Speaker C
                "speaker_C_ref": ("AUDIO",),
                "speaker_C_id": ("STRING", {"default": "", "placeholder": "CosyVoice 内部音色ID (可选)"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("full_audio",)
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
                         cosyvoice_model=None, vibevoice_model=None, 
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

        dialogue = json.loads(dialogue_json)
        full_waveform = []
        sample_rate = 22050 
        
        from .aiia_cosyvoice_nodes import AIIA_CosyVoice_TTS
        from .aiia_vibevoice_nodes import AIIA_VibeVoice_TTS
        
        cosy_gen = AIIA_CosyVoice_TTS()
        vibe_gen = AIIA_VibeVoice_TTS()

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

        # --- Batch Flusher ---
        def flush_batch(batch_items, current_full_wav, sr_ptr):
            if not batch_items: return
            
            if tts_engine == "VibeVoice":
                # VibeVoice Hybrid Batching
                # 1. 收集唯一角色并分配 ID
                unique_speakers = {} # key -> internal_id (0, 1...)
                ref_audio_list = []
                final_text_lines = []
                
                for item in batch_items:
                    spk_name = item["speaker"]
                    spk_key = get_speaker_key(spk_name)
                    
                    if spk_key not in unique_speakers:
                        unique_speakers[spk_key] = len(unique_speakers)
                        # Load Ref Audio
                        ref_audio_list.append(get_ref_audio(spk_key))
                    
                    internal_id = unique_speakers[spk_key]
                    text = item["text"]
                    # AIIA Fix: Ensure clean separation for VibeVoice prompt
                    final_text_lines.append(f"Speaker {internal_id}: {text}")
                
                full_text = "\n".join(final_text_lines)
                print(f"  [Batch Process] Processing {len(batch_items)} segments using {len(unique_speakers)} speakers.")
                
                try:
                    res = vibe_gen.generate(
                        vibevoice_model=vibevoice_model,
                        text=full_text,
                        reference_audio=ref_audio_list, # Passing List!
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
                        
                        # Resample check
                        if sr_ptr[0] != sr:
                            if current_full_wav: # define master SR from first clip or previous
                                wav = torchaudio.transforms.Resample(sr, sr_ptr[0])(wav)
                            else:
                                sr_ptr[0] = sr # First clip sets SR
                        
                        if wav.ndim == 3: wav = wav.squeeze(0)
                        if wav.ndim == 1: wav = wav.unsqueeze(0)
                        current_full_wav.append(wav)

                except Exception as e:
                    print(f"[Error] Batch generation failed: {e}")
                    # Fallback: append silence
                    current_full_wav.append(torch.zeros(1, 24000))
                    
            else:
                # CosyVoice (Iterative)
                for i, item in enumerate(batch_items):
                    spk_name = item["speaker"]
                    spk_key = get_speaker_key(spk_name)
                    text = item["text"]
                    emotion = item.get("emotion", "None")
                    
                    spk_id = kwargs.get(f"speaker_{spk_key}_id", "")
                    ref_audio = get_ref_audio(spk_key) # Cosy also needs fallback ref sometimes
                    
                    instruct = f"{emotion}." if emotion and emotion != "None" else ""
                    
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
                        
                        # CosyVoice separate items might need mini pause? 
                        # Original logic: pause_duration at end of item IF next is not pause.
                        # But here we are flushing a batch that was delimited by pauses.
                        # So inside a batch (continuous dialogue), user might expect flow.
                        # We append a tiny gap (e.g. 0.1s) for naturalness?
                        # Or stick to original logic: "pause_duration" was only pushed when "pause" item found (outside batch).
                        # Actually previous logic appended pause_duration at END of item unless next was Pause.
                        # Here, we treat the whole batch as one continuous flow.
                        # VibeVoice handles flow naturally. CosyVoice might need help.
                        if tts_engine == "CosyVoice" and i < len(batch_items) - 1:
                            # Small gap between turns in same batch
                            current_full_wav.append(torch.zeros(1, int(0.2 * sr_ptr[0])))
                            
                    except Exception as e:
                        print(f"[Error] Item generation failed: {e}")
                        current_full_wav.append(torch.zeros(1, 16000))


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
            max_chn = max([w.shape[0] for w in full_waveform])
            unified = []
            for w in full_waveform:
                if w.shape[0] < max_chn:
                    w = w.repeat(max_chn, 1)
                unified.append(w)
            final_tensor = torch.cat(unified, dim=-1)
            final_tensor = final_tensor.unsqueeze(0)
        
        return ({"waveform": final_tensor, "sample_rate": sample_rate},)


# 节点映射导出
NODE_CLASS_MAPPINGS = {
    "AIIA_Podcast_Script_Parser": AIIA_Podcast_Script_Parser,
    "AIIA_Dialogue_TTS": AIIA_Dialogue_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Podcast_Script_Parser": "📜 AIIA Podcast Script Parser",
    "AIIA_Dialogue_TTS": "🎧 AIIA Dialogue TTS (Multi-Role)"
}
