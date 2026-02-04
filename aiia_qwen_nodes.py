import os
import sys
import torch
import torchaudio
import numpy as np
import folder_paths
import subprocess

# --- Robust Package Installation ---
# --- Official Speaker List ---
QWEN_SPEAKER_LIST = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
QWEN_PRESET_NOTE = "Presets (9 premium timbres): Vivian/Serena/Uncle_Fu (CN), Dylan/Eric/Ryan/Aiden (EN), Ono_Anna (JP), Sohee (KR)"
QWEN_EMOTION_LIST = [
    "None", "开心 (Happy)", "悲伤 (Sad)", "生气 (Angry)", "兴奋 (Excited)", 
    "温柔 (Gentle)", "严肃 (Serious)", "恐惧 (Fearful)", "惊讶 (Surprised)", 
    "低语 (Whispering)", "呐喊 (Shouting)", "羞涩 (Shy)", "诱惑 (Seductive)", 
    "哭腔 (Crying)", "笑声 (Laughter)", "尴尬 (Embarrassed)", "失望 (Disappointed)", 
    "自豪 (Proud)", "疑惑 (Doubtful)", "焦虑 (Anxious)", "平静 (Calm)"
]

QWEN_EXPRESSION_LIST = [
    "None", 
    "带点羞涩的 (With a hint of shyness)", 
    "语气充满诱惑力 (Seductive tone)", 
    "语气带着哭腔 (Crying tone)",
    "稍微带一点点笑意 (With a slight smile)",
    "语气显得非常疲惫 (Sounding very tired)",
    "语速稍快，显得有些急促 (Hurried tone)",
    "充满自信且响亮的 (Confident and loud)",
    "稍微有点犹豫 and 不确定 (Hesitant and uncertain)",
    "语气极其冷淡 (Extremely cold tone)",
    "温柔且轻声细语的 (Gentle and whispering)"
]

QWEN_DIALECT_LIST = [
    "None", "普通话 (Mandarin)", "粤语 (Cantonese)", "上海话 (Shanghainese)", 
    "四川话 (Sichuanese)", "东北话 (Northeastern)", "闽南话 (Hokkien)", 
    "客家话 (Hakka)", "天津话 (Tianjinese)", "山东话 (Shandongnese)"
]

def _install_qwen_tts_if_needed():
    try:
        from qwen_tts import Qwen3TTSModel
        # Check if transformers is at least 4.48.0 and not a broken 5.0.0
        import transformers
        from packaging import version
        v = version.parse(transformers.__version__)
        if v.major >= 5 or v < version.parse("4.48.0"):
             print(f"[AIIA] Incompatible transformers version {transformers.__version__} found, fixing...")
             raise ImportError("Need stable transformers")
        return
    except (ImportError, ModuleNotFoundError, TypeError):
        print("[AIIA] Installing/Fixing Qwen3-TTS dependencies (qwen-tts, transformers)...")
        try:
            import subprocess
            import sys
            import site
            
            # Enforce stable transformers version to avoid 'MODELS_TO_PIPELINE' import errors
            # vllm needs >= 4.56.0, 4.57.x seems to have import issues with qwen-tts
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.56.2", "qwen-tts"])
            
            # Surgical patch for qwen-tts @check_model_inputs() bug
            # Find the qwen_tts installation path
            import importlib.util
            spec = importlib.util.find_spec("qwen_tts")
            if spec and spec.origin:
                qwen_path = os.path.dirname(os.path.dirname(spec.origin))
                target_file = os.path.join(qwen_path, "qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py")
                if os.path.exists(target_file):
                    print(f"[AIIA] Patching qwen-tts decorator bug at {target_file}")
                    with open(target_file, "r") as f:
                        content = f.read()
                    if "@check_model_inputs()" in content:
                        new_content = content.replace("@check_model_inputs()", "@check_model_inputs")
                        with open(target_file, "w") as f:
                            f.write(new_content)
                        print("[AIIA] Patch applied successfully.")
            
            print("[AIIA] Dependencies installed and patched successfully.")
        except Exception as e:
            print(f"[AIIA] Failed to install/patch qwen-tts dependencies: {e}")
            # Don't raise here, we want the subsequent load attempt to show the real error if any

class AIIA_Qwen_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ([
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
                ], {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}),
                "device": (["cuda", "cpu", "auto", "mps"], {"default": "auto"}),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "local_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load_model"
    CATEGORY = "AIIA/Loaders"

    def load_model(self, model_name, device, dtype, local_path=""):
        _install_qwen_tts_if_needed()
        from qwen_tts import Qwen3TTSModel

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available(): device = "cuda"
            elif torch.backends.mps.is_available(): device = "mps"
            else: device = "cpu"
        
        # Resolve dtype
        torch_dtype = torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "fp16" else torch.float32)
        
        # Resolve path
        path = local_path
        if not path or not os.path.exists(path):
            # Check ComfyUI models directory
            # model_name is like "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            potential_path = os.path.join(folder_paths.models_dir, "qwen_tts", model_name)
            if os.path.exists(potential_path):
                path = potential_path
                print(f"[AIIA] Found local model at: {path}")
            else:
                path = model_name
                print(f"[AIIA] Local model not found, will attempt to load/download via HuggingFace HUB: {path}")
        
        print(f"[AIIA] Loading Qwen3-TTS: {path} on {device} with {dtype}")
        
        # Flash Attention check
        attn_impl = "flash_attention_2" if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else "sdpa"
        
        model = Qwen3TTSModel.from_pretrained(
            path,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl
        )
        
        model_type = "Base"
        if "CustomVoice" in path: model_type = "CustomVoice"
        elif "VoiceDesign" in path: model_type = "VoiceDesign"
        
        return ({"model": model, "type": model_type, "name": path, "device": device},)

class AIIA_Qwen_TTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，这是 Qwen3-TTS 的测试。"}),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {"default": "Chinese"}),
            },
            "optional": {
                "speaker": (QWEN_SPEAKER_LIST, {"default": "Vivian"}),
                "instruct": ("STRING", {"multiline": True, "default": ""}),
                "reference_audio": ("AUDIO",),
                "reference_text": ("STRING", {"multiline": True, "default": ""}),
                "zero_shot_mode": ("BOOLEAN", {"default": False}),
                "emotion": (QWEN_EMOTION_LIST, {"default": "None"}),
                "dialect": (QWEN_DIALECT_LIST, {"default": "None"}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_batch_char": ("INT", {"default": 1000, "min": 100, "max": 32768}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/Synthesis"

    def generate(self, qwen_model, text, language, speaker="Vivian", instruct="", reference_audio=None, reference_text="", zero_shot_mode=False, emotion="None", dialect="None", seed=42, speed=1.0, cfg_scale=1.5, temperature=0.8, top_k=20, top_p=0.95):
        # 0. Handle Bundle Routing
        active_qwen = qwen_model
        if qwen_model.get("is_bundle"):
            # Auto-route based on generation intent
            if reference_audio is not None or zero_shot_mode:
                active_qwen = qwen_model.get("base") or qwen_model.get("default")
            elif instruct.strip() or dialect != "None" or emotion != "None":
                active_qwen = qwen_model.get("design") or qwen_model.get("custom") or qwen_model.get("default")
            else:
                active_qwen = qwen_model.get("custom") or qwen_model.get("default")
        
        if not active_qwen:
            raise ValueError("[AIIA Qwen] No active model found in bundle or input!")

        model = active_qwen["model"]
        m_type = active_qwen["type"]
        model_path = active_qwen.get("path", "").lower()
        
        # Warning for Base model with instruct
        if "base" in model_path and (instruct.strip() or dialect != "None" or emotion != "None"):
            print(f"\033[33m[AIIA Qwen] Warning: Base models primarily support cloning and may ignore text instructions/dialects.\033[0m")
        
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        lang_param = language if language != "Auto" else "Auto"
        
        # Generation Kwargs
        gen_kwargs = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "cfg_scale": cfg_scale
        }

        # Merge dialect and global emotion into instruct
        final_instruct = instruct
        
        # 1. Add Dialect
        if dialect and dialect != "None":
            dia_label = dialect.split(" (")[0] if " (" in dialect else dialect
            final_instruct = f"用{dia_label}说。" + (f"{final_instruct}" if final_instruct else "")

        # 2. Add Emotion
        if emotion and emotion != "None":
            # Extract basic emotion name from "Name (English)"
            emo_label = emotion.split(" (")[0] if " (" in emotion else emotion
            if not final_instruct:
                final_instruct = f"{emo_label}。"
            else:
                # Append if not already present
                if emo_label not in final_instruct:
                    if final_instruct.endswith("。"):
                        final_instruct = f"{final_instruct}{emo_label}。"
                    else:
                        final_instruct = f"{final_instruct}。{emo_label}。"

        wavs = None
        sr = 24000 # Default if unknown
        
        try:
            if m_type == "CustomVoice":
                print(f"[AIIA] Qwen3-TTS CustomVoice: {speaker} | Instruct: {final_instruct}")
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=lang_param,
                    speaker=speaker,
                    instruct=final_instruct if final_instruct else None,
                    **gen_kwargs
                )
            elif m_type == "VoiceDesign":
                print(f"[AIIA] Qwen3-TTS VoiceDesign: {final_instruct}")
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=lang_param,
                    instruct=final_instruct, # For Design, instruct IS the design
                    **gen_kwargs
                )
            else: # Base / Clone
                if reference_audio is not None:
                    # Convert ComfyUI Audio format to (numpy, sr) tuple
                    device = qwen_model["device"]
                    ref_wav = reference_audio["waveform"]
                    ref_sr = reference_audio["sample_rate"]
                    
                    # Convert to mono if needed
                    if ref_wav.ndim == 3: ref_wav = ref_wav[0]
                    if ref_wav.shape[0] > 1: ref_wav = torch.mean(ref_wav, dim=0, keepdim=True)
                    
                    ref_audio_data = (ref_wav.squeeze().cpu().numpy(), ref_sr)
                    
                    ref_text = reference_text if reference_text and reference_text.strip() != "" else None
                    mode_param = zero_shot_mode
                    
                    # Robustness: Qwen requires ref_text for ICL mode (zero_shot_mode=False)
                    if not ref_text and not mode_param:
                        print(f"[AIIA Warning] No 'reference_text' provided. Automatically switching to 'zero_shot_mode=True' (Zero-Shot) to prevent crash.")
                        mode_param = True
                    
                    print(f"[AIIA] Qwen3-TTS VoiceClone: Using reference. Mode: {'Zero-Shot' if mode_param else 'ICL'}")
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=lang_param,
                        ref_audio=ref_audio_data,
                        ref_text=ref_text,
                        x_vector_only_mode=mode_param,
                        instruct=final_instruct if final_instruct else None,
                        **gen_kwargs
                    )
                else:
                    # Fallback if no reference provided for Base model
                    # Typically Base model MUST have reference.
                    # We might want to provide a default one or error out.
                    raise ValueError("Qwen3-TTS Base model requires 'reference_audio' and 'reference_text' for cloning.")

            # Process output
            if wavs is not None and len(wavs) > 0:
                audio_out = torch.from_numpy(wavs[0]).float()
                if audio_out.ndim == 1: audio_out = audio_out.unsqueeze(0)
                
                # Speed adj (Qwen3-TTS might not have native speed param in generate_* yet, so we use torchaudio if needed)
                if speed != 1.0:
                    # Simple speed change via resampling (pitch change) - matches CosyVoice fallback
                    resampler = torchaudio.transforms.Resample(orig_freq=int(sr*speed), new_freq=sr)
                    audio_out = resampler(audio_out)
                
                return ({"waveform": audio_out.unsqueeze(0), "sample_rate": sr},)
            
        except Exception as e:
            print(f"[AIIA] Qwen3-TTS Generation Error: {e}")
            import traceback
            traceback.print_exc()
            raise e

class AIIA_Qwen_Dialogue_TTS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dialogue_json": ("STRING", {"multiline": True}),
                "pause_duration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "speed_global": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "dialect_note": ("STRING", {"default": "💡 提示：方言建议配合 Design 模式使用。", "is_label": True}),
                "base_note": ("STRING", {"default": "⚠️ 注意：Clone 模式下的 Base 模型不支持文字指令控制。", "is_label": True}),
                "qwen_model": ("QWEN_MODEL",),
            },
            "optional": {
                "preset_note": ("STRING", {"default": QWEN_PRESET_NOTE, "is_label": True}),
                # Speaker A
                "speaker_A_mode": (["Clone", "Preset", "Design"], {"default": "Clone"}),
                "speaker_A_id": (QWEN_SPEAKER_LIST, {"default": "Vivian"}),
                "speaker_A_emotion": (QWEN_EMOTION_LIST, {"default": "None"}),
                "speaker_A_expression": (QWEN_EXPRESSION_LIST, {"default": "None"}),
                "speaker_A_dialect": (QWEN_DIALECT_LIST, {"default": "None"}),
                "speaker_A_design": ("STRING", {"multiline": True, "default": ""}),
                "speaker_A_ref": ("AUDIO",),
                "speaker_A_ref_text": ("STRING", {"multiline": True, "default": ""}),
                # Speaker B
                "speaker_B_mode": (["Clone", "Preset", "Design"], {"default": "Clone"}),
                "speaker_B_id": (QWEN_SPEAKER_LIST, {"default": "Vivian"}),
                "speaker_B_emotion": (QWEN_EMOTION_LIST, {"default": "None"}),
                "speaker_B_expression": (QWEN_EXPRESSION_LIST, {"default": "None"}),
                "speaker_B_dialect": (QWEN_DIALECT_LIST, {"default": "None"}),
                "speaker_B_design": ("STRING", {"multiline": True, "default": ""}),
                "speaker_B_ref": ("AUDIO",),
                "speaker_B_ref_text": ("STRING", {"multiline": True, "default": ""}),
                # Speaker C
                "speaker_C_mode": (["Clone", "Preset", "Design"], {"default": "Design"}),
                "speaker_C_id": (QWEN_SPEAKER_LIST, {"default": "Vivian"}),
                "speaker_C_emotion": (QWEN_EMOTION_LIST, {"default": "None"}),
                "speaker_C_expression": (QWEN_EXPRESSION_LIST, {"default": "None"}),
                "speaker_C_dialect": (QWEN_DIALECT_LIST, {"default": "None"}),
                "speaker_C_design": ("STRING", {"multiline": True, "default": ""}),
                "speaker_C_ref": ("AUDIO",),
                "speaker_C_ref_text": ("STRING", {"multiline": True, "default": ""}),
                # New Optional Slots (Appended to prevent shift)
                "qwen_base_model": ("QWEN_MODEL",),
                "qwen_custom_model": ("QWEN_MODEL",),
                "qwen_design_model": ("QWEN_MODEL",),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "zero_shot_mode": ("BOOLEAN", {"default": False}),
                "max_batch_char": ("INT", {"default": 1000, "min": 100, "max": 32768}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_dialogue"
    CATEGORY = "AIIA/Qwen"

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
            
            # AIIA Fix: Attenuate volume to prevent clipping
            waveform = waveform * 0.8
            
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[AIIA Error] Failed to load fallback audio: {e}")
            return None

    def process_dialogue(self, dialogue_json, pause_duration, speed_global, seed=42, cfg_scale=1.5, temperature=0.8, top_k=20, top_p=0.95, zero_shot_mode=False, max_batch_char=1000, **kwargs):
        # Robustness: ensure max_batch_char is correctly picked up
        max_batch_char = kwargs.get("max_batch_char", max_batch_char)
        import json
        import torch
        import torchaudio
        import re
        
        dialogue = json.loads(dialogue_json)
        full_waveform = []
        sample_rate = 24000 
        
        qwen_gen = AIIA_Qwen_TTS()

        print(f"[AIIA Qwen Podcast] Processing {len(dialogue)} segments.")

        def get_speaker_key(speaker_name):
            spk_key = speaker_name.strip()
            if spk_key.upper() in ["A", "B", "C"]: return spk_key.upper()
            clean = re.sub(r'speaker[ _-]*', '', spk_key, flags=re.IGNORECASE).strip()
            if clean and clean[0].upper() in ["A", "B", "C"]: return clean[0].upper()
            return spk_key[0].upper()

        def get_ref_audio_with_fallback(spk_key):
            ref = kwargs.get(f"speaker_{spk_key}_ref")
            if ref is not None: return ref
            
            # Fallback logic parity with general dialogue node
            fallback_target = "Male"
            if "A" in spk_key: fallback_target = "Male_HQ"
            elif "B" in spk_key: fallback_target = "Female_HQ"
            elif "C" in spk_key: fallback_target = "Male"
            else: fallback_target = "Female"
            
            print(f"  [Qwen Auto-Fallback] Speaker {spk_key} using {fallback_target}")
            return self._load_fallback_audio(fallback_target)

        segments_info = []
        time_ptr = 0.0

        print(f"[AIIA Qwen Podcast] Starting Batch Processing (Limit: {max_batch_char} chars).")

        # --- Batching Logic ---
        # A batch is a list of speech segments that share the same:
        # 1. Qwen Model
        # 2. Generation Mode (Clone, Preset, Design)
        # 3. Parameters (ref_audio + ref_text for Clone, design text for Design)

        def get_segment_params(item):
            if "type" in item and item["type"] == "pause":
                return None
            
            spk_name = item.get("speaker", "Unknown")
            spk_key = get_speaker_key(spk_name)
            text = item.get("text", "")
            emotion = item.get("emotion", "")
            
            mode = kwargs.get(f"speaker_{spk_key}_mode", "Clone")
            spk_id = kwargs.get(f"speaker_{spk_key}_id", "Vivian")
            spk_emotion_preset = kwargs.get(f"speaker_{spk_key}_emotion", "None")
            spk_expression_preset = kwargs.get(f"speaker_{spk_key}_expression", "None")
            spk_dialect_preset = kwargs.get(f"speaker_{spk_key}_dialect", "None")
            design = kwargs.get(f"speaker_{spk_key}_design", "")
            # 0. Specialized Qwen Routing from Bundle or slots
            def get_model_from_bundle(mode, ref=None):
                # 1. Check primary qwen_model (might be a bundle)
                main_q = kwargs.get("qwen_model")
                if main_q and main_q.get("is_bundle"):
                    if mode == "Clone" or ref is not None:
                        return main_q.get("base") or main_q.get("default")
                    elif mode == "Design":
                        return main_q.get("design") or main_q.get("default")
                    else: # Preset
                        return main_q.get("custom") or main_q.get("default")
                
                # 2. Check direct slots (compatibility)
                base_q = kwargs.get("qwen_base_model")
                custom_q = kwargs.get("qwen_custom_model")
                design_q = kwargs.get("qwen_design_model")
                
                if mode == "Clone" or ref is not None:
                    return base_q or custom_q or main_q
                elif mode == "Design":
                    return design_q or custom_q or main_q
                else: # Preset
                    return custom_q or main_q or base_q or design_q

            # Extract Speaker Params
            def get_speaker_params(prefix):
                mode = kwargs.get(f"{prefix}_mode", "Preset")
                ref = kwargs.get(f"{prefix}_ref")
                tm = get_model_from_bundle(mode, ref)
                return {
                    "tm": tm,
                    "id": kwargs.get(f"{prefix}_id", "Vivian"),
                    "ref": ref,
                    "exp": kwargs.get(f"{prefix}_expression", ""),
                    "dialect": kwargs.get(f"{prefix}_dialect", "None")
                }
            
            # Check for at least one model
            if all(kwargs.get(k) is None for k in ["qwen_model", "qwen_base_model", "qwen_custom_model", "qwen_design_model"]):
                raise ValueError("AIIA_Qwen_Dialogue_TTS: No Qwen model connected! Connect at least one to qwen_model/base/custom/design.")
            
            ref_audio = get_ref_audio_with_fallback(spk_key) if mode == "Clone" else None
            ref_text = kwargs.get(f"speaker_{spk_key}_ref_text", "")
            
            # Determine the actual Qwen model to use for this segment
            segment_qwen_model = get_model_from_bundle(mode, ref_audio)

            # Unique key for "homogeneity"
            # For Preset, we can merge DIFFERENT speakers by using [Speaker] tags
            # So they only need to share the same qwen_model and mode="Preset"
            if mode == "Preset":
                param_hash = (f"Preset_{id(segment_qwen_model)}", spk_dialect_preset)
            elif mode == "Clone":
                # Must share same ref_audio and ref_text and dialect
                param_hash = (f"Clone_{id(segment_qwen_model)}", id(ref_audio), ref_text, spk_dialect_preset)
            else: # Design
                # Must share the same design text and dialect
                param_hash = (f"Design_{id(segment_qwen_model)}", design, spk_dialect_preset)
            
            return {
                "spk_name": spk_name,
                "spk_id": spk_id,
                "spk_key": spk_key,
                "text": text,
                "emotion": emotion,
                "spk_emotion_preset": spk_emotion_preset,
                "spk_expression_preset": spk_expression_preset,
                "spk_dialect_preset": spk_dialect_preset,
                "mode": mode,
                "qwen_model": segment_qwen_model, # Use the resolved model
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "design": design,
                "param_hash": param_hash
            }

        # Greedy grouping
        batches = []
        current_batch = []
        current_batch_char = 0
        current_hash = None

        for item in dialogue:
            params = get_segment_params(item)
            
            if params is None: # Pause item
                if current_batch:
                    batches.append({"type": "speech", "items": current_batch})
                    current_batch = []
                    current_batch_char = 0
                batches.append({"type": "pause", "duration": item.get("duration", pause_duration)})
                current_hash = None
                continue

            # Check compatibility
            can_merge = (current_hash is not None and params["param_hash"] == current_hash and (current_batch_char + len(params["text"]) < max_batch_char))
            
            if not can_merge:
                if current_batch:
                    batches.append({"type": "speech", "items": current_batch})
                current_batch = [params]
                current_batch_char = len(params["text"])
                current_hash = params["param_hash"]
            else:
                current_batch.append(params)
                current_batch_char += len(params["text"])

        if current_batch:
            batches.append({"type": "speech", "items": current_batch})

        for b_idx, batch in enumerate(batches):
            if batch["type"] == "pause":
                duration = batch["duration"]
                if duration > 0:
                    silence = torch.zeros((1, int(sample_rate * duration)))
                    full_waveform.append(silence)
                    time_ptr += duration
                continue
            
            # Process Speech Batch
            items = batch["items"]
            first = items[0]
            mode = first["mode"]
            qwen_model = first["qwen_model"]
            
            if qwen_model is None: continue

            # Construct batched text and merged instruct
            batched_text = ""
            total_char_count = 0
            item_char_counts = []

            for it in items:
                line_text = it["text"]
                if mode == "Preset":
                    # Use [Speaker] tags for CustomVoice multi-speaker batching
                    # Qwen format: [Speaker] Text [Speaker2] Text2
                    batched_text += f"[{it['spk_id']}] {line_text} "
                else:
                    batched_text += f"{line_text} "
                
                # Length for timestamping
                c_len = len(line_text.strip()) if line_text.strip() else 1
                total_char_count += c_len
                item_char_counts.append(c_len)

            # Instruct logic: merge all dialects, emotions and expressions in the batch
            emotions = []
            dialects = []
            for it in items:
                dia = it["spk_dialect_preset"] if it["spk_dialect_preset"] and it["spk_dialect_preset"] != "None" else ""
                if dia: dialects.append(dia.split(" (")[0] if " (" in dia else dia)
                
                emo = it["emotion"] if it["emotion"] and it["emotion"] != "None" else ""
                pres = it["spk_emotion_preset"] if it["spk_emotion_preset"] and it["spk_emotion_preset"] != "None" else ""
                expr = it["spk_expression_preset"] if it["spk_expression_preset"] and it["spk_expression_preset"] != "None" else ""
                if emo: emotions.append(emo)
                if pres: emotions.append(pres.split(" (")[0] if " (" in pres else pres)
                if expr: emotions.append(expr.split(" (")[0] if " (" in expr else expr)
            
            unique_dias = []
            for d in dialects:
                if d and d not in unique_dias: unique_dias.append(d)
                
            unique_emos = []
            for e in emotions:
                if e and e not in unique_emos: unique_emos.append(e)
            
            dia_instruct = "，".join([f"用{d}说" for d in unique_dias]) + "。" if unique_dias else ""
            emo_instruct = "，".join(unique_emos) + "。" if unique_emos else ""
            target_instruct = dia_instruct + emo_instruct
            if mode == "Design":
                target_instruct = first["design"] if first["design"] else target_instruct

            print(f"  [Qwen Batch] Processing Batch {b_idx}: {len(items)} segments, {total_char_count} chars. Mode: {mode}")
            print(f"  [Qwen Batch] Final Text: {batched_text.strip()}")
            if target_instruct:
                print(f"  [Qwen Batch] Final Instruct: {target_instruct}")

            try:
                res = qwen_gen.generate(
                    qwen_model=qwen_model,
                    text=batched_text.strip(),
                    language="Auto",
                    speaker=first["spk_id"],
                    instruct=target_instruct,
                    reference_audio=first["ref_audio"],
                    reference_text=first["ref_text"],
                    seed=seed if seed >= 0 else -1,
                    speed=speed_global,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    zero_shot_mode=zero_shot_mode
                )
                
                if res and res[0]:
                    wav = res[0]["waveform"]
                    sr = res[0]["sample_rate"]
                    
                    if sample_rate != sr:
                        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
                    
                    wav_data = wav.squeeze(0) # [C, T]
                    full_waveform.append(wav_data)
                    
                    # Timestamp Interpolation
                    batch_duration = wav_data.shape[1] / sample_rate
                    batch_start_time = time_ptr
                    accum_time = 0.0
                    
                    for i_idx, it in enumerate(items):
                        fraction = item_char_counts[i_idx] / max(total_char_count, 1)
                        seg_dur = fraction * batch_duration
                        
                        segments_info.append({
                            "start": round(batch_start_time + accum_time, 3),
                            "end": round(batch_start_time + accum_time + seg_dur, 3),
                            "text": it["text"],
                            "speaker": it["spk_name"]
                        })
                        accum_time += seg_dur
                    
                    time_ptr += batch_duration
                    
                    # Add intra-batch padding if configured (usually Pause items handling it)
                    if pause_duration > 0 and b_idx < len(batches)-1:
                        # Only add if next isn't already a pause
                        if batches[b_idx+1]["type"] != "pause":
                            sil = torch.zeros((wav_data.shape[0], int(sample_rate * pause_duration)))
                            full_waveform.append(sil)
                            time_ptr += pause_duration
                                
            except Exception as e:
                print(f"  [Error] Batch {b_idx} failed: {e}")
                import traceback
                traceback.print_exc()

        if not full_waveform:
            return ({"waveform": torch.zeros((1, 1, 1024)), "sample_rate": sample_rate}, "[]")

        final_wav = torch.cat(full_waveform, dim=1)
        return ({"waveform": final_wav.unsqueeze(0), "sample_rate": sample_rate}, json.dumps(segments_info, ensure_ascii=False))

class AIIA_Qwen_Model_Router:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "qwen_default": ("QWEN_MODEL",),
                "qwen_base": ("QWEN_MODEL",),
                "qwen_custom": ("QWEN_MODEL",),
                "qwen_design": ("QWEN_MODEL",),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_bundle",)
    FUNCTION = "bundle"
    CATEGORY = "AIIA/Loaders"

    def bundle(self, **kwargs):
        qwen_default = kwargs.get("qwen_default")
        qwen_base = kwargs.get("qwen_base")
        qwen_custom = kwargs.get("qwen_custom")
        qwen_design = kwargs.get("qwen_design")
        
        if all(m is None for m in [qwen_default, qwen_base, qwen_custom, qwen_design]):
            raise ValueError("[AIIA Qwen Router] At least one Qwen model must be connected!")
            
        bundle = {
            "is_bundle": True,
            "default": qwen_default or qwen_base or qwen_custom or qwen_design,
            "base": qwen_base,
            "custom": qwen_custom,
            "design": qwen_design,
            "path": (qwen_default or qwen_base or qwen_custom or qwen_design).get("path", "")
        }
        return (bundle,)


NODE_CLASS_MAPPINGS = {
    "AIIA_Qwen_Loader": AIIA_Qwen_Loader,
    "AIIA_Qwen_TTS": AIIA_Qwen_TTS,
    "AIIA_Qwen_Dialogue_TTS": AIIA_Qwen_Dialogue_TTS,
    "AIIA_Qwen_Model_Router": AIIA_Qwen_Model_Router
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Qwen_Loader": "🤖 Qwen3-TTS Loader",
    "AIIA_Qwen_TTS": "🗣️ Qwen3-TTS Synthesis",
    "AIIA_Qwen_Dialogue_TTS": "🎙️ Qwen3-TTS Dialogue (Specialist)",
    "AIIA_Qwen_Model_Router": "🔌 Qwen3-Model Router (Bundle)"

}
