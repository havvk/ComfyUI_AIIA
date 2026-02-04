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
                "preset_note": ("STRING", {"default": QWEN_PRESET_NOTE, "is_label": True}),
                "reference_audio": ("AUDIO",),
                "reference_text": ("STRING", {"multiline": True, "default": ""}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/Synthesis"

    def generate(self, qwen_model, text, language, speaker="Vivian", instruct="", reference_audio=None, reference_text="", x_vector_only=False, seed=42, speed=1.0, cfg_scale=1.5, temperature=0.8, top_k=20, top_p=0.95):
        model = qwen_model["model"]
        m_type = qwen_model["type"]
        
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        lang_param = language if language != "Auto" else "Auto"
        
        # Generation Kwargs
        gen_kwargs = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "cfg_scale": cfg_scale # Usually passed as cfg_scale or guidance_scale in Qwen3-TTS API
        }
        
        wavs = None
        sr = 24000 # Default if unknown
        
        try:
            if m_type == "CustomVoice":
                print(f"[AIIA] Qwen3-TTS CustomVoice: {speaker} | Instruct: {instruct}")
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=lang_param,
                    speaker=speaker,
                    instruct=instruct if instruct else None,
                    **gen_kwargs
                )
            elif m_type == "VoiceDesign":
                print(f"[AIIA] Qwen3-TTS VoiceDesign: {instruct}")
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=lang_param,
                    instruct=instruct,
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
                    mode_param = x_vector_only
                    
                    # Robustness: Qwen requires ref_text for ICL mode (x_vector_only=False)
                    if not ref_text and not mode_param:
                        print(f"[AIIA Warning] No 'reference_text' provided. Automatically switching to 'x_vector_only=True' (Zero-Shot) to prevent crash.")
                        mode_param = True
                    
                    print(f"[AIIA] Qwen3-TTS VoiceClone: Using reference. Mode: {'Zero-Shot' if mode_param else 'ICL'}")
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language=lang_param,
                        ref_audio=ref_audio_data,
                        ref_text=ref_text,
                        x_vector_only_mode=mode_param,
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
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "qwen_base_model": ("QWEN_MODEL",),
                "qwen_custom_model": ("QWEN_MODEL",),
                "qwen_design_model": ("QWEN_MODEL",),
                
                "preset_note": ("STRING", {"default": QWEN_PRESET_NOTE, "is_label": True}),
                
                # Speaker A
                "speaker_A_mode": (["Clone", "Preset", "Design"], {"default": "Clone"}),
                "speaker_A_id": (QWEN_SPEAKER_LIST, {"default": "Vivian"}),
                "speaker_A_design": ("STRING", {"multiline": True, "default": ""}),
                "speaker_A_ref": ("AUDIO",),
                "speaker_A_ref_text": ("STRING", {"multiline": True, "default": ""}),
                
                # Speaker B
                "speaker_B_mode": (["Clone", "Preset", "Design"], {"default": "Clone"}),
                "speaker_B_id": (QWEN_SPEAKER_LIST, {"default": "Vivian"}),
                "speaker_B_design": ("STRING", {"multiline": True, "default": ""}),
                "speaker_B_ref": ("AUDIO",),
                "speaker_B_ref_text": ("STRING", {"multiline": True, "default": ""}),
                
                # Speaker C
                "speaker_C_mode": (["Clone", "Preset", "Design"], {"default": "Design"}),
                "speaker_C_id": (QWEN_SPEAKER_LIST, {"default": "Vivian"}),
                "speaker_C_design": ("STRING", {"multiline": True, "default": ""}),
                "speaker_C_ref": ("AUDIO",),
                "speaker_C_ref_text": ("STRING", {"multiline": True, "default": ""}),
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
            
            # AIIA Fix: Attenuate volume to prevent clipping
            waveform = waveform * 0.8
            
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[AIIA Error] Failed to load fallback audio: {e}")
            return None

    def process_dialogue(self, dialogue_json, pause_duration, speed_global, seed=42, cfg_scale=1.5, temperature=0.8, top_k=20, top_p=0.95, **kwargs):
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

        for i, item in enumerate(dialogue):
            if "speaker" in item:
                spk_name = item["speaker"]
                spk_key = get_speaker_key(spk_name)
                text = item["text"]
                emotion = item.get("emotion", "")
                
                # --- Mode-Based Parameter Assembly ---
                mode = kwargs.get(f"speaker_{spk_key}_mode", "Clone")
                spk_id = kwargs.get(f"speaker_{spk_key}_id", "Vivian")
                design = kwargs.get(f"speaker_{spk_key}_design", "")
                
                # Use fallback if mode is Clone but ref is missing
                ref_audio = get_ref_audio_with_fallback(spk_key) if mode == "Clone" else None
                ref_text = kwargs.get(f"speaker_{spk_key}_ref_text", "")
                
                qwen_model = None
                target_instruct = ""
                target_id = spk_id
                
                if mode == "Clone":
                    qwen_model = kwargs.get("qwen_base_model")
                    if qwen_model is None: qwen_model = kwargs.get("qwen_custom_model") # Fallback
                elif mode == "Preset":
                    qwen_model = kwargs.get("qwen_custom_model")
                    target_instruct = f"{emotion}." if emotion and emotion != "None" else ""
                elif mode == "Design":
                    qwen_model = kwargs.get("qwen_design_model")
                    target_instruct = design if design else f"{emotion}."
                
                if qwen_model is None:
                    # Final fallback to any provided model
                    qwen_model = kwargs.get("qwen_base_model") or kwargs.get("qwen_custom_model") or kwargs.get("qwen_design_model")

                if qwen_model is None:
                    print(f"  [Warning] No Qwen model connected for {spk_name}. Skipping.")
                    continue

                print(f"  [Qwen Specialist] Segment {i}: {spk_name} ({mode}) -> {text[:20]}...")
                
                try:
                    res = qwen_gen.generate(
                        qwen_model=qwen_model,
                        text=text,
                        language="Auto",
                        speaker=target_id,
                        instruct=target_instruct,
                        reference_audio=ref_audio,
                        reference_text=ref_text,
                        seed=seed if seed >= 0 else -1,
                        speed=speed_global,
                        cfg_scale=cfg_scale,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        x_vector_only=kwargs.get("x_vector_only", False)
                    )
                    
                    if res and res[0]:
                        wav = res[0]["waveform"]
                        sr = res[0]["sample_rate"]
                        
                        if sample_rate != sr:
                            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
                        
                        # Remove batch dim if present properly for concatenation
                        wav_data = wav.squeeze(0) # [C, T]
                        full_waveform.append(wav_data)
                        
                        duration = wav_data.shape[1] / sample_rate
                        segments_info.append({
                            "start": time_ptr,
                            "end": time_ptr + duration,
                            "text": text,
                            "speaker": spk_name
                        })
                        time_ptr += duration
                        
                        # Add Pause
                        if pause_duration > 0:
                            silence = torch.zeros((wav_data.shape[0], int(sample_rate * pause_duration)))
                            full_waveform.append(silence)
                            time_ptr += pause_duration
                            
                except Exception as e:
                    print(f"  [Error] Qwen processing failed for {spk_name}: {e}")

        if not full_waveform:
            return ({"waveform": torch.zeros((1, 1, 1024)), "sample_rate": sample_rate}, "[]")

        final_wav = torch.cat(full_waveform, dim=1)
        return ({"waveform": final_wav.unsqueeze(0), "sample_rate": sample_rate}, json.dumps(segments_info, ensure_ascii=False))

NODE_CLASS_MAPPINGS = {
    "AIIA_Qwen_Loader": AIIA_Qwen_Loader,
    "AIIA_Qwen_TTS": AIIA_Qwen_TTS,
    "AIIA_Qwen_Dialogue_TTS": AIIA_Qwen_Dialogue_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Qwen_Loader": "🤖 Qwen3-TTS Loader",
    "AIIA_Qwen_TTS": "🗣️ Qwen3-TTS Synthesis",
    "AIIA_Qwen_Dialogue_TTS": "🎙️ Qwen3-TTS Dialogue (Specialist)"
}
