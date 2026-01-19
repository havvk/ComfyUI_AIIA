import os
import sys
import torch
import numpy as np
import torchaudio
import folder_paths
import subprocess
import tempfile
import soundfile as sf
# print(f"\n[AIIA DEBUG] Loaded aiia_vibevoice_nodes.py from: {os.path.abspath(__file__)}\n")
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, Qwen2TokenizerFast

def load_module_from_path_patched(module_name, file_path):
    """Utility to load a module from a specific file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # Fallback for some environments
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            exec(source_code, module.__dict__)
        return module
    return None

class AIIA_VibeVoice_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["0.5B (Realtime)", "1.5B (Standard)", "7B (Standard)"], {"default": "1.5B (Standard)"}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("VIBEVOICE_MODEL",)
    RETURN_NAMES = ("vibevoice_model",)
    FUNCTION = "load_vibevoice"
    CATEGORY = "AIIA/VibeVoice"

    @classmethod
    def load_vibevoice(cls, model_version="1.5B (Standard)", dtype="fp16"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if dtype == "fp16" else (torch.bfloat16 if dtype == "bf16" else torch.float32)

        # 0. Resolve Paths
        model_name = "VibeVoice-Realtime-0.5B" if "0.5B" in model_version else ("VibeVoice-7B" if "7B" in model_version else "VibeVoice-1.5B")
        nodes_path = os.path.dirname(os.path.abspath(__file__))
        comfy_root = os.path.dirname(os.path.dirname(nodes_path))
        core_path = os.path.join(nodes_path, "vibevoice_core")

        possible_base_paths = [
            os.path.join(nodes_path, "models", "vibevoice"),
            os.path.join(comfy_root, "models", "vibevoice")
        ]

        load_path = None
        for base in possible_base_paths:
            # Try direct match
            p = os.path.join(base, model_name)
            if os.path.exists(p):
                load_path = p
                break
            # Try with common vendor prefixes
            for prefix in ["microsoft", "vibevoice"]:
                p = os.path.join(base, prefix, model_name)
                if os.path.exists(p):
                    load_path = p
                    break
            if load_path: break

        if not load_path:
            raise FileNotFoundError(f"VibeVoice model '{model_name}' not found. Searched in: {possible_base_paths}. Please ensure it is in ComfyUI/models/vibevoice/{model_name}")

        # Add paths to sys.path
        if core_path not in sys.path: sys.path.insert(0, core_path)
        modular_path = os.path.join(core_path, "modular")
        if modular_path not in sys.path: sys.path.insert(0, modular_path)

        # 1. Detect Model Type (Streaming vs Standard)
        is_streaming_model = False
        try:
            import json
            with open(os.path.join(load_path, "config.json"), "r") as f:
                cfg_data = json.load(f)
                h_size = cfg_data.get("decoder_config", {}).get("hidden_size", 0)
                if 0 < h_size < 1000 or cfg_data.get("model_type") == "vibevoice_streaming":
                    is_streaming_model = True
        except: pass

        # Load Core Modules
        load_module_from_path_patched("vibevoice_tokenizer_processor", os.path.join(core_path, "vibevoice_tokenizer_processor.py"))
        load_module_from_path_patched("vibevoice_processor", os.path.join(core_path, "vibevoice_processor.py"))
        load_module_from_path_patched("vibevoice_streaming_processor", os.path.join(core_path, "vibevoice_streaming_processor.py"))
        
        module_order = ["configuration_vibevoice", "streamer", "configuration_vibevoice_streaming", 
                        "modeling_vibevoice_streaming", "modeling_vibevoice_streaming_inference"]
        for mod_name in module_order:
            for bp in [os.path.join(core_path, f"{mod_name}.py"), os.path.join(core_path, "modular", f"{mod_name}.py")]:
                if os.path.exists(bp):
                    load_module_from_path_patched(mod_name, bp)
                    break

        VibeVoiceConfig = None
        VibeVoiceProcessorClass = None
        VibeVoiceClass = None
        config_module = None
        model_module = None

        if is_streaming_model:
            print("[AIIA] Model identified as Streaming (Realtime) variant.")
            config_module = sys.modules.get("configuration_vibevoice_streaming")
            VibeVoiceConfig = getattr(config_module, "VibeVoiceStreamingConfig", None)
            model_module = sys.modules.get("modeling_vibevoice_streaming_inference")
            VibeVoiceClass = getattr(model_module, "VibeVoiceStreamingForConditionalGenerationInference", None)
            VibeVoiceProcessorClass = getattr(sys.modules.get("vibevoice_streaming_processor"), "VibeVoiceStreamingProcessor", None)
        else:
            print("[AIIA] Model identified as Standard variant.")
            config_module = sys.modules.get("configuration_vibevoice")
            VibeVoiceConfig = getattr(config_module, "VibeVoiceConfig", None)
            inf_file = os.path.join(core_path, "modular", "modeling_vibevoice_inference.py")
            if os.path.exists(inf_file):
                model_module = load_module_from_path_patched("modeling_vibevoice_inference", inf_file)
            else:
                model_module = sys.modules.get("modeling_vibevoice_streaming_inference")
            VibeVoiceClass = getattr(model_module, "VibeVoiceForConditionalGenerationInference", None)
            VibeVoiceProcessorClass = getattr(sys.modules.get("vibevoice_processor"), "VibeVoiceProcessor", None)

        # Registration
        VibeVoiceConfig.model_type = "vibevoice"
        AutoConfig.register("vibevoice", VibeVoiceConfig)
        
        # Register Tokenizer mapping
        try:
            from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
            TOKENIZER_MAPPING.register(VibeVoiceConfig, (Qwen2TokenizerFast, None))
        except: pass

        AutoModel.register(VibeVoiceConfig, VibeVoiceClass)

        # 5. Load Model
        print(f"[AIIA] Loading VibeVoice model variant: {VibeVoiceClass.__name__}")
        config = VibeVoiceConfig.from_pretrained(load_path)
        model = VibeVoiceClass.from_pretrained(load_path, config=config, torch_dtype=dtype, device_map="auto", trust_remote_code=False)
        
        # Load Generation Config
        try:
            from transformers import GenerationConfig
            gen_config_path = os.path.join(load_path, "generation_config.json")
            if os.path.exists(gen_config_path):
                model.generation_config = GenerationConfig.from_pretrained(load_path)
            else:
                h_size = config.decoder_config.hidden_size
                preset = "generation_config_7B.json" if h_size > 2048 else ("generation_config_0.5B.json" if h_size < 1000 else "generation_config_1.5B.json")
                preset_path = os.path.join(core_path, preset)
                if os.path.exists(preset_path):
                    with open(preset_path, "r") as f:
                        model.generation_config = GenerationConfig.from_dict(json.load(f))
        except Exception as ge: print(f"[AIIA WARNING] Failed to load generation config: {ge}")

        # 7. Load Tokenizer & Processor
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=False)
        except:
            for path in [os.path.join(load_path, "tokenizer"), load_path]:
                if os.path.exists(path) and (os.path.exists(os.path.join(path, "tokenizer.json")) or os.path.exists(os.path.join(path, "vocab.json"))):
                    try:
                        tokenizer = Qwen2TokenizerFast.from_pretrained(path)
                        break
                    except: pass
        
        if tokenizer is None: raise RuntimeError("Could not load tokenizer.")
        
        # Inject attributes
        if not hasattr(tokenizer, "speech_diffusion_id"):
            tokenizer.add_tokens(["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>"], special_tokens=True)
            tokenizer.speech_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            tokenizer.speech_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            tokenizer.speech_diffusion_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
            tokenizer.pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # Initialize Processor
        processor = None
        if VibeVoiceProcessorClass:
            try:
                AudioProcessorClass = sys.modules["vibevoice_tokenizer_processor"].VibeVoiceTokenizerProcessor
                processor = VibeVoiceProcessorClass(tokenizer=tokenizer, audio_processor=AudioProcessorClass())
                print(f"[AIIA] Initialized {VibeVoiceProcessorClass.__name__}")
            except Exception as pe: print(f"[AIIA WARNING] Failed to initialize processor: {pe}")

        model.eval()
        return ({"model": model, "tokenizer": tokenizer, "processor": processor, "dtype": dtype, "is_streaming": is_streaming_model},)

class AIIA_VibeVoice_TTS:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "vibevoice_model": ("VIBEVOICE_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of VibeVoice."}),
                # "reference_audio": ("AUDIO",),  <-- Moved to optional
                "cfg_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 10.0, "step": 0.1}),
                "ddpm_steps": ("INT", {"default": 20, "min": 10, "max": 100, "step": 1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "normalize_text": ("BOOLEAN", {"default": True}),
                "do_sample": (["auto", "true", "false"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/VibeVoice"

    def _load_fallback_audio(self):
        import torchaudio
        # 定位 assets 目录 (Shared with Podcast nodes)
        nodes_path = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(nodes_path, "assets")
        path = os.path.join(assets_dir, "seed_female_hq.wav") # Default to female HQ
        
        if not os.path.exists(path):
            print(f"[AIIA Warning] Fallback seed not found at {path}")
            return None
        
        try:
            waveform, sample_rate = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform * 0.8 # Attenuate
            return {"waveform": waveform, "sample_rate": sample_rate}
        except Exception as e:
            print(f"[AIIA Error] Failed to load fallback audio: {e}")
            return None

    def generate(self, vibevoice_model, text, cfg_scale, ddpm_steps, speed, normalize_text, 
                 do_sample, temperature, top_k, top_p, reference_audio=None):
        model = vibevoice_model["model"]
        tokenizer = vibevoice_model["tokenizer"]
        processor = vibevoice_model.get("processor")
        is_streaming = vibevoice_model.get("is_streaming", False)
        device = model.device 
        
        if processor is None: raise RuntimeError("Processor is missing.")
        
        # Validation
        if is_streaming:
             raise ValueError("You are using a 0.5B (Realtime) model with the Standard TTS node. Please use the 'VibeVoice TTS (Realtime 0.5B)' node instead.")

        print(f"[AIIA] Generating... streaming={is_streaming}, text len={len(text)}")
        
        # Text normalization
        import re
        if normalize_text:
            text = re.sub(r'(\d+年)\s*[-—–]\s*(\d+年)', r'\1至\2', text)
            text = text.replace('"', '').replace("'", '')
        
        # Default Speaker Tag
        if not re.search(r'^Speaker\s+\d+\s*:', text, re.IGNORECASE | re.MULTILINE):
            lines = text.split('\n')
            text = "\n".join([f"Speaker 1: {line.strip()}" for line in lines if line.strip()])

        # Process Reference Audio
        if reference_audio is None:
            print("[AIIA INFO] No reference audio provided. Loading default fallback audio...")
            fallback = self._load_fallback_audio()
            if fallback:
                reference_audio = fallback
            else:
                 raise ValueError("No reference audio provided and fallback audio could not be loaded!")

        voice_samples = []
        
        # Determine if input is list or single item
        if isinstance(reference_audio, list):
            raw_refs = reference_audio
        else:
            raw_refs = [reference_audio]
            
        for ref_item in raw_refs:
            if ref_item is None:
                # Should satisfy processor requirements, maybe silent audio
                # But typically caller ensures validity. We'll use 1s silence.
                voice_samples.append(np.zeros(24000, dtype=np.float32))
                continue
                
            wav = ref_item["waveform"]
            ref_sr = ref_item.get("sample_rate", 24000)
            
            if wav.ndim == 3: wav = wav[0]
            if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
            
            if ref_sr != 24000:
                resampler = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=24000)
                wav = resampler(wav)
                
            voice_samples.append(wav.squeeze().cpu().numpy())

        try:
            with torch.no_grad():
                # 1.5B/7B Standard Logic
                inputs = processor(text=text, voice_samples=voice_samples, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                input_args = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items() if k != "input_ids"}
                
                if hasattr(model, "set_ddpm_inference_steps"):
                    model.set_ddpm_inference_steps(num_steps=ddpm_steps)

                # Resolution for sampling
                f_do_sample = getattr(model.generation_config, "do_sample", False) if do_sample == "auto" else (do_sample == "true")
                
                # ComfyUI Progress Bar
                from comfy.utils import ProgressBar
                total_steps = len(text) * 20 # Rough estimate
                pbar = ProgressBar(total_steps)

                def progress_callback(step_increment):
                    pbar.update(step_increment)

                output = model.generate(
                    input_ids,
                    cfg_scale=cfg_scale,
                    do_sample=f_do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    expected_steps=len(text)*20,
                    max_length_times=10.0,
                    tokenizer=tokenizer,
                    progress_callback=progress_callback,
                    **input_args
                )
                
                # Format Audio Output
                if hasattr(output, "speech_outputs"):
                    audio_out = output.speech_outputs[0]
                elif isinstance(output, list):
                    audio_out = output[0]
                else:
                    audio_out = output

                if not isinstance(audio_out, torch.Tensor):
                    audio_out = torch.from_numpy(audio_out)
                
                # Reshape to [1, C, T]
                if audio_out.ndim == 1: audio_out = audio_out.unsqueeze(0)
                if audio_out.ndim == 3: audio_out = audio_out.squeeze(0)
                
                # Ensure float32 (Vital for downstream nodes like Resemble Enhance which fail on fp16)
                audio_out = audio_out.float()

                # Speed adj
                if speed != 1.0:
                    original_device = audio_out.device
                    
                    # Try System 'sox' command for time stretching (pitch preservation)
                    # This is more robust than torchaudio.sox_effects which may be missing in some builds
                    try:
                        # Prepare input
                        audio_cpu = audio_out.cpu().numpy()
                        # Ensure [C, T]
                        if audio_cpu.ndim == 1: audio_cpu = audio_cpu[np.newaxis, :] # [1, T]
                        elif audio_cpu.ndim == 3: audio_cpu = audio_cpu.squeeze(0)   # [C, T]
                        
                        # soundfile writes [T, C]
                        audio_cpu_t = audio_cpu.T 

                        # Normalize to -1dB (approx 0.9) to prevent clipping during sox processing
                        # Sox 'tempo' effect can increase peak amplitude
                        max_val = np.abs(audio_cpu_t).max()
                        if max_val > 0.9:
                             audio_cpu_t = audio_cpu_t * (0.9 / max_val)

                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_f, \
                             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_f:
                            in_path = in_f.name
                            out_path = out_f.name
                        
                        try:
                            # Write temp file
                            sf.write(in_path, audio_cpu_t, 24000)
                            
                            # Call sox
                            # tempo command: changes speed without pitch
                            # -q: quiet
                            cmd = ["sox", "-q", in_path, out_path, "tempo", str(speed)]
                            subprocess.run(cmd, check=True)
                            print(f"[AIIA] Applied time stretch: speed={speed}x (pitch preserved)")
                            
                            # Read back
                            out_wav, out_sr = sf.read(out_path)
                            # sf reads as [T, C] or [T] if mono
                            if out_wav.ndim == 1: 
                                out_wav = out_wav[np.newaxis, :] # [1, T]
                            else:
                                out_wav = out_wav.T # [C, T]
                            
                            audio_out = torch.from_numpy(out_wav).float().to(original_device)
                            
                        finally:
                            if os.path.exists(in_path): os.remove(in_path)
                            if os.path.exists(out_path): os.remove(out_path)
                        
                    except Exception as e:
                        # Fallback to Resample (Pitch Shift)
                        print(f"[AIIA WARNING] System 'sox' failed ({e}), using Resample (Pitch Shift).")
                        resampler = torchaudio.transforms.Resample(orig_freq=int(24000*speed), new_freq=24000).to(original_device)
                        audio_out = resampler(audio_out)
                
                if audio_out.ndim == 2: audio_out = audio_out.unsqueeze(0)

                # --- AIIA FIX: Trim Trailing Silence/Noise ---
                # VibeVoice often generates trailing static/silence. We conservatively trim it.
                try:
                    # Convert to mono for energy calc
                    w_mono = audio_out.squeeze()
                    if w_mono.ndim > 1: w_mono = w_mono.mean(dim=0)
                    
                    # Window size for energy calculation (e.g., 50ms)
                    win_size = int(24000 * 0.05)
                    if w_mono.shape[-1] > win_size:
                        # Calculate short-term energy
                        energy = w_mono.unfold(0, win_size, win_size // 2).pow(2).mean(dim=1)
                        # Threshold (approx -50dB)
                        threshold = 1e-5 
                        
                        # Find last frame above threshold
                        valid_frames = (energy > threshold).nonzero()
                        if valid_frames.numel() > 0:
                            last_valid_idx = valid_frames[-1].item()
                            # Convert back to sample index (roughly)
                            last_sample = (last_valid_idx + 1) * (win_size // 2) + win_size
                            
                            # Add small buffer (0.1s)
                            last_sample = min(last_sample + 2400, w_mono.shape[0])
                            
                            # Apply Trim
                            audio_out = audio_out[..., :last_sample]
                            
                            # Apply Fade Out (0.05s) to avoid clicks
                            fade_len = int(24000 * 0.05)
                            if audio_out.shape[-1] > fade_len:
                                fade = torch.linspace(1, 0, fade_len).to(audio_out.device)
                                audio_out[..., -fade_len:] *= fade
                except Exception as e:
                    print(f"[AIIA WARNING] Failed to trim audio: {e}")

                return ({"waveform": audio_out.cpu(), "sample_rate": 24000},)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e


NODE_CLASS_MAPPINGS = {
    "AIIA_VibeVoice_Loader": AIIA_VibeVoice_Loader,
    "AIIA_VibeVoice_TTS": AIIA_VibeVoice_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_VibeVoice_Loader": "🎤 VibeVoice Loader",
    "AIIA_VibeVoice_TTS": "🗣️ VibeVoice TTS (Standard)"
}

