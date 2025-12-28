
import os
import sys
import torch
import folder_paths
import torchaudio

class AIIA_VibeVoice_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["microsoft/VibeVoice-1.5B"],),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("VIBEVOICE_MODEL",)
    RETURN_NAMES = ("vibevoice_model",)
    FUNCTION = "load_model"
    CATEGORY = "AIIA/VibeVoice"

    def load_model(self, model_name, precision):
        print(f"[AIIA] Loading VibeVoice model: {model_name} ({precision})...")
        
        try:
            from transformers import AutoTokenizer, AutoModel, AutoConfig, Qwen2TokenizerFast
            import transformers
            print(f"[AIIA] Transformers version: {transformers.__version__}")
        except ImportError:
            raise ImportError("Missing dependencies: transformers. Please install it.")

        # Determine torch dtype
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Model Paths
        model_path = os.path.join(folder_paths.models_dir, "vibevoice")
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            
        local_model_path = os.path.join(model_path, "microsoft", "VibeVoice-1.5B") 
        load_path = model_name # Default to HF hub ID
        
        # Try to find local first
        if os.path.exists(local_model_path):
            load_path = local_model_path
            print(f"[AIIA] Found local model at: {load_path}")
        else:
            flat_path = os.path.join(model_path, "VibeVoice-1.5B")
            if os.path.exists(flat_path):
                 load_path = flat_path
                 print(f"[AIIA] Found local model at: {load_path}")
            else:
                 print(f"[AIIA] Model not found locally. Downloading from HuggingFace to cache...")

        # Fix for "KeyError: vibevoice":
        # Strategy: Manually import the model code and register it to AutoConfig/AutoModel
        
        sys_path_added = False

        import importlib.util
        import re
        import types

        try:
            # Paths to search
            config_file_path = os.path.join(load_path if os.path.isdir(load_path) else ".", "configuration_vibevoice_streaming.py") 
            if not os.path.exists(config_file_path):
                 config_file_path = os.path.join(load_path, "configuration_vibevoice.py")
                 
            model_file_path = os.path.join(load_path, "modeling_vibevoice_streaming_inference.py")
            if not os.path.exists(model_file_path):
                 model_file_path = os.path.join(load_path, "modeling_vibevoice.py")
            
            # Ensure sys.path has the load_path so absolute imports work
            if os.path.isdir(load_path) and load_path not in sys.path:
                sys.path.append(load_path)
                sys_path_added = True

            # Helper to load module from file with SOURCE PATCHING
            def load_module_from_path_patched(module_name, file_path):
                if not os.path.exists(file_path):
                     try:
                        return importlib.import_module(module_name)
                     except ImportError:
                        return None
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # PATCH: Convert relative imports to absolute
                source_code = re.sub(r'from \.(\w+)', r'from \1', source_code)
                
                module = types.ModuleType(module_name)
                module.__file__ = file_path
                sys.modules[module_name] = module
                
                exec(source_code, module.__dict__)
                return module

            # Pre-load dependencies manually
            module_order = [
                "configuration_vibevoice",
                "modular_vibevoice_tokenizer",
                "modular_vibevoice_text_tokenizer",
                "streamer",
                "configuration_vibevoice_streaming",
                "modular_vibevoice_diffusion_head",
                "vibevoice_processor",
                "modeling_vibevoice_streaming",
                "modeling_vibevoice_streaming_inference"
            ]
            
            if os.path.isdir(load_path):
                for mod_name in module_order:
                    f_path = os.path.join(load_path, f"{mod_name}.py")
                    if os.path.exists(f_path):
                         load_module_from_path_patched(mod_name, f_path)

            # 1. Get Config Class
            if "configuration_vibevoice_streaming" in sys.modules:
                config_module = sys.modules["configuration_vibevoice_streaming"]
            else:
                 config_module = load_module_from_path_patched("configuration_vibevoice_streaming", config_file_path)

            if config_module and hasattr(config_module, "VibeVoiceStreamingConfig"):
                VibeVoiceConfig = config_module.VibeVoiceStreamingConfig
            elif config_module and hasattr(config_module, "VibeVoiceConfig"):
                VibeVoiceConfig = config_module.VibeVoiceConfig
            else:
                # Fallback if manual load failed
                VibeVoiceConfig = AutoConfig.from_pretrained(load_path, trust_remote_code=True).__class__
            
            # PATCH: Force model_type to match "vibevoice"
            VibeVoiceConfig.model_type = "vibevoice"

            # 2. Register Config
            AutoConfig.register("vibevoice", VibeVoiceConfig)
            
            # 3. Register Tokenizer mapping for this config
            # VibeVoice uses Qwen2Tokenizer
            try:
                from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
                TOKENIZER_MAPPING.register(VibeVoiceConfig, (Qwen2TokenizerFast, None))
            except: 
                pass

            # 4. Get Model Class
            if "modeling_vibevoice_streaming_inference" in sys.modules:
                 model_module = sys.modules["modeling_vibevoice_streaming_inference"]
            else:
                 model_module = load_module_from_path_patched("modeling_vibevoice_streaming_inference", model_file_path)
            
            # Identify the correct class
            if hasattr(model_module, "VibeVoiceStreamingForConditionalGenerationInference"):
                 VibeVoiceClass = model_module.VibeVoiceStreamingForConditionalGenerationInference
            elif hasattr(model_module, "VibeVoiceForConditionalGeneration"):
                 VibeVoiceClass = model_module.VibeVoiceForConditionalGeneration
            else:
                 raise ImportError("Could not find VibeVoice model class.")

            # 5. Register Model Class
            AutoModel.register(VibeVoiceConfig, VibeVoiceClass)
            
            # 6. Load Model
            print("[AIIA] Loading VibeVoice 1.5B using aliased class...")
            config = VibeVoiceConfig.from_pretrained(load_path)
            model = VibeVoiceClass.from_pretrained(
                load_path,
                config=config,
                torch_dtype=dtype,
                device_map="auto"
            )
            
            if sys_path_added:
                sys.path.remove(load_path)

        except Exception as e:
            print(f"[AIIA] Manual import/registration failed: {e}")
            if 'sys_path_added' in locals() and sys_path_added:
                 sys.path.remove(load_path)
            raise e
            
        # 7. Load Tokenizer
        # Try finding Qwen tokenizer path
        tokenizer_load_path = load_path
        
        # Check subfolders for Qwen tokenizer files
        possible_tokenizer_paths = [
            os.path.join(model_path, "tokenizer"), # ../models/vibevoice/tokenizer
            os.path.join(load_path, "tokenizer"),  # ../models/vibevoice/VibeVoice-1.5B/tokenizer
            load_path # Root
        ]
        
        tokenizer = None
        
        # First try AutoTokenizer with our registration
        try:
             print("[AIIA] Attempting AutoTokenizer...")
             tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
        except Exception as e:
             print(f"[AIIA] AutoTokenizer failed ({e}), attempting fallback to Qwen2TokenizerFast...")
             
             # Try fallback paths
             for path in possible_tokenizer_paths:
                 if os.path.exists(path) and (os.path.exists(os.path.join(path, "tokenizer.json")) or os.path.exists(os.path.join(path, "vocab.json"))):
                     print(f"[AIIA] Found tokenizer files at {path}")
                     try:
                        tokenizer = Qwen2TokenizerFast.from_pretrained(path)
                        break
                     except Exception as te:
                        print(f"[AIIA] Failed to load tokenizer from {path}: {te}")
        
        if tokenizer is None:
             raise RuntimeError("Could not load tokenizer. Please ensure Qwen2.5-1.5B tokenizer files (tokenizer.json, vocab.json, merges.txt) are present in the model directory or a 'tokenizer' subdirectory.")
             
        print(f"[AIIA] Tokenizer loaded: {tokenizer.__class__.__name__}")
        
        model.eval()
        
        return ({"model": model, "tokenizer": tokenizer, "dtype": dtype},)


class AIIA_VibeVoice_TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vibevoice_model": ("VIBEVOICE_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of VibeVoice."}),
                "language": (["en", "zh", "ja"], {"default": "en"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/VibeVoice"

    def generate(self, vibevoice_model, text, language, speed, reference_audio=None):
        model = vibevoice_model["model"]
        tokenizer = vibevoice_model["tokenizer"]
        device = model.device # Device map put it somewhere
        
        print(f"[AIIA] Generating VibeVoice TTS... text length: {len(text)}")
        
        prompt_speech = None
        
        if reference_audio is not None:
             wav = reference_audio["waveform"] # [B, C, T]
             sr = reference_audio["sample_rate"]
             
             if wav.ndim == 3:
                 wav = wav[0] # Take batch 0
             if wav.shape[0] > 1:
                 wav = torch.mean(wav, dim=0, keepdim=True) # Mono
                 
             prompt_speech = wav.to(device)
             
        # Inference
        try:
             inputs = tokenizer(text, return_tensors="pt").to(device)
             
             with torch.no_grad():
                output_wav = model.generate(
                    input_ids=inputs["input_ids"],
                    prompt_speech_16k=prompt_speech if prompt_speech is not None else None, 
                    tokenizer=tokenizer,
                )
             
             # Format output
             # output_wav is likely [1, T] tensor
             return ({"waveform": output_wav.cpu().unsqueeze(0), "sample_rate": 24000},) 
             
        except Exception as e:
            print(f"[AIIA ERROR] VibeVoice Inference failed: {e}")
            raise e

NODE_CLASS_MAPPINGS = {
    "AIIA_VibeVoice_Loader": AIIA_VibeVoice_Loader,
    "AIIA_VibeVoice_TTS": AIIA_VibeVoice_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_VibeVoice_Loader": "🎤 VibeVoice Loader (1.5B)",
    "AIIA_VibeVoice_TTS": "🗣️ VibeVoice TTS"
}
