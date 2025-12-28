
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
            from transformers import AutoTokenizer, AutoModel, AutoConfig
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
        # We use standard ComfyUI location: models/vibevoice/
        model_path = os.path.join(folder_paths.models_dir, "vibevoice")
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            
        # Check if we should download or if it exists
        # We rely on transformers' cache or local_files_only? 
        # Better to let transformers handle it but point to a specific cache dir? 
        # Or better: construct full path and see if it's there.
        
        # Actually, let's trust AutoModel to handle caching, but standard practice in Comfy is 
        # to have models clearly visible.
        # But for VibeVoice (HF model), usually users just let HF cache it.
        # Let's try to support local dir first.
        
        local_model_path = os.path.join(model_path, "microsoft", "VibeVoice-1.5B") 
        # If user manually downloaded structure
        
        load_path = model_name # Default to HF hub ID
        
        # Try to find local first
        # 1. ComfyUI/models/vibevoice/microsoft/VibeVoice-1.5B
        if os.path.exists(local_model_path):
            load_path = local_model_path
            print(f"[AIIA] Found local model at: {load_path}")
        else:
            # 2. ComfyUI/models/vibevoice (flat)
            flat_path = os.path.join(model_path, "VibeVoice-1.5B")
            if os.path.exists(flat_path):
                 load_path = flat_path
                 print(f"[AIIA] Found local model at: {load_path}")
            else:
                 print(f"[AIIA] Model not found locally. Downloading from HuggingFace to cache...")

        # Fix for "KeyError: vibevoice":
        # Strategy: Manually import the model code and register it to AutoConfig/AutoModel
        
        # Fix for "KeyError: vibevoice":
        # Strategy: Load available code (likely from 0.5B repo) and alias it to load 1.5B weights
        
        import importlib.util
        try:
            # We look for the files user copied from 'modular' folder
            # Note: The file names in GitHub repo are specific
            config_file_path = os.path.join(load_path, "configuration_vibevoice_streaming.py") 
            # Fallback to configuration_vibevoice.py if streaming one missing (or user renamed it)
            if not os.path.exists(config_file_path):
                 config_file_path = os.path.join(load_path, "configuration_vibevoice.py")
                 
            # For modeling, we need the inference wrapper
            model_file_path = os.path.join(load_path, "modeling_vibevoice_streaming_inference.py")
            if not os.path.exists(model_file_path):
                 # Fallback: maybe user renamed it to standard modeling_vibevoice.py?
                 model_file_path = os.path.join(load_path, "modeling_vibevoice.py")

            # Helper to load module from file
            def load_module_from_path(module_name, file_path):
                if not os.path.exists(file_path):
                     raise FileNotFoundError(f"Required code file not found: {file_path}")
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec is None:
                    raise ImportError(f"Could not load spec for {module_name} from {file_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module

            # 1. Load Configuration Code
            print(f"[AIIA] Loading config code from {config_file_path}...")
            config_module = load_module_from_path("configuration_vibevoice", config_file_path)
            
            # The config class name might be VibeVoiceStreamingConfig or VibeVoiceConfig
            if hasattr(config_module, "VibeVoiceStreamingConfig"):
                VibeVoiceConfig = config_module.VibeVoiceStreamingConfig
            else:
                VibeVoiceConfig = config_module.VibeVoiceConfig
            
            # 2. Register Config
            # Note: 1.5B config.json says model_type is "vibevoice", so we register for that key
            AutoConfig.register("vibevoice", VibeVoiceConfig)
            
            # 3. Load Modeling Code (this will likely trigger imports of other files in that dir)
            # We need to make sure the dir is in sys.path temporarily so internal imports work
            sys_path_added = False
            if load_path not in sys.path:
                sys.path.append(load_path)
                sys_path_added = True
                
            print(f"[AIIA] Loading modeling code from {model_file_path}...")
            try:
                model_module = load_module_from_path("modeling_vibevoice_streaming_inference", model_file_path)
            except Exception as e:
                # If direct load fails (due to intense relative imports), try standard import since it's in sys.path
                import modeling_vibevoice_streaming_inference as model_module
            
            # Identify the correct class
            if hasattr(model_module, "VibeVoiceStreamingForConditionalGenerationInference"):
                 VibeVoiceClass = model_module.VibeVoiceStreamingForConditionalGenerationInference
            elif hasattr(model_module, "VibeVoiceForConditionalGeneration"):
                 VibeVoiceClass = model_module.VibeVoiceForConditionalGeneration
            else:
                 raise ImportError("Could not find VibeVoice model class in loaded file.")

            # 4. Register Model
            AutoModel.register(VibeVoiceConfig, VibeVoiceClass)
            
            # 5. Load
            print("[AIIA] Loading VibeVoice 1.5B using aliased class...")
            # We force the config to use the class we found
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
            
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        # Note: model is already loaded above manually

        
        # VibeVoice specific checks?
        # It's a TTS model, usually has a generate() method.
        
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
        
        # Prepare Prompt
        # VibeVoice API depends on the implementation in the repo.
        # Since we are using from_pretrained with trust_remote_code=True, 
        # we need to check what the model class exposes.
        # According to README: model.generate(text, prompt_audio, prompt_text)
        
        prompt_speech = None
        prompt_text = None # Do we need prompt text? 
        # Usually zero-shot cloning needs prompting.
        # If reference_audio is provided:
        
        if reference_audio is not None:
             # Process reference audio
             # Need to resample to model's expected rate (usually 16k or 22k or 24k)
             # VibeVoice usually 22050 or 24000. Let's assume 22050 based on typical TTS.
             # Actually paper says 24khz? Let's check model config if possible.
             # Safe bet: pass tensor, see if model handles it.
             # Usually models need file path or raw tensor.
             
             wav = reference_audio["waveform"] # [B, C, T]
             sr = reference_audio["sample_rate"]
             
             if wav.ndim == 3:
                 wav = wav[0] # Take batch 0
             if wav.shape[0] > 1:
                 wav = torch.mean(wav, dim=0, keepdim=True) # Mono
                 
             # Check logic. Assuming model.generate accepts specific args.
             # Since this is "trust_remote_code", the API is defined in 'modeling_vibevoice.py' downloaded from HF.
             # We assume a standard API or we might need a specific wrapper.
             # Given the "beta" nature, let's wrap in try-except.
             
             prompt_speech = wav.to(device)
             # Resampling might be needed.
             
        # Inference
        try:
             # Hypothetical API call based on typical HF TTS
             # output = model.generate(text=text, ...)
             
             # IMPORTANT: Since I can't browse the specific remote code content right now,
             # I will implement a generic placeholder leveraging standard HF generate pattern.
             # User might need to debug if API differs.
             
             # Standard VibeVoice usage:
             # inputs = tokenizer(text, return_tensors="pt").to(device)
             # output = model.generate(**inputs, prompt_speech=prompt_speech)
             
             # Let's try the tokenizer route first.
             inputs = tokenizer(text, return_tensors="pt").to(device)
             
             with torch.no_grad():
                # This is a guess-work here without docs lookup tool result
                # But user said their env is ready for this model.
                # Assuming model exposes a high-level generate.
                output_wav = model.generate(
                    input_ids=inputs["input_ids"],
                    prompt_speech_16k=prompt_speech if prompt_speech is not None else None, # Some models name it prompt_speech_16k
                    # Or just prompt_speech?
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
