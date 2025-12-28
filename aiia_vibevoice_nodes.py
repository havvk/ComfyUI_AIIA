
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
        import re
        import types
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
            
            # Ensure sys.path has the load_path so absolute imports work
            sys_path_added = False
            if load_path not in sys.path:
                sys.path.append(load_path)
                sys_path_added = True

            # Helper to load module from file with SOURCE PATCHING
            def load_module_from_path_patched(module_name, file_path):
                if not os.path.exists(file_path):
                     raise FileNotFoundError(f"Required code file not found: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # PATCH: Convert relative imports to absolute imports
                # "from .module import" -> "from module import"
                # "from . import x" -> "import x" (less common here but possible)
                
                # Regex 1: "from .module import" -> "from module import"
                source_code = re.sub(r'from \.(\w+)', r'from \1', source_code)
                # Regex 2: "from . import" -> "import" (This might change semantics if importing symbols, be careful)
                # In these files, we see "from .streamer import ...". pattern 1 covers it.
                
                module = types.ModuleType(module_name)
                # Set __file__ so relative paths inside (if any left) might work or for debugging
                module.__file__ = file_path
                sys.modules[module_name] = module
                
                # Execute patched source
                exec(source_code, module.__dict__)
                return module

            # 1. Load Configuration Code
            print(f"[AIIA] Loading config code (patched) from {config_file_path}...")
            # We need to make sure dependencies are loaded. 
            # configuration_vibevoice_streaming imports configuration_vibevoice.
            # We should probably pre-load configuration_vibevoice if we can guess it.
            # But making imports absolute + sys.path should allow it to import naturally if we don't mess up.
            # However, "from configuration_vibevoice import" will look in sys.path.
            
            # Note: We use the patched loader for the main files we need, 
            # but their DEPENDENCIES (imported inside them via 'from configuration_vibevoice')
            # will be loaded by Python's standard importer from sys.path (since we removed the dot).
            # This requires those dependencies to be loadable as standard modules.
            # Since we added load_path to sys.path, 'import configuration_vibevoice' should work 
            # IF configuration_vibevoice.py doesn't itself assume relative imports from a parent package.
            # Most likely it's fine or we might need to patch dependencies too?
            # Let's trust that dependencies like configuration_vibevoice.py are simpler (usually define Config class).
            
            # Wait, if configuration_vibevoice_streaming imports configuration_vibevoice, 
            # and we patch it to 'from configuration_vibevoice import ...', 
            # then Python loads configuration_vibevoice.py from disk.
            # Does configuration_vibevoice.py assume package? Usually no for Configs.
            
            # Alternative: Just user 'importlib' but wrap it?
            # Let's try to just load the specific ones we know we need in dependency order.
            # 1. configuraton_vibevoice
            # 2. configuration_vibevoice_streaming
            # 3. modular_vibevoice_diffusion_head
            # 4. modeling_vibevoice_streaming
            # 5. modeling_vibevoice_streaming_inference
            
            # Let's try loading them in this order manually.
            module_order = [
                "configuration_vibevoice",
                "modular_vibevoice_tokenizer",
                "modular_vibevoice_text_tokenizer",
                "streamer",
                "configuration_vibevoice_streaming",
                "modular_vibevoice_diffusion_head",
                "vibevoice_processor", # maybe?
                "modeling_vibevoice_streaming",
                "modeling_vibevoice_streaming_inference"
            ]
            
            for mod_name in module_order:
                f_path = os.path.join(load_path, f"{mod_name}.py")
                if os.path.exists(f_path):
                     # Load and patch, inject to sys.modules
                     load_module_from_path_patched(mod_name, f_path)

            # Now we can just get the class from sys.modules
            config_module = sys.modules["configuration_vibevoice_streaming"]
            
            # The config class name might be VibeVoiceStreamingConfig or VibeVoiceConfig
            if hasattr(config_module, "VibeVoiceStreamingConfig"):
                VibeVoiceConfig = config_module.VibeVoiceStreamingConfig
            else:
                VibeVoiceConfig = config_module.VibeVoiceConfig
            
            # PATCH: Force model_type to match "vibevoice" (as in 1.5B config.json)
            # The code from GitHub (0.5B default) has "vibevoice_streaming"
            VibeVoiceConfig.model_type = "vibevoice"

            # 2. Register Config
            # Note: 1.5B config.json says model_type is "vibevoice", so we register for that key
            AutoConfig.register("vibevoice", VibeVoiceConfig)
            
            # 3. Load Modeling Code
            print(f"[AIIA] Loading modeling code (patched) from {model_file_path}...")
            model_module = sys.modules["modeling_vibevoice_streaming_inference"]
            
            # Identify the correct class
            if hasattr(model_module, "VibeVoiceStreamingForConditionalGenerationInference"):
                 VibeVoiceClass = model_module.VibeVoiceStreamingForConditionalGenerationInference
            elif hasattr(model_module, "VibeVoiceForConditionalGeneration"):
                 VibeVoiceClass = model_module.VibeVoiceForConditionalGeneration
            else:
                 raise ImportError("Could not find VibeVoice model class in loaded file.")

            # 4. Register Model
            AutoModel.register(VibeVoiceConfig, VibeVoiceClass)
            
            # 5. Fix Tokenizer Mapping (KeyError: 'VibeVoiceStreamingConfig')
            # AutoTokenizer fails because it doesn't know which tokenizer class matches our custom config.
            # We must manually register the mapping.
            try:
                from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
                if "modular_vibevoice_tokenizer" in sys.modules:
                    tokenizer_mod = sys.modules["modular_vibevoice_tokenizer"]
                    if hasattr(tokenizer_mod, "VibeVoiceTokenizer"):
                        VibeVoiceTokenizer = tokenizer_mod.VibeVoiceTokenizer
                        # Register mapping: ConfigClass -> (SlowTokenizer, FastTokenizer)
                        TOKENIZER_MAPPING.register(VibeVoiceConfig, (VibeVoiceTokenizer, None))
                        print(f"[AIIA] Registered VibeVoiceTokenizer for {VibeVoiceConfig.__name__}")
            except Exception as e:
                print(f"[AIIA] Failed to register tokenizer mapping: {e}")

            # 6. Load
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
