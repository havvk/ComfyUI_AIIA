
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
                # NOTE: We intentionally DO NOT check sys.modules early return here for some modules
                # to ensure our hot-patches are applied even if the module was loaded by other means.
                # However, completely reloading might break isinstance checks if classes are recreated.
                # So we try to reuse the module object but EXECUTE the patched source into it.
                
                if not os.path.exists(file_path):
                     try:
                        return importlib.import_module(module_name)
                     except ImportError:
                        return None
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                
                # PATCH: Convert relative imports to absolute
                # Regex handles "from .module" with variable spacing
                source_code, n_subs = re.subn(r'from\s+\.(\w+)', r'from \1', source_code)
                if n_subs > 0:
                    print(f"[AIIA] Patched {n_subs} relative imports in {module_name}")

                # PATCH: Fix unsafe .to(device) on potential None types in VibeVoice generation code
                # IDEMPOTENT PATCH: We use negative lookahead to avoid double-patching if the check already exists.
                
                # 1. speech_tensors
                source_code, n_subs = re.subn(
                    r'(([\"\'])speech_tensors\2\s*:\s*speech_tensors\.to\(.*?\))(?!\s*if)',
                    r'"speech_tensors": speech_tensors.to(device) if speech_tensors is not None else None',
                    source_code,
                    flags=re.DOTALL
                )
                if n_subs > 0: print(f"[AIIA] Hot-patched speech_tensors safety check ({n_subs} hits)")

                # 2. speech_masks
                source_code, n_subs = re.subn(
                    r'(([\"\'])speech_masks\2\s*:\s*speech_masks\.to\(.*?\))(?!\s*if)',
                    r'"speech_masks": speech_masks.to(device) if speech_masks is not None else None',
                    source_code,
                    flags=re.DOTALL
                )
                if n_subs > 0: print(f"[AIIA] Hot-patched speech_masks safety check ({n_subs} hits)")

                # 3. speech_input_mask
                source_code, n_subs = re.subn(
                    r'(([\"\'])speech_input_mask\2\s*:\s*speech_input_mask\.to\(.*?\))(?!\s*if)',
                    r'"speech_input_mask": speech_input_mask.to(device) if speech_input_mask is not None else None',
                    source_code,
                    flags=re.DOTALL
                )
                if n_subs > 0: print(f"[AIIA] Hot-patched speech_input_mask safety check ({n_subs} hits)")
                
                # Get existing or create new module
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    print(f"[AIIA] Updating existing module: {module_name}")
                else:
                    module = types.ModuleType(module_name)
                    sys.modules[module_name] = module
                
                module.__file__ = file_path
                
                try:
                    exec(source_code, module.__dict__)
                except Exception as e:
                    print(f"[AIIA ERROR] Failed to exec module {module_name}: {e}")
                    # Print context around error line if possible
                    if hasattr(e, 'lineno'):
                        lines = source_code.split('\n')
                        start = max(0, e.lineno - 5)
                        end = min(len(lines), e.lineno + 5)
                        print(f"[AIIA DEBUG] Context around line {e.lineno}:")
                        for i in range(start, end):
                            indicator = ">>" if i + 1 == e.lineno else "  "
                            print(f"{indicator} {i+1}: {lines[i]}")
                    raise e
                return module
            # Add VibeVoice core path to sys.path
            nodes_path = os.path.dirname(os.path.abspath(__file__))
            core_path = os.path.join(nodes_path, "vibevoice_core")
            if os.path.exists(core_path):
                 sys.path.insert(0, core_path) # Insert at 0 to prioritize our bundled code
                 sys_path_added = True
                 print(f"[AIIA] Added bundled VibeVoice core to path: {core_path}")
            else:
                 sys_path_added = False
                 print(f"[AIIA WARNING] Bundled VibeVoice core not found at {core_path}")

            # Pre-load dependencies manually
            # 1. Load Modular files first (as processor depends on them)
            if os.path.isdir(os.path.join(core_path, "modular")):
                 modular_path = os.path.join(core_path, "modular")
                 sys.path.insert(0, modular_path) # Add to path so processed imports (from .config) work
                 load_module_from_path_patched("modular_vibevoice_tokenizer", os.path.join(modular_path, "modular_vibevoice_tokenizer.py"))
                 load_module_from_path_patched("modular_vibevoice_text_tokenizer", os.path.join(modular_path, "modular_vibevoice_text_tokenizer.py"))
                 load_module_from_path_patched("modular_vibevoice_diffusion_head", os.path.join(modular_path, "modular_vibevoice_diffusion_head.py"))
            
            # 2. Load Processors from Core
            load_module_from_path_patched("vibevoice_tokenizer_processor", os.path.join(core_path, "vibevoice_tokenizer_processor.py"))
            load_module_from_path_patched("vibevoice_processor", os.path.join(core_path, "vibevoice_processor.py"))

            # 3. Load Model files (Try from Model Dir first for Config, or Bundled if generic)
            module_order = [
                "configuration_vibevoice",
                "streamer",
                "configuration_vibevoice_streaming",
                "modeling_vibevoice_streaming",
                "modeling_vibevoice_streaming_inference"
            ]
            
            if os.path.isdir(load_path):
                # Search in root and subdirectories
                search_paths = [load_path, os.path.join(load_path, "modular")]
                
                for mod_name in module_order:
                    found = False
                    
                    # 1. Check Bundled Core FIRST (To use our fixes)
                    bundled_checks = [
                        os.path.join(core_path, f"{mod_name}.py"), 
                        os.path.join(core_path, "modular", f"{mod_name}.py")
                    ]
                    for bp in bundled_checks:
                        if os.path.exists(bp):
                            load_module_from_path_patched(mod_name, bp)
                            print(f"[AIIA] Loaded bundled {mod_name}")
                            found = True
                            break
                    
                    # 2. Check Model Directory (Fallback)
                    if not found:
                         for p in search_paths:
                             f_path = os.path.join(p, f"{mod_name}.py")
                             if os.path.exists(f_path):
                                 load_module_from_path_patched(mod_name, f_path)
                                 found = True
                                 break
                                
                    if not found:
                        print(f"[AIIA WARNING] Could not find module file for {mod_name}")

            # PRE-PATCH: Explicitly patch dependency modules that contain logic (like generate)
            # which might be imported by the main model file. Standard imports won't use our patched loader!
            pre_patch_modules = [
                "modeling_vibevoice_inference", 
                "modeling_vibevoice_streaming",
                "streamer"
            ]
            for mod in pre_patch_modules:
                if mod not in sys.modules:
                    # Look for it in core
                    p_path = os.path.join(core_path, "modular", f"{mod}.py")
                    if os.path.exists(p_path):
                         load_module_from_path_patched(mod, p_path)

            # 1. Get Config Class
            if "configuration_vibevoice_streaming" in sys.modules:
                config_module = sys.modules["configuration_vibevoice_streaming"]
            else:
                 config_module = load_module_from_path_patched("configuration_vibevoice_streaming", config_file_path)

            if config_module and hasattr(config_module, "VibeVoiceStreamingConfig"):
                VibeVoiceConfig = config_module.VibeVoiceStreamingConfig
                VibeVoiceAcousticTokenizerConfig = getattr(config_module, "VibeVoiceAcousticTokenizerConfig", None)
                VibeVoiceSemanticTokenizerConfig = getattr(config_module, "VibeVoiceSemanticTokenizerConfig", None)
                VibeVoiceDiffusionHeadConfig = getattr(config_module, "VibeVoiceDiffusionHeadConfig", None)
                VibeVoiceStreamingConfig = getattr(config_module, "VibeVoiceStreamingConfig", None)
            elif config_module and hasattr(config_module, "VibeVoiceConfig"):
                VibeVoiceConfig = config_module.VibeVoiceConfig
                VibeVoiceAcousticTokenizerConfig = getattr(config_module, "VibeVoiceAcousticTokenizerConfig", None)
                VibeVoiceSemanticTokenizerConfig = getattr(config_module, "VibeVoiceSemanticTokenizerConfig", None)
                VibeVoiceDiffusionHeadConfig = getattr(config_module, "VibeVoiceDiffusionHeadConfig", None)
                VibeVoiceStreamingConfig = getattr(config_module, "VibeVoiceStreamingConfig", None)
            else:
                # Fallback if manual load failed
                VibeVoiceConfig = AutoConfig.from_pretrained(load_path, trust_remote_code=True).__class__
                VibeVoiceAcousticTokenizerConfig = None
            
            # PATCH: Force model_type to match "vibevoice"
            VibeVoiceConfig.model_type = "vibevoice"

            # 2. Register Configs
            AutoConfig.register("vibevoice", VibeVoiceConfig)
            
            # Register auxiliary configs if available
            if VibeVoiceAcousticTokenizerConfig: AutoConfig.register("vibevoice_acoustic_tokenizer", VibeVoiceAcousticTokenizerConfig)
            if VibeVoiceSemanticTokenizerConfig: AutoConfig.register("vibevoice_semantic_tokenizer", VibeVoiceSemanticTokenizerConfig)
            if VibeVoiceDiffusionHeadConfig: AutoConfig.register("vibevoice_diffusion_head", VibeVoiceDiffusionHeadConfig)
            
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

            # Retrieve sub-model classes for registration
            # NOTE: modeling_vibevoice_inference might NOT have these classes imported, so we try checking there first,
            # but if not found, we load `modeling_vibevoice` which definitely has them.
            VibeVoiceAcousticTokenizerModel = getattr(model_module, "VibeVoiceAcousticTokenizerModel", None)
            VibeVoiceSemanticTokenizerModel = getattr(model_module, "VibeVoiceSemanticTokenizerModel", None)
            VibeVoiceDiffusionHead = getattr(model_module, "VibeVoiceDiffusionHead", None)
            
            if not (VibeVoiceAcousticTokenizerModel and VibeVoiceSemanticTokenizerModel):
                print("[AIIA] Sub-model classes not found in inference module, loading modeling_vibevoice...")
                try:
                    if "modeling_vibevoice" in sys.modules:
                         mod_vv = sys.modules["modeling_vibevoice"]
                    else:
                         mod_vv = load_module_from_path_patched("modeling_vibevoice", os.path.join(core_path, "modular", "modeling_vibevoice.py"))
                    
                    if mod_vv:
                        VibeVoiceAcousticTokenizerModel = getattr(mod_vv, "VibeVoiceAcousticTokenizerModel", VibeVoiceAcousticTokenizerModel)
                        VibeVoiceSemanticTokenizerModel = getattr(mod_vv, "VibeVoiceSemanticTokenizerModel", VibeVoiceSemanticTokenizerModel)
                        VibeVoiceDiffusionHead = getattr(mod_vv, "VibeVoiceDiffusionHead", VibeVoiceDiffusionHead)
                except Exception as e:
                    print(f"[AIIA WARNING] Failed to load modeling_vibevoice for sub-models: {e}")

            # 5. Register Model Classes
            AutoModel.register(VibeVoiceConfig, VibeVoiceClass)
            
            # Register auxiliary model classes
            if VibeVoiceAcousticTokenizerConfig and VibeVoiceAcousticTokenizerModel:
                AutoModel.register(VibeVoiceAcousticTokenizerConfig, VibeVoiceAcousticTokenizerModel)
            if VibeVoiceSemanticTokenizerConfig and VibeVoiceSemanticTokenizerModel:
                AutoModel.register(VibeVoiceSemanticTokenizerConfig, VibeVoiceSemanticTokenizerModel)
            if VibeVoiceDiffusionHeadConfig and VibeVoiceDiffusionHead:
                AutoModel.register(VibeVoiceDiffusionHeadConfig, VibeVoiceDiffusionHead)
            
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
            
        # 7. Load Tokenizer & Processor
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
        
        # MONKEY PATCH: Ensure tokenizer has VibeVoice special properties
        # The VibeVoiceProcessor relies on these attributes which specific VibeVoiceTextTokenizer has.
        # But since we load generic Qwen2TokenizerFast, we must inject them.
        if not hasattr(tokenizer, "speech_diffusion_id"):
            print("[AIIA] Monkey-patching tokenizer with VibeVoice attributes...")
            
            # 1. Add Special Tokens if missing
            special_tokens = ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>"]
            tokenizer.add_tokens(special_tokens, special_tokens=True)

            # 2. Get IDs
            speech_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            speech_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            speech_diffusion_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
            pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            
            # 3. Inject Properties
            # We use bound properties or just set attributes depending on how it's accessed (as a property or attr)
            # The class uses @property, but setting instance attribute shadows class property usually.
            tokenizer.speech_start_id = speech_start_id
            tokenizer.speech_end_id = speech_end_id
            tokenizer.speech_diffusion_id = speech_diffusion_id
            tokenizer.pad_id = pad_id
            
            print(f"[AIIA] Tokenizer Patched: speech_diffusion_id={speech_diffusion_id}")
        
        # 8. Init VibeVoiceProcessor
        processor = None
        if "vibevoice_processor" in sys.modules:
             try:
                 VibeVoiceProcessor = sys.modules["vibevoice_processor"].VibeVoiceProcessor
                 # Initialize processor with our loaded tokenizer
                 # We need to manually initialize audio_processor as well since we are bypassing from_pretrained
                 if "vibevoice_tokenizer_processor" in sys.modules:
                     AudioProcessorClass = sys.modules["vibevoice_tokenizer_processor"].VibeVoiceTokenizerProcessor
                     audio_processor = AudioProcessorClass()
                     processor = VibeVoiceProcessor(tokenizer=tokenizer, audio_processor=audio_processor)
                     print("[AIIA] VibeVoiceProcessor initialized successfully.")
                 else:
                     print("[AIIA WARNING] vibevoice_tokenizer_processor not found, skipping processor init.")
             except Exception as pe:
                 print(f"[AIIA WARNING] Failed to initialize VibeVoiceProcessor: {pe}")
        
        model.eval()
        
        return ({"model": model, "tokenizer": tokenizer, "processor": processor, "dtype": dtype},)


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
        processor = vibevoice_model.get("processor")
        device = model.device 
        
        if processor is None:
             raise RuntimeError("VibeVoiceProcessor is missing. Model loading might have been incomplete.")
             
        if reference_audio is None:
             raise ValueError("Reference Audio is REQUIRED for VibeVoice. Please connect an AUDIO input (e.g., Load Audio, Microphone).")

        print(f"[AIIA] Generating VibeVoice TTS... text length: {len(text)}")
        
        # FIX: Processor expects script format "Speaker X: text" for raw strings
        import re
        if not re.search(r'^Speaker\s+\d+\s*:', text, re.IGNORECASE | re.MULTILINE):
            print("[AIIA] No speaker tag found, adding default 'Speaker 1:' prefix")
            # If multi-line, prefix each line that has content?
            # Or just wrap the whole thing?
            # VibeVoice supports multiple speakers. If user gives plain text, assume single speaker.
            
            # Simple approach: Prepend Speaker 1: to the whole block?
            # The processor parsing logic splits by newline.
            # If I wrap the whole text, newlines inside might be treated as... 
            # Processor parse_script iterates lines. Each line MUST start with Speaker X:.
            
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                if line.strip():
                     formatted_lines.append(f"Speaker 1: {line.strip()}")
            text = "\n".join(formatted_lines)
            print(f"[AIIA] Formatted input: {text[:50]}...")
        
        # Prepare reference audio for processor
        voice_samples = None
        if reference_audio is not None:
             wav = reference_audio["waveform"] # [B, C, T]
             # Processor expects numpy array or path
             if wav.ndim == 3:
                 wav = wav[0] # Take batch 0
             if wav.shape[0] > 1:
                 wav = torch.mean(wav, dim=0, keepdim=True) # Mono
             
             # Convert to numpy [T]
             wav_np = wav.squeeze().cpu().numpy()
             voice_samples = [wav_np] # List for batch processing logic in processor
        
        # Use Processor to prepare inputs
        try:
             # Processor returns BatchEncoding
             inputs = processor(
                 text=text,
                 voice_samples=voice_samples,
                 return_tensors="pt"
             )
             
             # Move all tensors to device
             input_args = {
                 "input_ids": inputs["input_ids"].to(device),
                 "speech_tensors": inputs["speech_tensors"].to(device) if inputs["speech_tensors"] is not None else None,
                 "speech_masks": inputs["speech_masks"].to(device) if inputs["speech_masks"] is not None else None,
                 "speech_input_mask": inputs["speech_input_mask"].to(device) if inputs["speech_input_mask"] is not None else None,
                 "tokenizer": tokenizer, # Pass tokenizer as required by model.generate logic
             }
             
             with torch.no_grad():
                output_wav = model.generate(
                    **input_args
                )
             
             # Format output
             # output_wav is likely list of numpy arrays or tensors from generate
             # Check output format from modeling_vibevoice_inference.py:
             # Returns: VibeVoiceGenerationOutput or tensors. 
             # Our node implementation of generate returns VibeVoiceGenerationOutput class if return_dict=True (default)
             
             if hasattr(output_wav, "speech_outputs"):
                 audio_out = output_wav.speech_outputs[0] # List[Tensor]
             elif isinstance(output_wav, list):
                 audio_out = output_wav[0]
             else:
                 audio_out = output_wav

             if audio_out is None:
                 raise RuntimeError("No audio generated.")
                 
             # Ensure [1, C, T] format for ComfyUI
             if isinstance(audio_out, torch.Tensor):
                 if audio_out.ndim == 1:
                     audio_out = audio_out.unsqueeze(0).unsqueeze(0)
                 elif audio_out.ndim == 2:
                     audio_out = audio_out.unsqueeze(0)
                 return ({"waveform": audio_out.cpu(), "sample_rate": 24000},) 
             else:
                 # Numpy fallback
                 return ({"waveform": torch.from_numpy(audio_out).unsqueeze(0).unsqueeze(0), "sample_rate": 24000},) 

        except Exception as e:
            print(f"[AIIA ERROR] VibeVoice Inference failed: {e}")
            import traceback
            traceback.print_exc()
            raise e

NODE_CLASS_MAPPINGS = {
    "AIIA_VibeVoice_Loader": AIIA_VibeVoice_Loader,
    "AIIA_VibeVoice_TTS": AIIA_VibeVoice_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_VibeVoice_Loader": "🎤 VibeVoice Loader (1.5B)",
    "AIIA_VibeVoice_TTS": "🗣️ VibeVoice TTS"
}
