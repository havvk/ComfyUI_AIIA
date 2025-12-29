
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
                "model_name": (["microsoft/VibeVoice-1.5B", "microsoft/VibeVoice-7B"],),
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
        
        # Extract model version from model_name (e.g., "microsoft/VibeVoice-1.5B" -> "VibeVoice-1.5B")
        model_version = model_name.split("/")[-1]  # "VibeVoice-1.5B" or "VibeVoice-7B"
        
        # Try to find local first
        local_model_path = os.path.join(model_path, "microsoft", model_version) 
        load_path = model_name # Default to HF hub ID
        
        if os.path.exists(local_model_path):
            load_path = local_model_path
            print(f"[AIIA] Found local model at: {load_path}")
        else:
            flat_path = os.path.join(model_path, model_version)
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
             # Helper to load module from file WITHOUT patching (we rely on static fixes now)
            def load_module_from_path_patched(module_name, file_path):
                if not os.path.exists(file_path):
                     try:
                        return importlib.import_module(module_name)
                     except ImportError:
                        return None
                
                # Get existing or create new module
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                else:
                    module = types.ModuleType(module_name)
                    sys.modules[module_name] = module
                
                module.__file__ = file_path
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    exec(source_code, module.__dict__)
                except Exception as e:
                    print(f"[AIIA ERROR] Failed to exec module {module_name}: {e}")
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

            # Pre-load dependencies manually from VibeVoice Core
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
            load_module_from_path_patched("vibevoice_streaming_processor", os.path.join(core_path, "vibevoice_streaming_processor.py"))

            # 3. Load Model files from Core/Modular
            module_order = [
                "configuration_vibevoice",
                "streamer",
                "configuration_vibevoice_streaming",
                "modeling_vibevoice_streaming",
                "modeling_vibevoice_streaming_inference"
            ]
            
            # Load all core modules explicitly
            for mod_name in module_order:
                bundled_checks = [
                    os.path.join(core_path, f"{mod_name}.py"), 
                    os.path.join(core_path, "modular", f"{mod_name}.py")
                ]
                found = False
                for bp in bundled_checks:
                    if os.path.exists(bp):
                        load_module_from_path_patched(mod_name, bp)
                        print(f"[AIIA] Loaded bundled {mod_name}")
                        found = True
                        break
                if not found:
                    print(f"[AIIA WARNING] Bundled module {mod_name} not found in core!")

            # 1. Get Config Class - PREFER non-streaming config to match non-streaming model
            # First try non-streaming configuration_vibevoice
            config_module = None
            VibeVoiceConfig = None
            use_streaming_config = False
            
            if "configuration_vibevoice" in sys.modules:
                config_module = sys.modules["configuration_vibevoice"]
            
            if config_module and hasattr(config_module, "VibeVoiceConfig"):
                VibeVoiceConfig = config_module.VibeVoiceConfig
                VibeVoiceAcousticTokenizerConfig = getattr(config_module, "VibeVoiceAcousticTokenizerConfig", None)
                VibeVoiceSemanticTokenizerConfig = getattr(config_module, "VibeVoiceSemanticTokenizerConfig", None)
                VibeVoiceDiffusionHeadConfig = getattr(config_module, "VibeVoiceDiffusionHeadConfig", None)
                print("[AIIA] Using non-streaming VibeVoiceConfig")
            else:
                # Fallback to streaming config
                if "configuration_vibevoice_streaming" in sys.modules:
                    config_module = sys.modules["configuration_vibevoice_streaming"]
                
                if config_module and hasattr(config_module, "VibeVoiceStreamingConfig"):
                    VibeVoiceConfig = config_module.VibeVoiceStreamingConfig
                    VibeVoiceAcousticTokenizerConfig = getattr(config_module, "VibeVoiceAcousticTokenizerConfig", None)
                    VibeVoiceSemanticTokenizerConfig = getattr(config_module, "VibeVoiceSemanticTokenizerConfig", None)
                    VibeVoiceDiffusionHeadConfig = getattr(config_module, "VibeVoiceDiffusionHeadConfig", None)
                    use_streaming_config = True
                    print("[AIIA] Using streaming VibeVoiceStreamingConfig (fallback)")
            
            if VibeVoiceConfig is None:
                # Ultimate fallback if manual load failed
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

            # 4. Get Model Class - PREFER non-streaming inference class (as used by TTS-Audio-Suite)
            # Non-streaming version is more stable and proven to work
            model_file_path_non_streaming = os.path.join(core_path, "modular", "modeling_vibevoice_inference.py")
            VibeVoiceClass = None
            
            # Try non-streaming first (recommended)
            if os.path.exists(model_file_path_non_streaming):
                 if "modeling_vibevoice_inference" in sys.modules:
                      model_module = sys.modules["modeling_vibevoice_inference"]
                 else:
                      model_module = load_module_from_path_patched("modeling_vibevoice_inference", model_file_path_non_streaming)
                 
                 if hasattr(model_module, "VibeVoiceForConditionalGenerationInference"):
                      VibeVoiceClass = model_module.VibeVoiceForConditionalGenerationInference
                      print("[AIIA] Using non-streaming VibeVoiceForConditionalGenerationInference (recommended)")
            
            # Fallback to streaming version
            if VibeVoiceClass is None:
                 print("[AIIA] Warning: Falling back to streaming inference class...")
                 if "modeling_vibevoice_streaming_inference" in sys.modules:
                      model_module = sys.modules["modeling_vibevoice_streaming_inference"]
                 else:
                      model_module = load_module_from_path_patched("modeling_vibevoice_streaming_inference", model_file_path)
                 
                 # Identify the correct class
                 if hasattr(model_module, "VibeVoiceStreamingForConditionalGenerationInference"):
                      VibeVoiceClass = model_module.VibeVoiceStreamingForConditionalGenerationInference
                 elif hasattr(model_module, "VibeVoiceForConditionalGeneration"):
                      VibeVoiceClass = model_module.VibeVoiceForConditionalGeneration
            
            if VibeVoiceClass is None:
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
                device_map="auto",
                trust_remote_code=False
            )
            
            # 6.1 Load Generation Config (with bundled fallback)
            try:
                gen_config_path = os.path.join(load_path, "generation_config.json")
                bundled_gen_config_path = os.path.join(core_path, "generation_config.json")
                
                if not os.path.exists(gen_config_path) and os.path.exists(bundled_gen_config_path):
                     print(f"[AIIA] Loading bundled generation_config.json from {bundled_gen_config_path}")
                     from transformers import GenerationConfig
                     final_gen_config = GenerationConfig.from_pretrained(core_path)
                     model.generation_config = final_gen_config
                elif os.path.exists(gen_config_path):
                     print(f"[AIIA] Found generation_config.json in model directory.")
                else:
                     print("[AIIA WARNING] No generation_config.json found (local or bundled). Model might use hardcoded defaults.")
            except Exception as ge:
                print(f"[AIIA WARNING] Failed to load generation config: {ge}")

            # Cleanup not needed as we didn't add load_path to sys.path
            # if sys_path_added:
            #    sys.path.remove(core_path) # We keep core_path for now to ensure sub-modules resolve
            pass

        except Exception as e:
            print(f"[AIIA] Manual import/registration failed: {e}")
            # if 'sys_path_added' in locals() and sys_path_added:
            #      sys.path.remove(load_path)
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
             print("[AIIA] Attempting AutoTokenizer with trust_remote_code=False...")
             tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=False)
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
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "tooltip": "CFG scale for speech generation. Higher = more faithful to text."}),
                "ddpm_steps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 10, "tooltip": "Diffusion steps. Higher = better quality but slower."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "tooltip": "Playback speed. >1 = faster, <1 = slower (post-process time-stretch)."}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/VibeVoice"

    def generate(self, vibevoice_model, text, cfg_scale, ddpm_steps, speed, reference_audio=None):
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
             # Extract input_ids for positional argument
             input_ids = inputs["input_ids"].to(device)
             
             # Prepare other kwargs
             input_args = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items() if k != "input_ids"}
             
             max_new_tokens = 4096 # Default safe limit
             
             # Prepare generation kwargs
             generation_kwargs = {
                 "max_new_tokens": max_new_tokens,
                 "tokenizer": tokenizer, # Model might use it
                 "bos_token_id": 151643, # Qwen2 BOS
                 "eos_token_id": tokenizer.eos_token_id, 
                 "pad_token_id": tokenizer.eos_token_id,
                 "cfg_scale": cfg_scale, # User-controlled CFG scale for speech generation
             }
             
             # Set diffusion inference steps (crucial for quality/speed tradeoff)
             if hasattr(model, "set_ddpm_inference_steps"):
                 model.set_ddpm_inference_steps(num_steps=ddpm_steps)
                 
             with torch.no_grad():
                output_wav = model.generate(
                    input_ids,
                    **input_args,
                    **generation_kwargs
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
                 
             # Ensure tensor format
             if not isinstance(audio_out, torch.Tensor):
                 audio_out = torch.from_numpy(audio_out)
             
             # Ensure [C, T] format for processing
             if audio_out.ndim == 1:
                 audio_out = audio_out.unsqueeze(0)  # [1, T]
             elif audio_out.ndim == 3:
                 audio_out = audio_out.squeeze(0)  # Remove batch dim
             
             # Apply speed adjustment via resampling (post-process time-stretch)
             original_sample_rate = 24000
             if speed != 1.0:
                 effective_rate = int(original_sample_rate * speed)
                 resampler = torchaudio.transforms.Resample(
                     orig_freq=effective_rate,
                     new_freq=original_sample_rate
                 )
                 audio_out = resampler(audio_out)
                 print(f"[AIIA] Applied speed adjustment: {speed}x")
             
             # Ensure [1, C, T] format for ComfyUI
             if audio_out.ndim == 2:
                 audio_out = audio_out.unsqueeze(0)
             
             return ({"waveform": audio_out.cpu(), "sample_rate": original_sample_rate},) 

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
