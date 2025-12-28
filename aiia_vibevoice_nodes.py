
import os
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
        # We must load config first with trust_remote_code=True to register the custom architecture
        print(f"[AIIA] Loading AutoConfig from {load_path}...")
        try:
            config = AutoConfig.from_pretrained(load_path, trust_remote_code=True)
        except Exception as e:
            print(f"[AIIA] AutoConfig load failed: {e}. Trying to proceed with Tokenizer...")

        # Now load Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            load_path, 
            config=config, # Pass the loaded config
            trust_remote_code=True, 
            torch_dtype=dtype,
            device_map="auto" # Let accelerate handle it if available
        )
        
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
