import os
import torch
import torchaudio

class AIIA_VibeVoice_Realtime_TTS:
    @classmethod
    def INPUT_TYPES(cls):
        # Scan for voice presets
        import folder_paths
        base_path = folder_paths.models_dir
        voice_path = os.path.join(base_path, "vibevoice", "voices", "streaming_model")
        presets = ["None"]
        if os.path.exists(voice_path):
            presets += sorted([f for f in os.listdir(voice_path) if f.endswith(".pt")])

        return {
            "required": {
                "vibevoice_model": ("VIBEVOICE_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test of VibeVoice Realtime."}),
                "voice_preset": (presets,),
                "ddpm_steps": ("INT", {"default": 20, "min": 10, "max": 100, "step": 1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "normalize_text": ("BOOLEAN", {"default": True}),
                "do_sample": (["auto", "false"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0}),
                "cfg_scale": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2147483647, "tooltip": "Random seed for reproducible generation. -1 = random."}),
            },
            "optional": {
                "voice_preset_input": ("VOICE_PRESET",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/VibeVoice"

    def generate(self, vibevoice_model, text, voice_preset, ddpm_steps, speed, normalize_text, 
                 do_sample, temperature, top_k, top_p, cfg_scale, seed=0, voice_preset_input=None): 
        model = vibevoice_model["model"]
        tokenizer = vibevoice_model["tokenizer"]
        processor = vibevoice_model.get("processor")
        is_streaming = vibevoice_model.get("is_streaming", False)
        device = model.device 
        
        if not is_streaming:
            raise ValueError("This node is ONLY for VibeVoice-Realtime-0.5B. Please use 'VibeVoice TTS (Standard)' for other models.")
        
        # Logic: Prioritize direct connection (voice_preset_input) over widget (voice_preset)
        if voice_preset_input is not None and isinstance(voice_preset_input, str) and os.path.exists(voice_preset_input):
             print(f"[AIIA] Using Connected Preset: {voice_preset_input}")
             # When using input, we ignore the widget
             preset_path = voice_preset_input
             voice_preset_name = os.path.basename(voice_preset_input)
        else:
             if voice_preset == "None" or not voice_preset:
                  raise ValueError("You MUST select a Valid 'voice_preset' (or connect one) for the 0.5B model.")
             voice_preset_name = voice_preset
             import folder_paths
             preset_path = os.path.join(folder_paths.models_dir, "vibevoice", "voices", "streaming_model", voice_preset)

        print(f"[AIIA] Generating Realtime... preset={voice_preset_name}, text len={len(text)}")
        
        # Text normalization
        import re
        if normalize_text:
            text = re.sub(r'(\d+年)\s*[-—–]\s*(\d+年)', r'\1至\2', text)
            text = text.replace('"', '').replace("'", '')

        # [AIIA v1.13.0] Strip emotion tags like [Happy], [Calm] etc.
        # These come from Emotion Annotator via Splitter; VibeVoice doesn't support them
        text = re.sub(r'\[(?:neutral|happy|sad|angry|excited|gentle|fearful|surprised|'
                       r'disappointed|serious|calm|romantic|sarcastic|proud|confused|'
                       r'anxious|disgusted|nostalgic|mysterious|enthusiastic|lazy|'
                       r'gossip|innocent|nervous)\]\s*', '', text, flags=re.IGNORECASE)
        
        try:
            # 设置随机种子以确保可复现性
            if seed >= 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                print(f"[AIIA] VibeVoice Realtime seed set to {seed}")

            with torch.no_grad():
                print(f"[AIIA] Using 0.5B Streaming Inference with preset: {voice_preset_name}")
                
                # Load Preset
                if not os.path.exists(preset_path):
                    raise FileNotFoundError(f"Preset file not found at: {preset_path}")

                preset_data = torch.load(preset_path, map_location=device)
                
                # Helper to get attributes safe and move to device + cast to dtype (Recursive for tuples/lists)
                def get_tensor(obj, key):
                     val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
                     return val
                
                # Extract Caches & Hidden States
                lm_cache = get_tensor(preset_data.get('lm'), 'past_key_values')
                lm_last_hidden = get_tensor(preset_data.get('lm'), 'last_hidden_state')
                
                tts_lm_cache = get_tensor(preset_data.get('tts_lm'), 'past_key_values')
                tts_lm_last_hidden = get_tensor(preset_data.get('tts_lm'), 'last_hidden_state')
                
                # Extract Speech Tensors (Critical for Cross-Attention)
                speech_data = preset_data.get('speech')
                if speech_data:
                    speech_tensors = get_tensor(speech_data, 'speech_tensors')
                    speech_masks = get_tensor(speech_data, 'speech_masks')
                    speech_input_mask = get_tensor(speech_data, 'speech_input_mask')
                else:
                    # Fallback for old presets (might fail if model needs them)
                    speech_tensors, speech_masks, speech_input_mask = None, None, None

                if lm_cache is None or tts_lm_cache is None:
                    raise ValueError(f"Invalid preset file: {voice_preset}. Missing cache data.")

                # Extract Negative Caches (if present) or Create Dummy
                if 'neg_lm' in preset_data:
                     neg_lm_cache = get_tensor(preset_data.get('neg_lm'), 'past_key_values')
                     neg_lm_last_hidden = get_tensor(preset_data.get('neg_lm'), 'last_hidden_state')
                     
                     neg_tts_lm_cache = get_tensor(preset_data.get('neg_tts_lm'), 'past_key_values')
                     neg_tts_lm_last_hidden = get_tensor(preset_data.get('neg_tts_lm'), 'last_hidden_state')
                else:
                     print("[AIIA] Preset missing negative cache, generating on fly...")
                     seq_len = lm_cache[0][0].shape[2]
                     neg_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
                     neg_input_ids = torch.full((1, seq_len), neg_input_id, device=device)
                     
                     # Note: This logic assumes model.forward_lm/tts_lm handles everything
                     # But for robustness we should use try/except block if model methods fail? 
                     # Assuming original code was functional for this part.
                     neg_lm_o = model.forward_lm(input_ids=neg_input_ids, attention_mask=torch.ones_like(neg_input_ids), use_cache=True, return_dict=True)
                     neg_lm_cache = neg_lm_o.past_key_values
                     neg_lm_last_hidden = neg_lm_o.last_hidden_state
                     
                     neg_tts_lm_o = model.forward_tts_lm(
                        input_ids=neg_input_ids, 
                        attention_mask=torch.ones_like(neg_input_ids), 
                        use_cache=True, 
                        return_dict=True, 
                        lm_last_hidden_state=neg_lm_o.last_hidden_state, 
                        tts_text_masks=torch.zeros_like(neg_input_ids)
                    )
                     neg_tts_lm_cache = neg_tts_lm_o.past_key_values
                     neg_tts_lm_last_hidden = neg_tts_lm_o.last_hidden_state

                # Universal casting helper
                def cast_recursive(item, dtype, device):
                    if isinstance(item, torch.Tensor):
                         if item.is_floating_point():
                             return item.to(device=device, dtype=dtype)
                         return item.to(device)
                    
                    # Handle DynamicCache by casting internals IN PLACE (preserving object type)
                    try:
                        from transformers.cache_utils import DynamicCache
                        if isinstance(item, DynamicCache):
                             # Qwen2 Strictness: Must be a Cache object. Convert data inside.
                             if hasattr(item, 'key_cache'):
                                 item.key_cache = [cast_recursive(k, dtype, device) for k in item.key_cache]
                             if hasattr(item, 'value_cache'):
                                 item.value_cache = [cast_recursive(v, dtype, device) for v in item.value_cache]
                             return item
                    except:
                        pass

                    if isinstance(item, (list, tuple)):
                         return type(item)(cast_recursive(x, dtype, device) for x in item)
                    return item

                # Robustly get model dtype
                def get_deep_dtype(model_obj):
                    try:
                        return model_obj.model.language_model.model.layers[0].self_attn.q_proj.weight.dtype
                    except:
                        pass
                    try:
                        return model_obj.model.language_model.dtype
                    except:
                        pass
                    try:
                        return next(model_obj.parameters()).dtype
                    except:
                        return model_obj.dtype

                target_dtype = get_deep_dtype(model)
                
                # Force Cast ALL caches to model dtype (Crucial for FP16/Mixed Precision)
                lm_cache = cast_recursive(lm_cache, target_dtype, device)
                tts_lm_cache = cast_recursive(tts_lm_cache, target_dtype, device)
                
                # Handle Negative Cache Variables
                if 'neg_lm_cache' in locals():
                    neg_lm_cache = cast_recursive(neg_lm_cache, target_dtype, device)
                if 'neg_tts_lm_cache' in locals():
                    neg_tts_lm_cache = cast_recursive(neg_tts_lm_cache, target_dtype, device)
                
                # Cast Hidden States
                lm_last_hidden = cast_recursive(lm_last_hidden, target_dtype, device)
                if 'tts_lm_last_hidden' in locals() and tts_lm_last_hidden is not None:
                     tts_lm_last_hidden = cast_recursive(tts_lm_last_hidden, target_dtype, device)
                if 'neg_lm_last_hidden' in locals() and neg_lm_last_hidden is not None:
                     neg_lm_last_hidden = cast_recursive(neg_lm_last_hidden, target_dtype, device)
                if 'neg_tts_lm_last_hidden' in locals() and neg_tts_lm_last_hidden is not None:
                     neg_tts_lm_last_hidden = cast_recursive(neg_tts_lm_last_hidden, target_dtype, device)
                
                # Cast Speech Tensors
                if 'speech_tensors' in locals() and speech_tensors is not None:
                     speech_tensors = cast_recursive(speech_tensors, target_dtype, device)
                if 'speech_masks' in locals() and speech_masks is not None:
                     speech_masks = cast_recursive(speech_masks, torch.bool, device)
                if 'speech_input_mask' in locals() and speech_input_mask is not None:
                     speech_input_mask = cast_recursive(speech_input_mask, torch.bool, device)

                # Use ModelOutput (fixes BOTH 'not iterable' and 'has no attribute' errors)
                from transformers.modeling_outputs import ModelOutput
                

                # Helper to create ModelOutput safe
                def create_output(cache, hidden):
                    if hidden is not None:
                        return ModelOutput(past_key_values=cache, last_hidden_state=hidden)
                    return ModelOutput(past_key_values=cache)

                all_prefilled = {
                    "lm": create_output(lm_cache, lm_last_hidden),
                    "tts_lm": create_output(tts_lm_cache, tts_lm_last_hidden),
                    "neg_lm": create_output(neg_lm_cache, neg_lm_last_hidden),
                    "neg_tts_lm": create_output(neg_tts_lm_cache, neg_tts_lm_last_hidden)
                }

                # Target tokens
                target_text = text # Use text as target
                target_tokens = tokenizer.encode(target_text.strip() + "\n", add_special_tokens=False, return_tensors="pt").to(device)
                
                # Dummy inputs
                # Fix: LM and TTS_LM have DIFFERENT cache lengths. Must track separately.
                
                # Helper to get cache length regardless of tuple or DynamicCache
                def get_cache_len(cache_item):
                     if isinstance(cache_item, (list, tuple)):
                         return cache_item[0][0].shape[2]
                     elif hasattr(cache_item, 'key_cache'): # DynamicCache
                         return cache_item.key_cache[0].shape[2]
                     return 0

                lm_cache_len = get_cache_len(lm_cache)
                tts_cache_len = get_cache_len(tts_lm_cache)
                
                # Main LM inputs (matches lm_cache)
                lm_dummy_ids = torch.zeros((1, lm_cache_len), dtype=torch.long, device=device)
                lm_dummy_mask = torch.ones((1, lm_cache_len), dtype=torch.long, device=device)
                
                # TTS LM inputs (matches tts_lm_cache)
                tts_dummy_ids = torch.zeros((1, tts_cache_len), dtype=torch.long, device=device)
                tts_dummy_mask = torch.ones((1, tts_cache_len), dtype=torch.long, device=device)
                
                # Resolution for sampling
                f_do_sample = getattr(model.generation_config, "do_sample", False) if do_sample == "auto" else (do_sample == "true")
                
                # ComfyUI Progress Bar
                from comfy.utils import ProgressBar
                total_steps = len(text) * 20 # Rough estimate
                pbar = ProgressBar(total_steps)

                def progress_callback(step_increment):
                    pbar.update(step_increment)

                output = model.generate(
                    all_prefilled_outputs=all_prefilled,
                    tts_text_ids=target_tokens,
                    tts_lm_input_ids=tts_dummy_ids,
                    tts_lm_attention_mask=tts_dummy_mask,
                    input_ids=lm_dummy_ids,
                    attention_mask=lm_dummy_mask,
                    max_new_tokens=4000,
                    cfg_scale=cfg_scale,
                    do_sample=f_do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    expected_steps=len(text)*20,
                    max_length_times=10.0,
                    show_progress_bar=True,
                    tokenizer=tokenizer,
                    progress_callback=progress_callback,
                    
                    # Pass Acoustic Conditioning - REMOVED (Official presets don't use this during gen)
                    # speech_tensors=speech_tensors,
                    # speech_masks=speech_masks,
                    # speech_input_mask=speech_input_mask
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
                
                if audio_out.ndim == 1: audio_out = audio_out.unsqueeze(0)
                if audio_out.ndim == 3: audio_out = audio_out.squeeze(0)
                
                if speed != 1.0:
                    audio_out = torchaudio.transforms.Resample(orig_freq=int(24000*speed), new_freq=24000)(audio_out)
                
                if audio_out.ndim == 2: audio_out = audio_out.unsqueeze(0)
                return ({"waveform": audio_out.cpu(), "sample_rate": 24000},)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

NODE_CLASS_MAPPINGS = {
    "AIIA_VibeVoice_Realtime_TTS": AIIA_VibeVoice_Realtime_TTS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_VibeVoice_Realtime_TTS": "🗣️ VibeVoice TTS (Realtime 0.5B)"
}
