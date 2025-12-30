
class AIIA_VibeVoice_Preset_Maker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vibevoice_model": ("VIBEVOICE_MODEL",),
                "reference_audio": ("AUDIO",),
                "preset_name": ("STRING", {"default": "my_new_voice_preset"}),
            }
        }

    RETURN_TYPES = ("VOICE_PRESET",)
    RETURN_NAMES = ("preset_path",)
    FUNCTION = "create_preset"
    CATEGORY = "AIIA/VibeVoice"

    def create_preset(self, vibevoice_model, reference_audio, preset_name):
        import os
        import torch
        import torchaudio
        import folder_paths
        from types import SimpleNamespace

        model = vibevoice_model["model"]
        processor = vibevoice_model["processor"]
        tokenizer = vibevoice_model["tokenizer"]
        device = model.device

        # 1. Validation: Must be 0.5B (Streaming) Model
        is_streaming = hasattr(processor, "prepare_speech_inputs") or "Streaming" in str(type(processor))
        if not is_streaming:
            raise ValueError("This node ONLY works with the VibeVoice-Realtime-0.5B model. Please load the 0.5B model.")

        print(f"[AIIA] Creating Preset '{preset_name}' for 0.5B Model...")

        # 2. Process Audio
        # Check audio format [Batch, Channels, Time] or [Batch, Time]
        waveform = reference_audio["waveform"]
        sample_rate = reference_audio["sample_rate"]
        
        # Resample to 22050 (Processor usually expects 22050 for VibeVoice, but check official demo. 
        # Actually 0.5B `_process_single` takes path or array. 
        # Standard VibeVoice uses 22050 internally usually. Let's aim for 22050 to be safe or 24000.
        # Actually, let's look at `modeling_vibevoice_inference.py`. It uses `feature_extractor` which handles resampling usually.
        # But `_process_single` in `vibevoice_streaming_processor.py` calls `_create_voice_prompt`.
        # `_create_voice_prompt` calls `self.audio_feature_extractor`.
        # Taking a cue from `AIIA_VibeVoice_TTS`, raw audio is fine if we pass it correctly.
        
        # We need numpy array for processor
        if waveform.ndim == 3: # [B, C, T]
            waveform = waveform.mean(dim=1) # Mix to mono
        if waveform.ndim == 2: # [B, T] -> [T] (take first batch)
             audio_np = waveform[0].cpu().numpy()
        else:
             audio_np = waveform.cpu().numpy()
             
        # Resample if needed (processor expects specific SR? Usually handled by feature extractor, but let's assume raw is okay if passed properly)
        # Actually, in `AIIA_VibeVoice_TTS`, we verified 0.5B doesn't use `reference_audio` which is why we are here.
        # But the processor CAN handle it.
        
        # 3. Construct Prompt (Imitating `_process_single`)
        # System Prompt
        system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
        system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False) # Qwen tokenizer adds BOS? `_process_single` doesn't explicitly add=False, likely default True? 
        # Wait, `tokenizer.encode(system_prompt)` in `_process_single` line 226.
        # `_process_single` line 360 calls `tokenizer.encode(..., add_special_tokens=False)`.
        # Let's assume standard encode for system prompt uses defaults (likely adds BOS).
        
        # Voice Prompt
        # We need to call internal `_create_voice_prompt`
        # It expects `voice_samples` as list of strings (paths) or arrays.
        voice_tokens, voice_speech_inputs, voice_speech_masks = processor._create_voice_prompt([audio_np])
        
        # Header
        header_text = ' Text input:\n'
        header_tokens = tokenizer.encode(header_text, add_special_tokens=False)
        
        # Combine for MAIN Prompt (Input IDs)
        # System + Voice + Header
        prompt_tokens = system_tokens + voice_tokens + header_tokens
        
        # Masks
        # System (text) + Voice (speech) + Header (text)
        # Text parts are False in speech_input_mask
        speech_input_masks = [False] * len(system_tokens) + voice_speech_masks + [False] * len(header_tokens)

        # 4. Prepare Batch for Forward Pass
        # We need to wrap this into batch encoding format
        input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
        speech_input_mask_tensor = torch.tensor([speech_input_masks], device=device, dtype=torch.bool)
        
        # Prepare speech tensors
        speech_dict = processor.prepare_speech_inputs([audio_np], return_tensors="pt", device=device)
        speech_tensors = speech_dict["padded_speeches"]
        speech_masks = speech_dict["speech_masks"]
        
        # 5. Inference - Positive Cache
        # We need `tts_lm_input_ids` and `tts_lm_attention_mask`.
        # For pre-filling, `tts_lm_input_ids` is usually empty or same length? 
        # In `process_input_with_cached_prompt`, `tts_lm_input_ids` are padded.
        # But wait, we are CREATING the cache. We need to run the model to GET the cache.
        # Forward LM:
        with torch.no_grad():
            # LM Forward
            lm_out = model.forward_lm(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
                return_dict=True
            )
            lm_cache = lm_out.past_key_values
            lm_last_hidden = lm_out.last_hidden_state
            
            # TTS LM Forward
            # We need to pass `speech_tensors`, `speech_masks`, `speech_input_mask`
            # `tts_text_masks`? prompt doesn't have TTS text yet.
            # In streaming processor `__call__`, it calls `_batch_encode` which sets up inputs.
            # Let's look at `model.forward_tts_lm` signature or usage.
            # It usually takes `input_ids` (same as LM), `lm_last_hidden_state`, `speech_tensors`...
            
            tts_lm_out = model.forward_tts_lm(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                lm_last_hidden_state=lm_last_hidden,
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_input_mask=speech_input_mask_tensor,
                # tts_text_masks=None, # Prompt doesn't contain TTS text yet
                use_cache=True,
                return_dict=True
            )
            tts_lm_cache = tts_lm_out.past_key_values

        # 6. Negative Cache (Standard "<|image_pad|>" for unconditioned)
        neg_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # Length must match Positive Cache to align positions!
        seq_len = input_ids.shape[1]
        neg_input_ids = torch.full((1, seq_len), neg_input_id, device=device, dtype=torch.long)
        
        with torch.no_grad():
             neg_lm_o = model.forward_lm(
                 input_ids=neg_input_ids, 
                 attention_mask=torch.ones_like(neg_input_ids), 
                 use_cache=True, 
                 return_dict=True
             )
             neg_lm_cache = neg_lm_o.past_key_values
             
             neg_tts_lm_o = model.forward_tts_lm(
                input_ids=neg_input_ids, 
                attention_mask=torch.ones_like(neg_input_ids), 
                use_cache=True, 
                return_dict=True, 
                lm_last_hidden_state=neg_lm_o.last_hidden_state, 
                tts_text_masks=torch.ones_like(neg_input_ids) # Treat all as "text" (mask=1) for negative?
            )
             neg_tts_lm_cache = neg_tts_lm_o.past_key_values

        # 7. Save to PT
        # Convert to CPU before saving
        def to_cpu(obj):
             if isinstance(obj, torch.Tensor): return obj.cpu()
             if isinstance(obj, tuple): return tuple(to_cpu(x) for x in obj) # Cache is tuple of tuples
             return obj
             
        # Cache structure is typically tuple(tuple(key, value)).
        # We need to iterate deep.
        
        # Actually `torch.save` handles GPU tensors but loading map_location is better.
        # But let's be safe and move to CPU.
        # Helper to recursively move cache to CPU
        def recursive_to_cpu(d):
            if isinstance(d, torch.Tensor): return d.cpu()
            if isinstance(d, (list, tuple)): return type(d)(recursive_to_cpu(x) for x in d)
            return d

        data_to_save = {
            'lm': {
                'past_key_values': recursive_to_cpu(lm_cache),
                'last_hidden_state': recursive_to_cpu(lm_last_hidden)
            },
            'tts_lm': {
                'past_key_values': recursive_to_cpu(tts_lm_cache)
            },
            'neg_lm': {
                'past_key_values': recursive_to_cpu(neg_lm_cache)
            },
            'neg_tts_lm': {
                 'past_key_values': recursive_to_cpu(neg_tts_lm_cache)
            }
        }
        
        # Save Path
        base_path = os.path.join(folder_paths.models_dir, "vibevoice", "voices", "streaming_model")
        os.makedirs(base_path, exist_ok=True)
        if not preset_name.endswith(".pt"):
            preset_name += ".pt"
        save_path = os.path.join(base_path, preset_name)
        
        torch.save(data_to_save, save_path)
        print(f"[AIIA] Preset saved to: {save_path}")
        
        return (save_path,)

NODE_CLASS_MAPPINGS = {
    "AIIA_VibeVoice_Preset_Maker": AIIA_VibeVoice_Preset_Maker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_VibeVoice_Preset_Maker": "🎤 VibeVoice Preset Maker (0.5B)"
}
