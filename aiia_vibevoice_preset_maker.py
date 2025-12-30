import os

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
        # ComfyUI AUDIO is typically [Batch, Samples, Channels] or [Batch, Samples]
        if waveform.ndim == 3: 
            # Check heuristic: if dim 2 is small (channels) and dim 1 is large (time) -> [B, T, C]
            if waveform.shape[2] < waveform.shape[1]:
                # Input is [B, T, C]. Mix query channels (dim 2) to mono
                waveform = waveform.mean(dim=2) # -> [B, T]
            else:
                 # Input is [B, C, T]. Mix query channels (dim 1) to mono
                 waveform = waveform.mean(dim=1) # -> [B, T]
                 
        if waveform.ndim == 2: # [B, T] -> [T] (take first batch)
             waveform = waveform[0]
        
        # Resample to Processor's expected SR
        target_sr = 22050 # Default for VibeVoice / Encodec
        if hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "sampling_rate"):
            target_sr = processor.feature_extractor.sampling_rate
            
        if sample_rate != target_sr:
            print(f"[AIIA] Resampling audio from {sample_rate} to {target_sr}")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = waveform.cpu() # ensure cpu for torchaudio transforms usually
            if waveform.dim() == 1: waveform = waveform.unsqueeze(0) # [1, T]
            waveform = resampler(waveform)
            waveform = waveform.squeeze(0) # [T]
            
        audio_np = waveform.cpu().numpy()
        
        # KEY FIX: Normalize audio explicitly for consistency with processor
        # _create_voice_prompt normalizes internally, but prepare_speech_inputs DOES NOT.
        # We must normalize here so both share the same volume level (-25dB).
        if hasattr(processor, "audio_normalizer") and processor.audio_normalizer:
             print("[AIIA] Normalizing audio volume to -25dB...")
             audio_np = processor.audio_normalizer(audio_np)
        
        # 3. Construct Prompt (Imitating `_process_single`)
        # System Prompt
        system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
        system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False) 

        # Voice Prompt
        # We need to call internal `_create_voice_prompt`
        voice_tokens, voice_speech_inputs, voice_speech_masks = processor._create_voice_prompt([audio_np])
        
        # Debug Voice Tokens
        if len(voice_tokens) > 0:
             print(f"[AIIA] Generated {len(voice_tokens)} Voice Tokens. Range: [{min(voice_tokens)}, {max(voice_tokens)}]")
        else:
             print("[AIIA] WARNING: Generated 0 Voice Tokens! Input audio might be silent or too short.")
        
        # Header
        header_text = ' Text input:\n'
        header_tokens = tokenizer.encode(header_text, add_special_tokens=False)
        
        
        # Add " Speaker 0:" suffix (Crucial for prompt continuity!)
        prefix_text = " Speaker 0:"
        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        
        # KEY FIX: Separate inputs for LM and TTS_LM
        # LM (Semantic) sees ONLY text. TTS_LM sees Text + Speech.
        # This explains the cache size discrepancy (108 vs 316) in official presets.
        
        # LM Input: System + Header + Prefix
        lm_tokens = system_tokens + header_tokens + prefix_tokens
        
        # TTS LM Input: System + Voice + Header + Prefix
        tts_lm_tokens = system_tokens + voice_tokens + header_tokens + prefix_tokens
        
        # Masks (aligned with tts_lm_tokens)
        # System (text) + Voice (speech) + Header (text) + Prefix (text)
        # speech_input_masks: Text=False, Speech=True
        # tts_text_masks:     Text=1,     Speech=0
        speech_input_masks = [False] * len(system_tokens) + voice_speech_masks + [False] * len(header_tokens) + [False] * len(prefix_tokens)
        tts_text_masks_list = [1] * len(system_tokens) + [0] * len(voice_tokens) + [1] * len(header_tokens) + [1] * len(prefix_tokens)

        # 4. Prepare Batch for Forward Pass
        # We need to wrap this into batch encoding format
        input_ids = torch.tensor([lm_tokens], device=device, dtype=torch.long) # For LM
        tts_lm_input_ids = torch.tensor([tts_lm_tokens], device=device, dtype=torch.long) # For TTS LM
        
        speech_input_mask_tensor = torch.tensor([speech_input_masks], device=device, dtype=torch.bool)
        tts_text_masks_tensor = torch.tensor([tts_text_masks_list], device=device, dtype=torch.long)
        
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
            
            # Ensure speech_tensors match model dtype (Half/Float16)
            if speech_tensors is not None:
                speech_tensors = speech_tensors.to(dtype=model.dtype, device=model.device)

            tts_lm_out = model.forward_tts_lm(
                input_ids=tts_lm_input_ids, # Use TTS_LM specific input (includes speech tokens)
                attention_mask=torch.ones_like(tts_lm_input_ids),
                lm_last_hidden_state=lm_last_hidden,
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_input_mask=speech_input_mask_tensor,
                tts_text_masks=tts_text_masks_tensor, # Prompt mixed mask (System/Header=1, Voice=0)
                use_cache=True,
                return_dict=True
            )
            tts_lm_cache = tts_lm_out.past_key_values
            tts_lm_last_hidden = tts_lm_out.last_hidden_state

        # 6. Negative Cache (Standard "<|image_pad|>" for unconditioned)
        neg_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # FIX: Negative Cache should be length 1 (as seen in official presets), not prompt length!
        neg_input_ids = torch.full((1, 1), neg_input_id, device=device, dtype=torch.long)
        
        with torch.no_grad():
             neg_lm_o = model.forward_lm(
                 input_ids=neg_input_ids, 
                 attention_mask=torch.ones_like(neg_input_ids), 
                 use_cache=True, 
                 return_dict=True
             )
             neg_lm_cache = neg_lm_o.past_key_values
             neg_lm_last_hidden = neg_lm_o.last_hidden_state
             
             neg_tts_lm_o = model.forward_tts_lm(
                input_ids=neg_input_ids, 
                attention_mask=torch.ones_like(neg_input_ids), 
                use_cache=True, 
                return_dict=True, 
                lm_last_hidden_state=neg_lm_o.last_hidden_state, 
                # FIX: Negative Cache (Padding) is TEXT (Type 1), not Speech (0). 
                # Marking it as Speech prevents Scatter from working (0!=1) and adds wrong Type Embedding.
                tts_text_masks=torch.ones_like(neg_input_ids) 
            )
             neg_tts_lm_cache = neg_tts_lm_o.past_key_values
             neg_tts_lm_last_hidden = neg_tts_lm_o.last_hidden_state

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
                'past_key_values': recursive_to_cpu(tts_lm_cache),
                'last_hidden_state': recursive_to_cpu(tts_lm_last_hidden)
            },
            'neg_lm': {
                'past_key_values': recursive_to_cpu(neg_lm_cache),
                'last_hidden_state': recursive_to_cpu(neg_lm_last_hidden)
            },
            'neg_tts_lm': {
                'past_key_values': recursive_to_cpu(neg_tts_lm_cache),
                'last_hidden_state': recursive_to_cpu(neg_tts_lm_last_hidden)
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
