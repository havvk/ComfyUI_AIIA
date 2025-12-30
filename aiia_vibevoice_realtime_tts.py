
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
                 do_sample, temperature, top_k, top_p, cfg_scale, voice_preset_input=None): # cfg_scale added based on standard
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
        
        try:
            with torch.no_grad():
                print(f"[AIIA] Using 0.5B Streaming Inference with preset: {voice_preset_name}")
                
                # Load Preset
                # Preset path already resolved
                if not os.path.exists(preset_path):
                    raise FileNotFoundError(f"Preset file not found at: {preset_path}")

                preset_data = torch.load(preset_path, map_location=device)
                
                # Helper to get attributes safe and move to device
                def get_tensor(obj, key):
                     val = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
                     if isinstance(val, torch.Tensor): return val.to(device)
                     return val
                
                # Extract Caches
                lm_cache = get_tensor(preset_data.get('lm'), 'past_key_values')
                tts_lm_cache = get_tensor(preset_data.get('tts_lm'), 'past_key_values')
                lm_last_hidden = get_tensor(preset_data.get('lm'), 'last_hidden_state')
                
                if lm_cache is None or tts_lm_cache is None:
                    raise ValueError(f"Invalid preset file: {voice_preset}. Missing cache data.")

                # Extract Negative Caches (if present) or Create Dummy
                if 'neg_lm' in preset_data:
                     neg_lm_cache = get_tensor(preset_data.get('neg_lm'), 'past_key_values')
                     neg_tts_lm_cache = get_tensor(preset_data.get('neg_tts_lm'), 'past_key_values')
                else:
                     print("[AIIA] Preset missing negative cache, generating on fly...")
                     seq_len = lm_cache[0][0].shape[2]
                     neg_input_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
                     neg_input_ids = torch.full((1, seq_len), neg_input_id, device=device)
                     
                     neg_lm_o = model.forward_lm(input_ids=neg_input_ids, attention_mask=torch.ones_like(neg_input_ids), use_cache=True, return_dict=True)
                     neg_lm_cache = neg_lm_o.past_key_values
                     
                     neg_tts_lm_o = model.forward_tts_lm(
                        input_ids=neg_input_ids, 
                        attention_mask=torch.ones_like(neg_input_ids), 
                        use_cache=True, 
                        return_dict=True, 
                        lm_last_hidden_state=neg_lm_o.last_hidden_state, 
                        tts_text_masks=torch.ones_like(neg_input_ids)
                    )
                     neg_tts_lm_cache = neg_tts_lm_o.past_key_values

                # Wrap in SimpleNamespace
                from types import SimpleNamespace
                all_prefilled = {
                    "lm": SimpleNamespace(past_key_values=lm_cache, last_hidden_state=lm_last_hidden),
                    "tts_lm": SimpleNamespace(past_key_values=tts_lm_cache),
                    "neg_lm": SimpleNamespace(past_key_values=neg_lm_cache),
                    "neg_tts_lm": SimpleNamespace(past_key_values=neg_tts_lm_cache)
                }

                # Target tokens
                target_text = text # Use text as target
                target_tokens = tokenizer.encode(target_text.strip() + "\n", add_special_tokens=False, return_tensors="pt").to(device)
                
                # Dummy inputs
                cache_len = lm_cache[0][0].shape[2]
                dummy_ids = torch.zeros((1, cache_len), dtype=torch.long, device=device)
                
                # Resolution for sampling
                f_do_sample = getattr(model.generation_config, "do_sample", False) if do_sample == "auto" else (do_sample == "false")
                
                output = model.generate(
                    all_prefilled_outputs=all_prefilled,
                    tts_text_ids=target_tokens,
                    tts_lm_input_ids=dummy_ids,
                    input_ids=dummy_ids,
                    max_new_tokens=4000,
                    cfg_scale=cfg_scale,
                    do_sample=f_do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    expected_steps=len(text)*20,
                    max_length_times=10.0,
                    show_progress_bar=True
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
