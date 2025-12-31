import torch
import numpy as np
import os
import random
import tempfile
import soundfile as sf
import warnings

# Suppress annoying warnings
# Suppress FutureWarnings (autocast, weight_norm, etc)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress specific ONNX/Torch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider' is not in available provider names")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ONNXRUNTIME_QUIET"] = "1"
from typing import Dict, Any, Tuple, List
import sys
import subprocess
import folder_paths
from huggingface_hub import snapshot_download

# Lazy-loaded global variable
CosyVoice = None

def _install_cosyvoice_if_needed():
    global CosyVoice
    if CosyVoice is not None:
        return

    # Try import first
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice as CV
        CosyVoice = CV
        return
    except ImportError:
        pass
    
    # Only print if we are actually about to do work
    # print("[AIIA] CosyVoice invalid or missing. Attempting robust installation...")
    try:
        # Define local libs path
        libs_dir = os.path.join(os.path.dirname(__file__), "libs")
        cosyvoice_dir = os.path.join(libs_dir, "CosyVoice")
        matcha_dir = os.path.join(cosyvoice_dir, "third_party", "Matcha-TTS")

        if not os.path.exists(libs_dir):
            os.makedirs(libs_dir, exist_ok=True)

        # 1. Clone if not exists
        if not os.path.exists(cosyvoice_dir):
            print("[AIIA] Cloning CosyVoice from GitHub...")
            subprocess.check_call(["git", "clone", "--recursive", "https://github.com/FunAudioLLM/CosyVoice.git", cosyvoice_dir])
        else:
             # Verify consistency?
             pass

        # 2. Add to sys.path with PRIORITY
        # Using insert(0) ensures we use OUR cloned version, not some broken pip version
        if cosyvoice_dir not in sys.path:
            sys.path.insert(0, cosyvoice_dir)
        else:
            # Move to front
            sys.path.remove(cosyvoice_dir)
            sys.path.insert(0, cosyvoice_dir)

        if matcha_dir not in sys.path:
            sys.path.insert(0, matcha_dir)
            
        # 3. Clean sys.modules to prevent conflict with previous failed imports
        # If 'cosyvoice' was partially loaded or loaded from wrong place, clear it
        keys_to_remove = [k for k in sys.modules.keys() if k.startswith("cosyvoice") or k.startswith("matcha")]
        for k in keys_to_remove:
            del sys.modules[k]

        # 4. Install Requirements
        # Critical dependencies from requirements.txt
        reqs = ["modelscope", "hyperpyyaml", "onnxruntime", "hjson", "openai-whisper", "webrtcvad", "pydub"]
        for r in reqs:
            try:
                # Handle special import names
                if r == "openai-whisper":
                    import_name = "whisper"
                else:
                    import_name = r.replace("-", "_")
                
                __import__(import_name)
            except ImportError:
                 print(f"[AIIA] Installing missing dependency: {r}")
                 subprocess.check_call([sys.executable, "-m", "pip", "install", r], env=os.environ)

        # 5. Retry import
        # Print file existence for debugging
        expected_file = os.path.join(cosyvoice_dir, "cosyvoice", "cli", "cosyvoice.py")
        if not os.path.exists(expected_file):
             print(f"[AIIA] CRITICAL WARNING: Expected file not found at {expected_file}")

        from cosyvoice.cli.cosyvoice import CosyVoice as CV
        CosyVoice = CV

    except Exception as e:
        print(f"[AIIA] Failed to install cosyvoice via cloning: {e}")
        print("Please manually clone CosyVoice into 'ComfyUI/custom_nodes/ComfyUI_AIIA/libs/CosyVoice' and install requirements.")
 

class AIIA_CosyVoice_ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                    "FunAudioLLM/CosyVoice2-0.5B",
                    "CosyVoice-300M",
                    "CosyVoice-300M-SFT", 
                    "CosyVoice-300M-Instruct"
                ],),
                "use_fp16": ("BOOLEAN", {"default": True}),
                "use_rl_model": ("BOOLEAN", {"default": True, "tooltip": "Use llm.rl.pt (Reinforcement Learning optimized) if available. Only for V3/V2 models."}),
            }
        }

    RETURN_TYPES = ("COSYVOICE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AIIA/Loaders"

    def load_model(self, model_name, use_fp16, use_rl_model=True):
        _install_cosyvoice_if_needed()
        
        # Ensure paths are in sys.path even if installed previously (e.g. reload)
        libs_dir = os.path.join(os.path.dirname(__file__), "libs")
        cosyvoice_dir = os.path.join(libs_dir, "CosyVoice")
        matcha_dir = os.path.join(cosyvoice_dir, "third_party", "Matcha-TTS")
        
        if os.path.exists(cosyvoice_dir) and cosyvoice_dir not in sys.path:
            sys.path.append(cosyvoice_dir)
        if os.path.exists(matcha_dir) and matcha_dir not in sys.path:
            sys.path.append(matcha_dir)
            
        if CosyVoice is None:
            raise ImportError("CosyVoice package is not installed. Please install it manually.")

        # Setup paths
        base_path = folder_paths.models_dir
        cosyvoice_path = os.path.join(base_path, "cosyvoice")
        
        # Handle different naming conventions
        if "/" in model_name:
             # e.g. FunAudioLLM/Fun-CosyVoice3-0.5B-2512 -> local_dir: Fun-CosyVoice3-0.5B-2512
             local_name = model_name.split("/")[-1]
             repo_id = model_name
        else:
             # Legacy mapping for v1
             local_name = model_name
             repo_id = f"FunAudioLLM/{model_name}"

        model_dir = os.path.join(cosyvoice_path, local_name)
        
        # --- Override: Check specific server path for Fun-CosyVoice3 ---
        server_v3_path = "/app/ComfyUI/models/cosyvoice/Fun-CosyVoice3-0.5B-2512"
        if local_name == "Fun-CosyVoice3-0.5B-2512" and os.path.exists(server_v3_path):
             model_dir = server_v3_path
             print(f"[AIIA] Using server cache for CosyVoice3: {model_dir}")
        
        # Validation logic for V1/V2 vs V3
        yaml_path = os.path.join(model_dir, "cosyvoice.yaml")
        yaml3_path = os.path.join(model_dir, "cosyvoice3.yaml")
        model_pt_path = os.path.join(model_dir, "model.pt")
        
        # Check V3 specific files (flow.pt, llm.pt, hift.pt)
        is_v3 = False
        if os.path.exists(yaml3_path) or os.path.exists(os.path.join(model_dir, "flow.pt")):
            is_v3 = True
            
        has_valid_model_file = os.path.exists(model_pt_path) or (os.path.exists(os.path.join(model_dir, "flow.pt")) and os.path.exists(os.path.join(model_dir, "llm.pt")))
        has_valid_config = os.path.exists(yaml_path) or os.path.exists(yaml3_path)

        # Download if missing or incomplete
        if "ttsfrd" in model_name.lower() and not has_valid_config:
            raise ValueError(f"'{model_name}' is a system dependency package, not a loadable CosyVoice model. Please select a valid model (e.g., Fun-CosyVoice3-0.5B-2512).")

        if not os.path.exists(model_dir) or not has_valid_config or not has_valid_model_file:
            print(f"[AIIA] Model missing or incomplete. Downloading {model_name} to {model_dir}...")
            try:
                snapshot_download(repo_id=repo_id, local_dir=model_dir)
            except Exception as e:
                print(f"Failed to download from HF: {e}. Checking if ModelScope works...")
                raise e
        else:
             print(f"[AIIA] Model verified at {model_dir}")
             
        # --- RL Model Switching Logic (Strict Symlink Strategy) ---
        llm_pt = os.path.join(model_dir, "llm.pt")
        llm_rl = os.path.join(model_dir, "llm.rl.pt")
        llm_base = os.path.join(model_dir, "llm.base.pt")

        # 1. Ensure we have a persistent llm.base.pt backup if llm.pt is a real file
        if os.path.exists(llm_pt) and not os.path.islink(llm_pt):
             if not os.path.exists(llm_base):
                print(f"[AIIA] Initializing: Renaming original llm.pt to llm.base.pt")
                os.rename(llm_pt, llm_base)
             else:
                # If both exist and llm.pt is real, something is messy. Delete llm.pt to make room for link
                os.remove(llm_pt)

        # 2. Determine target based on use_rl_model
        # Use an explicit truthy check to handle string inputs if ComfyUI sends them as such
        is_rl_requested = str(use_rl_model).lower() == 'true' if not isinstance(use_rl_model, bool) else use_rl_model
        
        print(f"[AIIA] Loader Debug: use_rl_model_raw={use_rl_model} (type: {type(use_rl_model)}) -> effective_rl={is_rl_requested}")
        print(f"[AIIA] Loader Debug: llm_rl exists: {os.path.exists(llm_rl)}")
        
        target_source = llm_rl if is_rl_requested and os.path.exists(llm_rl) else llm_base
        
        # 3. Aggressively Manage Symlink
        if os.path.exists(target_source):
            target_basename = os.path.basename(target_source)
            current_target = None
            
            if os.path.exists(llm_pt):
                if os.path.islink(llm_pt):
                    current_target = os.readlink(llm_pt)
                else:
                    os.remove(llm_pt) # Real file blocking link
            
            if current_target != target_basename:
                print(f"[AIIA] Model Switching: Linking llm.pt -> {target_basename}")
                if os.path.exists(llm_pt): os.remove(llm_pt)
                try:
                    os.symlink(target_basename, llm_pt)
                except Exception:
                    # Fallback to copy if symlink fails
                    import shutil
                    shutil.copy2(target_source, llm_pt)
            else:
                print(f"[AIIA] Model Status: llm.pt already linked to {target_basename}")
        # --------------------------------------------------------

        # Load Model
        print(f"Loading CosyVoice model from {model_dir}...")
        
        try:
            # Import specific classes from the CLI module since we need V3 support
            # Use aliases to avoid UnboundLocalError shadowing the global 'CosyVoice' variable
            from cosyvoice.cli.cosyvoice import CosyVoice as CV1, CosyVoice2 as CV2, CosyVoice3 as CV3
            
            # Smart loading logic
            is_v3 = False
            is_v2 = False
            
            if os.path.exists(os.path.join(model_dir, "cosyvoice3.yaml")):
                print(f"[AIIA] Detected V3 Model. Using CosyVoice3 class (fp16={use_fp16}).")
                model_instance = CV3(model_dir, fp16=use_fp16)
                is_v3 = True
            elif os.path.exists(os.path.join(model_dir, "cosyvoice2.yaml")) or os.path.exists(os.path.join(model_dir, "flow.pt")):
                print(f"[AIIA] Detected V2 Model. Using CosyVoice2 class (fp16={use_fp16}).")
                model_instance = CV2(model_dir, fp16=use_fp16)
                is_v2 = True
            else:
                print(f"[AIIA] Detected V1 Model (or default). Using CosyVoice class (fp16={use_fp16}).")
                # V1 CosyVoice might not support fp16 arg in all versions, but typical implementations do or ignore it.
                # Checking source is hard securely, but we aim for V3 support mainly.
                # If V1 signature differs, we might need try-except.
                # Official CosyVoice V1 init: (model_dir). No fp16 arg in old versions?
                # Let's try passing it. If TypeError, fallback.
                try:
                    model_instance = CV1(model_dir, fp16=use_fp16)
                except TypeError:
                     print("[AIIA] V1 Class rejected fp16 argument. Initializing default.")
                     model_instance = CV1(model_dir)

            # --- CRITICAL: Clear Matcha-TTS Internal Caches ---
            # This prevents STFT window size mismatch errors when switching between 22k and 24k models.
            try:
                import matcha.utils.audio as matcha_audio
                if hasattr(matcha_audio, "mel_basis"): matcha_audio.mel_basis.clear()
                if hasattr(matcha_audio, "hann_window"): matcha_audio.hann_window.clear()
                print("[AIIA] Cleared Matcha-TTS global audio caches for multi-model stability.")
            except Exception as e:
                print(f"[AIIA] Warning: Could not clear Matcha cache: {e}")
                
        except ImportError:
             # Fallback if library is old
             print("[AIIA] Warning: Advanced CosyVoice classes not found. Fallback to default global class.")
             model_instance = CosyVoice(model_dir)
        except Exception as e:
           raise RuntimeError(f"Failed to initialize CosyVoice model: {e}")
           
        # IMPORTANT: The TTS node expects version flags.
        return ({"model": model_instance, "model_dir": model_dir, "is_v3": is_v3, "is_v2": is_v2},)

class AIIA_CosyVoice_VoiceConversion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "source_audio": ("AUDIO",),
                "target_audio": ("AUDIO",),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "chunk_size": ("INT", {"default": 25, "min": 10, "max": 28, "step": 1, "tooltip": "建议保持在25秒左右"}),
                "overlap_size": ("INT", {"default": 1, "min": 0, "max": 4, "step": 1, "tooltip": "重叠部分也会计入30秒限制"}),
            },
            "optional": {
                "whisper_chunks": ("WHISPER_CHUNKS",),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
            }
        }

    RETURN_TYPES = ("AUDIO", "SPLICE_INFO")
    RETURN_NAMES = ("audio", "splice_info")
    FUNCTION = "convert_voice_unlimited"
    CATEGORY = "AIIA/Synthesis"

    def _find_best_split_point(self, waveform, target_idx, search_range_samples):
        start = max(0, target_idx - search_range_samples)
        end = min(waveform.shape[-1], target_idx + search_range_samples)
        if start >= end: return target_idx
        search_region = waveform[:, start:end]
        energy = torch.abs(search_region).mean(dim=0)
        window_size = 100
        if energy.shape[0] > window_size:
            energy = torch.conv1d(energy.unsqueeze(0).unsqueeze(0), 
                                 torch.ones(1, 1, window_size, device=energy.device) / window_size,
                                 padding=window_size//2).squeeze()
        min_idx = torch.argmin(energy).item()
        return start + min_idx

    def convert_voice_unlimited(self, model, source_audio, target_audio, speed, chunk_size, overlap_size, whisper_chunks=None, seed=42):
        cosyvoice_model = model["model"]
        sample_rate = cosyvoice_model.sample_rate
        
        target_waveform = target_audio["waveform"]
        if target_audio["sample_rate"] != sample_rate:
            import torchaudio
            target_waveform = torchaudio.transforms.Resample(target_audio["sample_rate"], sample_rate)(target_waveform)
        
        max_target_samples = 30 * sample_rate
        if target_waveform.shape[-1] > max_target_samples:
            target_waveform = target_waveform[..., :max_target_samples]
        
        # Normalize and Save Target
        if target_waveform.abs().max() > 1.0:
            target_waveform = target_waveform / target_waveform.abs().max()
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_target:
            target_path = tmp_target.name
            target_np = target_waveform.squeeze().cpu().numpy()
            if target_np.ndim == 2: target_np = target_np.T
            sf.write(target_path, target_np, sample_rate, subtype='FLOAT')

        source_waveform = source_audio["waveform"]
        if source_audio["sample_rate"] != sample_rate:
            import torchaudio
            source_waveform = torchaudio.transforms.Resample(source_audio["sample_rate"], sample_rate)(source_waveform)
        
        # Normalize Source
        if source_waveform.abs().max() > 1.0:
            source_waveform = source_waveform / source_waveform.abs().max()

        source_waveform = source_waveform.squeeze() 
        if source_waveform.ndim == 1: source_waveform = source_waveform.unsqueeze(0)
        total_samples = source_waveform.shape[-1]
        
        MAX_TOTAL_SEC = 29.8
        chunk_samples = chunk_size * sample_rate
        overlap_samples = overlap_size * sample_rate
        max_total_samples = int(MAX_TOTAL_SEC * sample_rate)
        
        # 记录切片信息用于调试
        splice_info = {
            "sample_rate": sample_rate,
            "total_samples": total_samples,
            "splice_points": []
        }

        if total_samples <= max_total_samples:
            result_waveform = self._inference_single_chunk(cosyvoice_model, source_waveform, target_path, speed, sample_rate, seed)
            if os.path.exists(target_path): os.unlink(target_path)
            return ({"waveform": result_waveform.unsqueeze(0), "sample_rate": sample_rate}, splice_info)

        # 3. 智能分块逻辑
        chunks_to_process = []
        is_pre_chunked = whisper_chunks and any(c.get("speaker") == "AIIA_SMART_CHUNK" for c in whisper_chunks.get("chunks", []))
        
        if is_pre_chunked:
            print(f"[AIIA CosyVoice] Detected Smart Chunker input. Following pre-defined segments strictly.")
            for c in whisper_chunks["chunks"]:
                s_time, e_time = c["timestamp"]
                s_idx = int(s_time * sample_rate)
                e_idx = int(e_time * sample_rate)
                e_idx_with_overlap = min(e_idx + overlap_samples, total_samples)
                chunks_to_process.append(source_waveform[:, s_idx:e_idx_with_overlap])
        else:
            current_start = 0
            search_range = 2 * sample_rate
            while current_start < total_samples:
                target_end_time = (current_start + chunk_samples) / sample_rate
                best_split_time = target_end_time
                if whisper_chunks and "chunks" in whisper_chunks:
                    closest_gap_dist = float('inf')
                    for i in range(len(whisper_chunks["chunks"]) - 1):
                        gap_mid = (whisper_chunks["chunks"][i]["timestamp"][1] + whisper_chunks["chunks"][i+1]["timestamp"][0]) / 2
                        if (gap_mid * sample_rate + overlap_samples) - current_start > max_total_samples: continue
                        dist = abs(gap_mid - target_end_time)
                        if dist < 5 and dist < closest_gap_dist:
                            closest_gap_dist = dist
                            best_split_time = gap_mid
                split_point = int(best_split_time * sample_rate)
                split_point = self._find_best_split_point(source_waveform, split_point, search_range)
                split_point = min(split_point, total_samples)
                if (split_point + overlap_samples) - current_start > max_total_samples:
                    split_point = current_start + max_total_samples - overlap_samples - 100
                actual_end = min(split_point + overlap_samples, total_samples)
                if actual_end >= total_samples:
                    chunks_to_process.append(source_waveform[:, current_start:])
                    break
                else:
                    chunks_to_process.append(source_waveform[:, current_start:actual_end])
                current_start = split_point
                if total_samples - current_start < sample_rate:
                    break

        # 4. 拼接逻辑 (Sacrificial Context Cross-fade)
        final_segments = []
        MAX_CROSS_FADE_DURATION = 0.05 # 50ms 强制最大淡入淡出
        max_xfade_samples = int(MAX_CROSS_FADE_DURATION * sample_rate)

        for i, chunk in enumerate(chunks_to_process):
            print(f"[AIIA CosyVoice] Processing chunk {i+1}/{len(chunks_to_process)}, actual input len: {chunk.shape[-1]/sample_rate:.2f}s")
            converted_chunk = self._inference_single_chunk(cosyvoice_model, chunk, target_path, speed, sample_rate, seed)
            
            if not final_segments:
                final_segments.append(converted_chunk)
            else:
                prev_chunk = final_segments[-1]
                # 计算重叠
                # overlap_samples 是输入时的重叠，对应输出需要除以 speed (虽然 CosyVoice 不一定会严格且线性地遵循 speed，但这是一个估算)
                estimated_overlap_samples = int(overlap_samples / speed)
                
                # 我们只使用仅仅够消除爆音的短 crossfade，剩下的重叠部分作为“牺牲上下文”被丢弃
                xfade_len = min(estimated_overlap_samples, max_xfade_samples)
                sacrificial_len = 0
                
                if estimated_overlap_samples > xfade_len:
                    sacrificial_len = estimated_overlap_samples - xfade_len

                # 确保 tensor 够长
                if prev_chunk.shape[-1] > (sacrificial_len + xfade_len):
                     # 1. 裁剪前一段的尾部 (Sacrificial Context)
                    trim_idx = prev_chunk.shape[-1] - sacrificial_len
                    prev_chunk_trimmed = prev_chunk[:, :trim_idx]
                    
                    # 2. 准备 Cross-fade
                    t = torch.linspace(0, np.pi, xfade_len, device=converted_chunk.device)
                    fade_out = 0.5 * (1.0 + torch.cos(t))
                    fade_in = 1.0 - fade_out
                    
                    prev_tail = prev_chunk_trimmed[:, -xfade_len:]
                    curr_head = converted_chunk[:, :xfade_len]
                    
                    # 只有当维度匹配时才做 xfade，否则直接拼接
                    if prev_tail.shape[-1] == xfade_len and curr_head.shape[-1] == xfade_len:
                        overlap_part = prev_tail * fade_out + curr_head * fade_in
                        
                        final_segments[-1] = prev_chunk_trimmed[:, :-xfade_len] # 移除 xfade 部分
                        final_segments.append(overlap_part)
                        final_segments.append(converted_chunk[:, xfade_len:])   # 添加 xfade 之后的部分
                        
                        # 记录拼接点 (相对于最终音频的采样点)
                        current_total_len = sum([seg.shape[-1] for seg in final_segments]) - converted_chunk.shape[-1] + xfade_len # 指向 overlap 中心
                        splice_info["splice_points"].append(current_total_len)

                    else:
                         # 维度不够，直接硬拼接
                        final_segments.append(converted_chunk)
                else:
                    final_segments.append(converted_chunk)
            
            torch.cuda.empty_cache()

        merged_waveform = torch.cat(final_segments, dim=-1)
        if os.path.exists(target_path): os.unlink(target_path)
        return ({"waveform": merged_waveform.unsqueeze(0).cpu(), "sample_rate": sample_rate}, splice_info)

    def _inference_single_chunk(self, model, waveform, target_path, speed, sample_rate, seed):
        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_source:
            source_path = tmp_source.name
            source_np = waveform.cpu().numpy()
            if source_np.ndim == 2: source_np = source_np.T
            sf.write(source_path, source_np, sample_rate, subtype='FLOAT')
        try:
            output = model.inference_vc(source_wav=source_path, prompt_wav=target_path, stream=False, speed=speed)
            all_speech = [chunk['tts_speech'] for chunk in output]
            return torch.cat(all_speech, dim=-1)
        finally:
            if os.path.exists(source_path): os.unlink(source_path)

class AIIA_CosyVoice_TTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("COSYVOICE_MODEL",),
                "提示1_说的内容": ("STRING", {"default": "📖 第一步：在此输入您想让 AI 说的话 (TTS Text)", "is_label": True}),
                "tts_text": ("STRING", {"multiline": True, "default": "你好，这是 CosyVoice 3.0 的全能模式测试。"}),
                "提示2_音色描述": ("STRING", {"default": "🎨 第二步：在此输入对表现力/情感的文字描述 (Style Description)", "is_label": True}),
                "instruct_text": ("STRING", {"multiline": True, "default": "语速非常慢，语气充满磁性，情感饱满。", "tooltip": "文字描述：在 0.5B 中主要控制情感、方言、语速等‘表现风格’，而非从零生成音色身份。"}),
                "base_gender": (["Female", "Male"], {"default": "Female", "tooltip": "基础性别底色。在“描述生成”模式下，这提供初始的声音身份（性别/音感底色）。"}),
                "dialect": (["None (Auto)", "广东话 (Cantonese)", "东北话 (Northeastern)", "四川话 (Sichuan)", "河南话 (Henan)", "天津话 (Tianjin)", "上海话 (Shanghai)", "山东话 (Shandong)", "湖北话 (Hubei)", "湖南话 (Hunan)", "陕西话 (Shaanxi)", "山西话 (Shanxi)", "甘肃话 (Gansu)", "宁夏话 (Ningxia)", "闽南话 (Hokkien)", "贵州话 (Guizhou)", "云南话 (Yunnan)", "江西话 (Jiangxi)"], {"default": "None (Auto)", "tooltip": "预设方言指令。会自动添加在自定义描述之前。若与自定义文字描述冲突，模型表现将不可预测。"}),
                "emotion": (["None (Neutral)", "开心 (Happy)", "伤心 (Sad)", "生气 (Angry)", "机器人的方式 (Robotic)", "小猪佩奇风格 (Peppa Pig)"], {"default": "None (Neutral)", "tooltip": "预设情感指令。会自动添加在自定义描述之前。"}),
                "spk_id": ("STRING", {"default": "", "tooltip": "固定音色 ID (如 pure_1)。对于 0.5B/V3 等 Zero-Shot 模型，此项通常为空，需配合参考音频使用。"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 2147483647}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/Synthesis"

    def generate(self, model, tts_text, instruct_text, spk_id, speed, seed, dialect="None", emotion="None", reference_audio=None, base_gender="Female", **kwargs):
        cosyvoice_model = model["model"]
        sample_rate = cosyvoice_model.sample_rate

        if seed >= 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        final_waveform = None

        try:
            # Detect Model Type from dictionary if available, else class name
            is_v3 = model.get("is_v3", "CosyVoice3" in type(cosyvoice_model).__name__)
            is_v2 = model.get("is_v2", "CosyVoice2" in type(cosyvoice_model).__name__)
            model_dir = model.get("model_dir", getattr(cosyvoice_model, 'model_dir', ''))
            
            # --- Model Version Verification ---
            llm_pt = os.path.join(model_dir, "llm.pt") if model_dir else None
            active_model = "Unknown"
            if llm_pt and os.path.exists(llm_pt) and os.path.islink(llm_pt):
                active_model = os.readlink(llm_pt)
            print(f"[AIIA] CosyVoice Active LLM: {active_model}")
            
            if "base" in active_model.lower() and instruct_text:
                print("\033[93m" + "[AIIA] WARNING: You are using the BASE model with instructions." + "\033[0m")
                print("\033[93m" + "[AIIA] It will LIKELY read your instructions aloud. Switch to RL model in Loader!" + "\033[0m")
        
        # --- Speaker Identity Validation & Fallback ---
            available_spks = list(cosyvoice_model.frontend.spk2info.keys())
            use_seed_fallback = False
            
            if spk_id:
                if spk_id not in available_spks:
                    raise ValueError(f"Speaker ID '{spk_id}' not found. Available: {available_spks if available_spks else 'None (Zero-Shot model)'}")
            elif reference_audio is None:
                if available_spks:
                    # Improve auto-selection based on base_gender
                    gender_keyword = "男" if base_gender == "Male" else "女"
                    # Try to find a speaker that matches the requested gender (Chinese/English keywords)
                    matching_spks = [s for s in available_spks if gender_keyword in s or base_gender.lower() in s.lower()]
                    
                    if matching_spks:
                        spk_id = matching_spks[0]
                    else:
                        spk_id = available_spks[0]
                    
                    print(f"[AIIA] Auto-selecting speaker based on gender({base_gender}): {spk_id}")
                else:
                    # 0.5B models typically have no spk2info. Use internal neutral seed for "Pure Instruct"
                    use_seed_fallback = True
                    print("[AIIA] No Identity provided for Zero-Shot model. Falling back to internal 'Neutral Canvas' seed.")

            # 1. Hybrid / Cross-Lingual / Zero-Shot / Pure Instruct (Audio-driven Logic)
            if reference_audio is not None or use_seed_fallback:
                if reference_audio is not None:
                    print(f"[AIIA] CosyVoice: Audio provided ({'Hybrid' if instruct_text else 'Zero-Shot'}).")
                    ref_wav = reference_audio["waveform"]
                    if reference_audio["sample_rate"] != sample_rate:
                        import torchaudio
                        ref_wav = torchaudio.transforms.Resample(reference_audio["sample_rate"], sample_rate)(ref_wav)
                    
                    if ref_wav.abs().max() > 1.0: ref_wav = ref_wav / ref_wav.abs().max()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
                        ref_path = tmp_ref.name
                        ref_np = ref_wav.squeeze().cpu().numpy()
                        if ref_np.ndim == 2: ref_np = ref_np.T
                        sf.write(ref_path, ref_np, sample_rate, subtype='FLOAT')
                        cleanup_ref = True
                else:
                    # Pure Instruct Fallback: Use built-in asset based on selection
                    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
                    raw_seed_path = os.path.join(assets_dir, "seed_male.wav" if base_gender == "Male" else "seed_female.wav")
                    
                    if not os.path.exists(raw_seed_path):
                        # Fallback for server structure or library default
                        raw_seed_path = os.path.join(os.path.dirname(__file__), "libs", "CosyVoice", "asset", "zero_shot_prompt.wav")
                    
                    # --- CRITICAL: Always Resample Seed to Model Sample Rate ---
                    # Built-in seeds might be 22050Hz, but V2/V3 models expect 24000Hz.
                    # Mismatch here can cause the "STFT window length" RuntimeError.
                    import torchaudio
                    seed_wav, seed_sr = torchaudio.load(raw_seed_path)
                    if seed_sr != sample_rate:
                        print(f"[AIIA] Resampling seed audio from {seed_sr}Hz to {sample_rate}Hz...")
                        seed_wav = torchaudio.transforms.Resample(seed_sr, sample_rate)(seed_wav)
                    
                    if seed_wav.abs().max() > 1.0: seed_wav = seed_wav / seed_wav.abs().max()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_seed:
                        ref_path = tmp_seed.name
                        seed_np = seed_wav.squeeze().cpu().numpy()
                        if seed_np.ndim == 2: seed_np = seed_np.T
                        sf.write(ref_path, seed_np, sample_rate, subtype='FLOAT')
                        cleanup_ref = True
                        
                    print(f"[AIIA] CosyVoice: Pure Instruct Mode using {base_gender} seed ({sample_rate}Hz).")

                try:
                    if not os.path.exists(ref_path):
                        raise FileNotFoundError(f"Internal seed audio not found at {ref_path}. Please check installation.")

                    if is_v3 or is_v2:
                        print(f"[AIIA] CosyVoice V3/V2 Core: Multi-modal Inference.")
                        
                        # --- Preset Assembly (Fusion Logic) ---
                        preset_instructs = []
                        dialect_core = dialect.split(' ')[0] if dialect != "None (Auto)" else None
                        emotion_core = emotion.split(' ')[0] if emotion != "None (Neutral)" else None
                        
                        if dialect_core and emotion_core:
                            # Combined case: Merge into one sentence for better LLM attention
                            if "机器人" in emotion_core:
                                preset_instructs.append(f"请用{dialect_core}，并且尝试用机器人的方式解答。")
                            elif "小猪佩奇" in emotion_core:
                                preset_instructs.append(f"请用{dialect_core}，并且我想体验一下小猪佩奇风格。")
                            else:
                                preset_instructs.append(f"请用{dialect_core}，并且非常{emotion_core}地说一句话。")
                        elif dialect_core:
                            preset_instructs.append(f"请用{dialect_core}表达。")
                        elif emotion_core:
                            if "机器人" in emotion_core:
                                preset_instructs.append("你可以尝试用机器人的方式解答吗？")
                            elif "小猪佩奇" in emotion_core:
                                preset_instructs.append("我想体验一下小猪佩奇风格，可以吗？")
                            else:
                                preset_instructs.append(f"请非常{emotion_core}地说一句话。")
                                
                        combined_custom = " ".join(preset_instructs)
                        if instruct_text:
                            combined_custom = f"{combined_custom} {instruct_text}".strip()
                        
                        modified_instruct = combined_custom
                        if combined_custom and "<|endofprompt|>" not in combined_custom:
                            modified_instruct = f"You are a helpful assistant. {combined_custom}<|endofprompt|>"
                            print(f"[AIIA] Applied V3 Instruction Formatting: {modified_instruct[:80]}...")
                        elif combined_custom:
                            print(f"[AIIA] Using Raw Instruction: {combined_custom[:50]}...")
                        
                        output = cosyvoice_model.inference_instruct2(
                            tts_text=tts_text, 
                            instruct_text=modified_instruct, 
                            prompt_wav=ref_path, 
                            zero_shot_spk_id=spk_id, 
                            stream=False, 
                            speed=speed
                        )
                    else:
                        # V1 path: In V1, we cannot easily combine ref_audio + custom instructions in one CLI call.
                        # Always use zero_shot if we have a reference audio (including seed).
                        # For V1, prompt_text is REQUIRED for zero-shot to work correctly.
                        # If using the seed fallback, provide a default transcript.
                        p_text = "希望你以后能够做的比我还好呦。" if use_seed_fallback else ""
                        output = cosyvoice_model.inference_zero_shot(tts_text=tts_text, prompt_text=p_text, prompt_wav=ref_path, stream=False, speed=speed)
                    
                    all_speech = [chunk['tts_speech'] for chunk in output]
                    final_waveform = torch.cat(all_speech, dim=-1)
                finally:
                    if cleanup_ref and os.path.exists(ref_path): os.unlink(ref_path)

            # 2. SFT / Instruct (Fixed Speaker ID, No Reference Audio)
            else:
                print(f"[AIIA] CosyVoice: Fixed Identity Mode (SFT/ID). Speaker: {spk_id}")
                if is_v3 or is_v2:
                    output = cosyvoice_model.inference_instruct2(
                        tts_text=tts_text, 
                        instruct_text=instruct_text, 
                        prompt_wav=None, 
                        zero_shot_spk_id=spk_id, 
                        stream=False, 
                        speed=speed
                    )
                else:
                    # For V1, we must have a valid speaker ID. 
                    # If empty, default to 'pure_1' which exists in most official 300M checkpoints.
                    effective_spk = spk_id if (spk_id and spk_id.strip()) else "pure_1"
                    if "SFT" in type(cosyvoice_model).__name__ or not instruct_text:
                        output = cosyvoice_model.inference_sft(tts_text, effective_spk, stream=False, speed=speed)
                    else:
                        output = cosyvoice_model.inference_instruct(tts_text, effective_spk, instruct_text, stream=False, speed=speed)
                
                all_speech = [chunk['tts_speech'] for chunk in output]
                final_waveform = torch.cat(all_speech, dim=-1)

        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)): raise e
            raise RuntimeError(f"CosyVoice generation failed: {e}")

        return ({"waveform": final_waveform.unsqueeze(0).cpu(), "sample_rate": sample_rate},)

        return ({"waveform": final_waveform.unsqueeze(0).cpu(), "sample_rate": sample_rate},)

NODE_CLASS_MAPPINGS = {
    "AIIA_CosyVoice_ModelLoader": AIIA_CosyVoice_ModelLoader,
    "AIIA_CosyVoice_VoiceConversion": AIIA_CosyVoice_VoiceConversion,
    "AIIA_CosyVoice_TTS": AIIA_CosyVoice_TTS
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_CosyVoice_ModelLoader": "CosyVoice Model Loader (AIIA)",
    "AIIA_CosyVoice_VoiceConversion": "Voice Conversion (AIIA Unlimited)",
    "AIIA_CosyVoice_TTS": "CosyVoice 3.0 TTS (AIIA)"
}
