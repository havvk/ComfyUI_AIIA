import torch
import os
import json
import tempfile
import soundfile as sf
from omegaconf import OmegaConf, open_dict # open_dict 允许添加新键
import folder_paths # ComfyUI 的路径管理模块

# --- 全局模型路径定义 ---
_NEMO_MODELS_SUBDIR_STR = "nemo_models" # 在 ComfyUI/models/ 下
_E2E_MODEL_FILENAME_MAP = { # 支持多种E2E模型
    "diar_sortformer_4spk-v1": "diar_sortformer_4spk-v1.nemo",
    "diar_streaming_sortformer_4spk-v2.1": "diar_streaming_sortformer_4spk-v2.1.nemo",
    # "another_e2e_model": "another_e2e_model.nemo", # 未来可以扩展
}

# 路径初始化逻辑 (只在模块加载时执行一次)
_NEMO_E2E_MODEL_PATHS = {}
_NEMO_E2E_PATHS_INIT_LOG = [] # 用于收集重要日志
_NEMO_E2E_PATH_INIT_SUCCESS = True # 初始假设成功

try:
    comfyui_root_dir = folder_paths.base_path
    if not (comfyui_root_dir and isinstance(comfyui_root_dir, str) and os.path.isdir(comfyui_root_dir)):
        _NEMO_E2E_PATHS_INIT_LOG.append(f"严重错误: [AIIA E2E Diarizer] ComfyUI 根目录 '{comfyui_root_dir}' 无效。")
        _NEMO_E2E_PATH_INIT_SUCCESS = False
    
    if _NEMO_E2E_PATH_INIT_SUCCESS: # 仅当上一步成功时继续
        comfyui_main_models_dir = os.path.join(comfyui_root_dir, "models")
        if not os.path.isdir(comfyui_main_models_dir):
            models_dir_attr = getattr(folder_paths, 'models_dir', None)
            if models_dir_attr and isinstance(models_dir_attr, str) and os.path.isdir(models_dir_attr):
                comfyui_main_models_dir = models_dir_attr
            else:
                _NEMO_E2E_PATHS_INIT_LOG.append(f"严重错误: [AIIA E2E Diarizer] 主 'models' 目录 '{comfyui_main_models_dir}' 未找到，且 folder_paths.models_dir 也无效。")
                _NEMO_E2E_PATH_INIT_SUCCESS = False
    
    if _NEMO_E2E_PATH_INIT_SUCCESS: # 仅当上一步成功时继续
        nemo_models_full_path = os.path.join(comfyui_main_models_dir, _NEMO_MODELS_SUBDIR_STR)
        if not os.path.isdir(nemo_models_full_path):
            _NEMO_E2E_PATHS_INIT_LOG.append(f"警告: [AIIA E2E Diarizer] NeMo 模型子目录 '{nemo_models_full_path}' 未找到。部分模型可能不可用。")
            # 即使子目录不存在，也可能只是该子目录的问题，不立即将整体初始化设为失败
        
        found_any_model = False
        for model_key, model_filename in _E2E_MODEL_FILENAME_MAP.items():
            # 如果 nemo_models_full_path 不存在，os.path.join 仍然会构造路径，但 isfile 会失败
            model_file_path = os.path.join(nemo_models_full_path, model_filename) 
            if os.path.isfile(model_file_path):
                _NEMO_E2E_MODEL_PATHS[model_key] = model_file_path
                _NEMO_E2E_PATHS_INIT_LOG.append(f"信息: [AIIA E2E Diarizer] 找到模型 '{model_key}': {model_file_path}")
                found_any_model = True
            else:
                _NEMO_E2E_PATHS_INIT_LOG.append(f"错误: [AIIA E2E Diarizer] 模型 '{model_key}' 的文件 '{model_filename}' 未在 '{nemo_models_full_path}' 找到。")
        
        if not found_any_model:
            _NEMO_E2E_PATHS_INIT_LOG.append(f"警告: [AIIA E2E Diarizer] 未能定位到任何配置的 E2E NeMo 模型文件。节点可能无法选择模型。")
            _NEMO_E2E_PATH_INIT_SUCCESS = False # 如果一个模型都没找到，则路径初始化视为不完全成功

except Exception as e:
    _NEMO_E2E_PATH_INIT_SUCCESS = False

class AIIA_E2E_Speaker_Diarization:

    @classmethod
    def INPUT_TYPES(cls):
        available_models = list(_NEMO_E2E_MODEL_PATHS.keys()) 
        if not available_models:
            available_models = ["NO_MODELS_FOUND"]
            
        return {
            "required": {
                "whisper_chunks": ("WHISPER_CHUNKS",),
                "audio": ("AUDIO",),
                "backend_model": (available_models, {"default": available_models[0]}),
                "num_speakers": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1, 
                                         "tooltip": "期望说话人数 (0=自动, 最多4人)。Sortformer模型上限4人。"}),
            },
            "optional": {
                 "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("WHISPER_CHUNKS",)
    RETURN_NAMES = ("diarized_whisper_chunks",)
    FUNCTION = "execute_diarization"
    CATEGORY = "AIIA/audio"

    def _get_model_path(self, backend_model_key: str):
        model_path = _NEMO_E2E_MODEL_PATHS.get(backend_model_key)
        if not model_path: 
            print(f"错误: [AIIA E2E Diarizer] E2E 模型 '{backend_model_key}' 的路径在运行时未找到。")
            return None
        return model_path

    def _format_speaker_label(self, raw_speaker_id: str) -> str:
        final_numeric_id = -1
        if raw_speaker_id.startswith("speaker_"):
            try: final_numeric_id = int(raw_speaker_id.split('_')[-1])
            except: pass
        elif raw_speaker_id.isdigit():
            try: final_numeric_id = int(raw_speaker_id)
            except: pass
        elif raw_speaker_id.startswith("SPEAKER_") and raw_speaker_id.split('_')[-1].isdigit():
                try: final_numeric_id = int(raw_speaker_id.split('_')[-1])
                except: pass
        if final_numeric_id != -1:
            return f"SPEAKER_{final_numeric_id:02d}"
        return f"SPEAKER_{raw_speaker_id}" if not raw_speaker_id.startswith("SPEAKER_") else raw_speaker_id

    def _assign_speakers_to_chunks(self, whisper_chunks_data: dict, speaker_segments: list) -> dict:
        processed_chunks = []
        if not isinstance(whisper_chunks_data, dict) or not isinstance(whisper_chunks_data.get("chunks"), list):
            error_text = whisper_chunks_data.get("text", "") if isinstance(whisper_chunks_data, dict) else "错误：输入 whisper_chunks 格式不正确"
            return {"text": error_text, "chunks": [{"timestamp": [0,0], "text":"输入 whisper_chunks 结构错误", "speaker": "error_input_format"}], "language": ""}

        original_chunks = whisper_chunks_data.get("chunks", [])
        for _, chunk_orig in enumerate(original_chunks): # chunk_idx 未使用
            current_chunk = {}
            if isinstance(chunk_orig, dict): current_chunk = chunk_orig.copy()
            else: current_chunk = {"timestamp": [0,0], "text": str(chunk_orig), "speaker": "error_malformed_chunk_input"}

            if "timestamp" not in current_chunk or \
               not isinstance(current_chunk["timestamp"], (list, tuple)) or len(current_chunk["timestamp"]) != 2:
                current_chunk["speaker"] = "error_malformed_chunk_structure"
                processed_chunks.append(current_chunk)
                continue
            try:
                chunk_start = float(current_chunk["timestamp"][0])
                chunk_end = float(current_chunk["timestamp"][1])
            except (ValueError, TypeError, IndexError):
                current_chunk["speaker"] = "error_invalid_timestamp_values"
                processed_chunks.append(current_chunk)
                continue
            
            overlapping_segments_for_chunk = []
            for _, segment in enumerate(speaker_segments): # seg_idx 未使用
                try:
                    seg_start = float(segment["start"])
                    seg_end = float(segment["end"])
                except (ValueError, TypeError, KeyError): continue 
                overlap_start = max(chunk_start, seg_start)
                overlap_end = min(chunk_end, seg_end)
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > 0.01: 
                    overlapping_segments_for_chunk.append({
                        "speaker": segment.get("speaker", "unknown_speaker_in_segment"), 
                        "overlap_duration": overlap_duration,
                    })
            if not overlapping_segments_for_chunk:
                current_chunk["speaker"] = "unknown_no_overlap"
            else:
                speaker_total_overlaps = {} 
                for ov_seg in overlapping_segments_for_chunk:
                    spk = ov_seg["speaker"]
                    speaker_total_overlaps[spk] = speaker_total_overlaps.get(spk, 0.0) + ov_seg["overlap_duration"]
                if speaker_total_overlaps:
                    sorted_speakers_by_overlap = sorted(speaker_total_overlaps.items(), key=lambda item: item[1], reverse=True)
                    best_speaker_candidate = sorted_speakers_by_overlap[0][0]
                    max_overlap_value = sorted_speakers_by_overlap[0][1]
                    ties = [spk_info[0] for spk_info in sorted_speakers_by_overlap if spk_info[1] == max_overlap_value]
                    if len(ties) > 1:
                        ties.sort(); best_speaker_candidate = ties[0]
                    current_chunk["speaker"] = best_speaker_candidate
                else:
                    current_chunk["speaker"] = "unknown_logic_error" 
            processed_chunks.append(current_chunk)
        return {
            "text": whisper_chunks_data.get("text", ""), 
            "chunks": processed_chunks, 
            "language": whisper_chunks_data.get("language", "")
        }

    def execute_diarization(self, audio: dict, whisper_chunks: dict, backend_model: str, num_speakers: int, device: str = "cuda"):
        print(f"[AIIA E2E Diarization] 流程开始。模型: {backend_model}, 用户期望说话人数: {num_speakers}, 设备: {device}")

        if not _NEMO_E2E_PATH_INIT_SUCCESS:
             print(f"严重错误: [AIIA E2E Diarization] NeMo 模型路径初始化失败，无法继续。")
             return (self._assign_speakers_to_chunks(whisper_chunks, [{"start":0, "end":0, "speaker":"error_model_path_init"}]),)

        model_path = self._get_model_path(backend_model)
        if not model_path:
            return (self._assign_speakers_to_chunks(whisper_chunks, [{"start":0, "end":0, "speaker":f"error_model_not_found_{backend_model}"}]),)
        
        if audio is None or not isinstance(audio, dict) or \
           "waveform" not in audio or not isinstance(audio["waveform"], torch.Tensor) or \
           "sample_rate" not in audio or not isinstance(audio["sample_rate"], int) or \
           audio["waveform"].ndim < 1:
            print("错误: [AIIA E2E Diarization] 音频数据缺失、格式不正确或无效。")
            return (self._assign_speakers_to_chunks(whisper_chunks, [{"start":0, "end":0, "speaker":"error_no_audio"}]),)
        
        if not isinstance(whisper_chunks, dict) or not isinstance(whisper_chunks.get("chunks"), list) :
            print("错误: [AIIA E2E Diarization] whisper_chunks 数据无效。")
            return ({"text":whisper_chunks.get("text", "") if isinstance(whisper_chunks, dict) else "", 
                     "chunks": [{"timestamp": [0,0], "text":"输入 whisper_chunks 结构错误", "speaker": "error_input_format"}], 
                     "language":whisper_chunks.get("language", "") if isinstance(whisper_chunks, dict) else ""},)

        try:
            try: from nemo.collections.asr.models.msdd_models import SortformerEncLabelModel
            except ImportError: from nemo.collections.asr.models import SortformerEncLabelModel
            print(f"[AIIA E2E Diarization] 成功导入 SortformerEncLabelModel。")
        except ImportError as e_import_model:
            error_msg = f"错误: NeMo SortformerEncLabelModel 未找到 ({e_import_model})。请确保 nemo_toolkit['asr'] 已正确安装且版本兼容。"
            print(f"错误: [AIIA E2E Diarization] {error_msg}")
            return (self._assign_speakers_to_chunks(whisper_chunks, [{"start":0, "end":0, "speaker":"error_nemo_model_class_import"}]),)

        actual_device = torch.device(device)
        
        # 使用 with 来确保临时目录的创建和自动清理
        with tempfile.TemporaryDirectory(prefix="aiia_e2e_nemo_runtime_") as runtime_temp_dir:
            print(f"[AIIA E2E Diarization] 运行时临时目录: {runtime_temp_dir}")
            try:
                # 1. 保存临时音频文件
                waveform_tensor = audio["waveform"][0] 
                waveform_to_save = None
                if waveform_tensor.ndim == 1: waveform_to_save = waveform_tensor.cpu().numpy()
                elif waveform_tensor.ndim == 2 and waveform_tensor.shape[0] == 1: waveform_to_save = waveform_tensor.squeeze(0).cpu().numpy()
                elif waveform_tensor.ndim == 2 and waveform_tensor.shape[0] > 1: 
                    print(f"[AIIA E2E Diarization] 音频有 {waveform_tensor.shape[0]} 个通道。仅使用第一个通道。")
                    waveform_to_save = waveform_tensor[0].cpu().numpy()
                else: 
                    print(f"错误: [AIIA E2E Diarization] 不支持的音频波形维度 {waveform_tensor.ndim}")
                    return (self._assign_speakers_to_chunks(whisper_chunks, [{"start":0, "end":0, "speaker":"error_audio_format"}]),)
                
                unique_suffix = os.path.basename(runtime_temp_dir) 
                temp_wav_path = os.path.join(runtime_temp_dir, f"input_{unique_suffix}.wav")
                sf.write(temp_wav_path, waveform_to_save, audio["sample_rate"])
                print(f"[AIIA E2E Diarization] 已保存临时音频到 {temp_wav_path}")

                # 2. 加载 SortformerEncLabelModel 模型
                print(f"[AIIA E2E Diarization] 加载 Sortformer E2E 模型从: {model_path}")
                diar_model = SortformerEncLabelModel.restore_from(restore_path=model_path, map_location=actual_device)
                diar_model.eval()

                # 3. 准备 manifest 文件 (将直接传递给 diarize 方法)
                file_duration = sf.info(temp_wav_path).duration
                manifest_content = {'audio_filepath': temp_wav_path, 'offset': 0, 'duration': file_duration, 'label': 'infer', 'text': '-'}
                temp_manifest_path_for_diarize = os.path.join(runtime_temp_dir, f"manifest_for_diarize_{unique_suffix}.json")
                with open(temp_manifest_path_for_diarize, 'w', encoding='utf-8') as f:
                    json.dump(manifest_content, f); f.write('\n')
                print(f"[AIIA E2E Diarization] 已创建供 diarize() 使用的临时 manifest: {temp_manifest_path_for_diarize}")
                                
                # 4. 修改 diar_model.cfg 以确保必要的运行时参数被设置
                #    SpkDiarizationMixin.diarize() 会使用 self.cfg.output_dir
                with open_dict(diar_model.cfg):
                    # 确保 test_ds 的基本结构存在，即使 diarize() 会用参数覆盖 manifest_filepath
                    if not hasattr(diar_model.cfg, 'test_ds'): 
                        diar_model.cfg.test_ds = OmegaConf.create({})
                    diar_model.cfg.test_ds.sample_rate = 16000 # 模型期望
                    diar_model.cfg.test_ds.batch_size = 1     # diarize 参数会覆盖
                    diar_model.cfg.test_ds.num_workers = 0    # diarize 参数会覆盖
                    diar_model.cfg.test_ds.shuffle = False
                    
                    diar_model.cfg.output_dir = runtime_temp_dir # diarize() 会在这里创建 pred_rttms
                    
                    if hasattr(diar_model.cfg, 'verbose'): diar_model.cfg.verbose = True 

                print(f"[AIIA E2E Diarization] Sortformer 模型配置准备完成。Effective output_dir (from cfg): '{diar_model.cfg.output_dir}'")

                # 5. 准备并调用 diarize 方法
                diarize_call_params = {
                    "audio": temp_manifest_path_for_diarize,
                    "batch_size": 1, # 直接传递，会覆盖 cfg.test_ds.batch_size
                    "num_workers": 0,  # 直接传递，会覆盖 cfg.test_ds.num_workers
                    "verbose": True,
                    "include_tensor_outputs": False,
                }
                
                if 0 < num_speakers <= 4:
                    print(f"[AIIA E2E Diarization] 用户期望 {num_speakers} 个说话人。Sortformer (上限4人) 通常自行估计，但会记录用户期望。")
                    # SortformerEncLabelModel.diarize (来自 SpkDiarizationMixin) 不直接接受 oracle_num_speakers。
                    # 如果需要影响 Sortformer 内部的说话人数量，可能需要修改 diar_model.cfg.model.decoder.num_speakers 或类似参数 (如果存在且有效)。
                    # 目前，我们仅依赖 Sortformer 的自动估计能力。
                elif num_speakers > 4:
                     print(f"警告: [AIIA E2E Diarization] 用户指定的说话人数 {num_speakers} 超出 Sortformer 模型能力上限(4)。")
                
                print(f"[AIIA E2E Diarization] 开始运行 Sortformer diarization (调用参数: batch_size={diarize_call_params['batch_size']}, num_workers={diarize_call_params['num_workers']})...")
                
                list_of_rttm_lines_for_each_file = diar_model.diarize(**diarize_call_params)

                # 6. 解析 RTTM 内容
                speaker_segments = []
                actual_rttm_lines = []
                if isinstance(list_of_rttm_lines_for_each_file, list) and len(list_of_rttm_lines_for_each_file) > 0:
                    if isinstance(list_of_rttm_lines_for_each_file[0], list): 
                        actual_rttm_lines = list_of_rttm_lines_for_each_file[0]
                    elif isinstance(list_of_rttm_lines_for_each_file[0], str): 
                        actual_rttm_lines = list_of_rttm_lines_for_each_file
                
                if actual_rttm_lines:
                    print(f"[AIIA E2E Diarization] Diarization 返回了 {len(actual_rttm_lines)} RTTM 行。")
                    print("--- [AIIA E2E Diarization] RTTM 内容预览 (前5行): ---")
                    for i, line_content_debug in enumerate(actual_rttm_lines):
                        if i < 5: print(f"  RTTM Line {i+1}: '{line_content_debug.strip()}'")
                        else: break
                    print("--- [End of RTTM 预览] ---")

                    for line_idx, line_content in enumerate(actual_rttm_lines):
                        line = line_content.strip() 
                        if not line: continue
                        
                        parts = line.split()
                        # print(f"DEBUG: Processing RTTM line parts: {parts}")

                        # --- 新的解析逻辑：假设格式是 <start_time> <end_time> <speaker_id> ---
                        if len(parts) == 3: 
                            try:
                                start_time = float(parts[0])
                                end_time = float(parts[1]) # 假设 parts[1] 是 end_time
                                raw_id = parts[2]

                                duration = end_time - start_time # 根据 start 和 end 计算 duration

                                if duration <= 0: # 检查计算出的 duration 是否有效
                                    print(f"警告: [AIIA E2E Diarization] RTTM 行 {line_idx+1} 计算出的 duration ({duration}) 无效，跳过: '{line}'")
                                    continue

                                speaker_segments.append({
                                    "start": start_time,
                                    "end": end_time, # 直接使用 end_time
                                    "speaker": self._format_speaker_label(raw_id)
                                })
                            except ValueError:
                                print(f"警告: [AIIA E2E Diarization] RTTM 行 {line_idx+1} 无法将时间转换为浮点数，跳过: '{line}'")
                                continue
                            except IndexError: 
                                print(f"警告: [AIIA E2E Diarization] RTTM 行 {line_idx+1} 字段不足，跳过: '{line}'")
                                continue
                        else:
                            print(f"警告: [AIIA E2E Diarization] RTTM 行 {line_idx+1} 不符合预期的三字段格式，跳过: '{line}'")
                            continue
                            
                    print(f"[AIIA E2E Diarization] 从返回的 RTTM 内容解析了 {len(speaker_segments)} 个分段。")
                else: # 后备：尝试从磁盘读取 (如果 diarize() 没有返回内容)
                    print(f"[AIIA E2E Diarization] diarize() 未返回 RTTM 行或返回为空。尝试从磁盘读取。")
                    input_basename_for_rttm = os.path.basename(temp_wav_path).replace(".wav","") 
                    # SpkDiarizationMixin 会在 cfg.output_dir 下创建 pred_rttms
                    disk_rttm_path = os.path.join(diar_model.cfg.output_dir, "pred_rttms", f"{input_basename_for_rttm}.rttm")
                    if os.path.isfile(disk_rttm_path):
                        print(f"[AIIA E2E Diarization] 从磁盘后备读取 RTTM 文件: {disk_rttm_path}")
                        with open(disk_rttm_path, 'r', encoding='utf-8') as f:
                            # 读取并解析
                            for line_idx, line_content_disk in enumerate(f):
                                line_disk = line_content_disk.strip(); parts_disk = line_disk.split()
                                if not parts_disk or parts_disk[0] != "SPEAKER": continue
                                try: start_time_d, duration_d, raw_id_d = float(parts_disk[3]), float(parts_disk[4]), parts_disk[7]
                                except (IndexError, ValueError): continue
                                speaker_segments.append({"start": start_time_d, "end": start_time_d + duration_d, "speaker": self._format_speaker_label(raw_id_d)})
                        print(f"[AIIA E2E Diarization] 从磁盘的 RTTM 文件解析了 {len(speaker_segments)} 个分段。")
                    else:
                        print(f"[AIIA E2E Diarization] 未能在磁盘找到 RTTM 文件于: {disk_rttm_path}。")
                
                if not speaker_segments:
                     print(f"警告: [AIIA E2E Diarization] 最终未能获取任何说话人分段。")

                output_whisper_chunks = self._assign_speakers_to_chunks(whisper_chunks, speaker_segments)
                return (output_whisper_chunks,)

            except Exception as e:
                error_type_name = type(e).__name__
                print(f"错误: [AIIA E2E Diarization] NeMo E2E 处理过程中发生意外 ({error_type_name}): {e}")
                import traceback
                traceback.print_exc()
                return (self._assign_speakers_to_chunks(whisper_chunks, [{"start":0, "end":0, "speaker":f"error_processing_{error_type_name}"}]),)

# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_E2E_Speaker_Diarization": AIIA_E2E_Speaker_Diarization
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_E2E_Speaker_Diarization": "AIIA E2E Speaker Diarization"
}