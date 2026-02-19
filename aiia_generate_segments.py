import torch
import os
import json
import tempfile
import soundfile as sf
from omegaconf import OmegaConf, open_dict # open_dict 允许添加新键
import folder_paths # ComfyUI 的路径管理模块
from typing import Optional, List, Dict, Union # 确保导入类型提示

# --- Fix: lhotse 1.32 / PyTorch 2.10+ 兼容性补丁 ---
# PyTorch 2.10 移除了 Sampler.__init__(data_source=...) 参数，
# 但 lhotse 的 CutSampler 仍然传递它，导致:
#   TypeError: object.__init__() takes exactly one argument
# 在这里进行一次性修复，使 NeMo diarization 能正常工作。
try:
    from torch.utils.data import Sampler as _TorchSampler
    # 检测 Sampler 是否已经不接受 data_source (PyTorch >= 2.10)
    import inspect
    _sampler_params = inspect.signature(_TorchSampler.__init__).parameters
    if 'data_source' not in _sampler_params:
        try:
            from lhotse.dataset.sampling.base import CutSampler as _LhotseCutSampler
            _original_init = _LhotseCutSampler.__init__

            def _patched_cut_sampler_init(self, *args, **kwargs):
                # 绕过 super().__init__(data_source=None)，直接初始化 Sampler
                _TorchSampler.__init__(self)
                # 执行 CutSampler 自身的初始化逻辑
                self.drop_last = kwargs.get('drop_last', False)
                self.shuffle = kwargs.get('shuffle', False)
                self.seed = kwargs.get('seed', 0)
                self.epoch = 0
                from lhotse.dataset.sampling.base import SamplingDiagnostics, _filter_nothing
                self._diagnostics = SamplingDiagnostics()
                self._just_restored_state = False
                self._maybe_init_distributed(
                    world_size=kwargs.get('world_size', None),
                    rank=kwargs.get('rank', None)
                )
                from lhotse.cut import Cut
                self._filter_fn = _filter_nothing()
                self._transforms = []

            _LhotseCutSampler.__init__ = _patched_cut_sampler_init
            print("[AIIA] ✅ 已修补 lhotse CutSampler 以兼容 PyTorch 2.10+")
        except ImportError:
            pass  # lhotse 未安装，无需修补
except Exception as _patch_err:
    print(f"[AIIA] ⚠️ lhotse 兼容性补丁失败: {_patch_err}")

# --- 全局模型路径定义 ---
_NEMO_MODELS_SUBDIR_STR = "nemo_models" 
_E2E_MODEL_FILENAME_MAP = { # 支持多种E2E模型
    "diar_sortformer_4spk-v1": "diar_sortformer_4spk-v1.nemo",
    "diar_streaming_sortformer_4spk-v2.1": "diar_streaming_sortformer_4spk-v2.1.nemo",
    # "another_e2e_model": "another_e2e_model.nemo", # 未来可以扩展
}

# 路径初始化逻辑 (只在模块加载时执行一次)
_NEMO_E2E_MODEL_PATHS = {}
_NEMO_E2E_PATHS_INIT_LOG = [] 
_NEMO_E2E_PATH_INIT_SUCCESS = True 

try:
    comfyui_root_dir = folder_paths.base_path
    if not (comfyui_root_dir and isinstance(comfyui_root_dir, str) and os.path.isdir(comfyui_root_dir)):
        _NEMO_E2E_PATHS_INIT_LOG.append(f"严重错误: [AIIA Nodes] ComfyUI 根目录 '{comfyui_root_dir}' 无效。")
        _NEMO_E2E_PATH_INIT_SUCCESS = False
    
    if _NEMO_E2E_PATH_INIT_SUCCESS:
        comfyui_main_models_dir = os.path.join(comfyui_root_dir, "models")
        if not os.path.isdir(comfyui_main_models_dir):
            models_dir_attr = getattr(folder_paths, 'models_dir', None)
            if models_dir_attr and isinstance(models_dir_attr, str) and os.path.isdir(models_dir_attr):
                comfyui_main_models_dir = models_dir_attr
            else:
                _NEMO_E2E_PATHS_INIT_LOG.append(f"严重错误: [AIIA Nodes] 主 'models' 目录 '{comfyui_main_models_dir}' 未找到。")
                _NEMO_E2E_PATH_INIT_SUCCESS = False
    
    if _NEMO_E2E_PATH_INIT_SUCCESS:
        nemo_models_full_path = os.path.join(comfyui_main_models_dir, _NEMO_MODELS_SUBDIR_STR)
        if not os.path.isdir(nemo_models_full_path):
             _NEMO_E2E_PATHS_INIT_LOG.append(f"警告: [AIIA Nodes] NeMo 模型子目录 '{nemo_models_full_path}' 未找到。")
        
        found_any_model = False
        for model_key, model_filename in _E2E_MODEL_FILENAME_MAP.items():
            model_file_path = os.path.join(nemo_models_full_path, model_filename) 
            if os.path.isfile(model_file_path):
                _NEMO_E2E_MODEL_PATHS[model_key] = model_file_path
                _NEMO_E2E_PATHS_INIT_LOG.append(f"信息: [AIIA Nodes] 找到模型 '{model_key}': {model_file_path}")
                found_any_model = True
            else:
                _NEMO_E2E_PATHS_INIT_LOG.append(f"错误: [AIIA Nodes] 模型 '{model_key}' 的文件 '{model_filename}' 未在 '{nemo_models_full_path}' 找到。")
        
        if not found_any_model and _NEMO_E2E_PATH_INIT_SUCCESS: 
            _NEMO_E2E_PATHS_INIT_LOG.append(f"警告: [AIIA Nodes] 未能定位到任何配置的 E2E NeMo 模型文件。")
            # 如果一个模型都没找到，路径初始化也应该算作不完全成功
            if not _NEMO_E2E_MODEL_PATHS: # 如果字典为空
                _NEMO_E2E_PATH_INIT_SUCCESS = False


except Exception as e:
    _NEMO_E2E_PATH_INIT_SUCCESS = False

# aiia_generate_segments.py

# ... (顶部的 imports 和全局路径定义保持不变) ...

class AIIA_GenerateSpeakerSegments:
    NODE_NAME = "AIIA Generate Speaker Segments" 

    @classmethod
    def INPUT_TYPES(cls):
        available_models = list(_NEMO_E2E_MODEL_PATHS.keys())
        default_model_option = "NO_MODELS_FOUND" 
        if available_models: default_model_option = available_models[0]
        else: available_models = [default_model_option]
            
        # +++ 更新 Profile 列表 +++
        postprocessing_profiles = [
            "very_permissive", 
            "permissive",       # 之前的默认和宽松
            "balanced",         # 新增
            "strict",           # 之前的严格
            "very_strict",      # 新增
            "custom"
        ]
        default_profile = "balanced" # 将 "balanced" 作为新的默认值，或保持 "permissive"

        return {
            "required": {
                "audio": ("AUDIO",),
                "e2e_backend_model": (available_models, {"default": default_model_option}),
                "num_speakers_hint": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1, 
                                              "tooltip": "期望说话人数 (0=模型自动估计, 最多4人)。此为提示。"}),
                "postprocessing_profile": (postprocessing_profiles, {"default": default_profile, # 更新默认
                                               "tooltip": "选择后处理参数配置方案。"}),
            },
            "optional": {
                 "device": (["cuda", "cpu"], {"default": "cuda"}),
                 # 自定义参数的默认值可以与 "balanced" 或 "permissive" 的一致
                 "custom_onset": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"(Custom Profile) VAD onset threshold"}),
                 "custom_offset": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":"(Custom Profile) VAD offset threshold"}),
                 "custom_min_duration_on": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip":"(Custom Profile) Min duration for a speech segment"}),
                 "custom_min_duration_off": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 2.0, "step": 0.01, "tooltip":"(Custom Profile) Min duration for a non-speech segment"}),
                 "custom_pad_onset": ("FLOAT", {"default": 0.05, "min": -0.5, "max": 0.5, "step": 0.01, "tooltip":"(Custom Profile) Padding for speech segment onset"}),
                 "custom_pad_offset": ("FLOAT", {"default": 0.05, "min": -0.5, "max": 0.5, "step": 0.01, "tooltip":"(Custom Profile) Padding for speech segment offset"}),
            }
        }
    
    RETURN_TYPES = ("WHISPER_CHUNKS",) 
    RETURN_NAMES = ("speaker_segments",)
    FUNCTION = "generate_segments"
    CATEGORY = "AIIA/audio"

    def _get_model_path(self, backend_model_key: str) -> Optional[str]:
        model_path = _NEMO_E2E_MODEL_PATHS.get(backend_model_key)
        if not model_path: 
            print(f"错误: [{self.NODE_NAME}] E2E 模型 '{backend_model_key}' 的路径在运行时未找到。")
            return None
        return model_path

    def _format_speaker_label(self, raw_speaker_id: str) -> str:
        final_numeric_id = -1
        if raw_speaker_id.startswith("speaker_"):
            try: final_numeric_id = int(raw_speaker_id.split('_')[-1])
            except ValueError: pass 
        elif raw_speaker_id.isdigit():
            try: final_numeric_id = int(raw_speaker_id)
            except ValueError: pass 
        elif raw_speaker_id.startswith("SPEAKER_"):
            potential_id = raw_speaker_id.split('_')[-1]
            if potential_id.isdigit():
                try: final_numeric_id = int(potential_id)
                except ValueError: pass
        if final_numeric_id != -1:
            return f"SPEAKER_{final_numeric_id:02d}"
        return f"SPEAKER_{raw_speaker_id}" if not raw_speaker_id.startswith("SPEAKER_") else raw_speaker_id

    def _create_error_output(self, error_message_text: str) -> tuple:
        node_name_log = f"[{self.NODE_NAME}]"
        print(f"{node_name_log} 错误: {error_message_text}")
        error_label_suffix = error_message_text.lower().replace(' ', '_').split('(')[0].strip()
        for char_to_remove in ".:[]'\"": 
            error_label_suffix = error_label_suffix.replace(char_to_remove, '')
        error_speaker = f"error_{error_label_suffix[:30]}" 
        error_output_dict = {
            "text": "", "chunks": [{"timestamp": [0,0], "text": f"错误: {error_message_text}", "speaker": error_speaker}], "language": ""
        }
        return (error_output_dict,)

    def generate_segments(self, audio: dict, e2e_backend_model: str, num_speakers_hint: int, 
                          postprocessing_profile: str, device: str = "cuda",
                          custom_onset: float = 0.6, custom_offset: float = 0.4, # 保持与 INPUT_TYPES 中 custom 的默认值一致
                          custom_min_duration_on: float = 0.1, custom_min_duration_off: float = 0.2,
                          custom_pad_onset: float = 0.05, custom_pad_offset: float = 0.05):
        
        node_name_log = f"[{self.NODE_NAME} / {e2e_backend_model}]"
        print(f"{node_name_log} 流程开始。用户提示说话人数: {num_speakers_hint}, 后处理配置: {postprocessing_profile}, 设备: {device}")

        if not _NEMO_E2E_PATH_INIT_SUCCESS: 
            return self._create_error_output("NeMo 模型路径初始化失败")
        if e2e_backend_model == "NO_MODELS_FOUND": 
            return self._create_error_output("没有找到有效的 NeMo E2E 模型，请检查模型文件及路径日志")
        model_path = self._get_model_path(e2e_backend_model)
        if not model_path: 
            return self._create_error_output(f"模型 '{e2e_backend_model}' 文件路径无效")
        
        if audio is None:
             return self._create_error_output("音频数据为 None")
             
        # Handle cases where audio might be passed as a single-item list
        if isinstance(audio, list) and len(audio) > 0:
            audio = audio[0]

        # Try to treat as a dictionary or object with waveform/sample_rate
        try:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
        except (KeyError, TypeError):
             try:
                 waveform = getattr(audio, "waveform", None)
                 sample_rate = getattr(audio, "sample_rate", None)
             except:
                 waveform, sample_rate = None, None

        if waveform is None or sample_rate is None:
            return self._create_error_output(f"音频数据格式错误: 无法获取 waveform 或 sample_rate (输入类型: {type(audio)})")

        # Ensure waveform is a tensor and sample_rate is a number
        if not isinstance(waveform, torch.Tensor) or not isinstance(sample_rate, (int, float)):
            return self._create_error_output(f"音频数据类型错误: waveform={type(waveform)}, sample_rate={type(sample_rate)}")

        if waveform.ndim < 1:
            return self._create_error_output("音频波形维度不足")

        # 检查音频长度
        if audio["waveform"].shape[-1] == 0:
            return self._create_error_output("输入的音频长度为0，无法进行分段")


        try:
            import sys, time as _t
            print(f"{node_name_log} [DEBUG] 开始导入 NeMo 类... ({_t.strftime('%H:%M:%S')})", flush=True)
            sys.stdout.flush()
            try:
                print(f"{node_name_log} [DEBUG]  importing SortformerEncLabelModel from msdd_models...", flush=True)
                from nemo.collections.asr.models.msdd_models import SortformerEncLabelModel
            except ImportError:
                print(f"{node_name_log} [DEBUG]  msdd_models failed, trying asr.models...", flush=True)
                from nemo.collections.asr.models import SortformerEncLabelModel
            print(f"{node_name_log} [DEBUG]  SortformerEncLabelModel imported OK ({_t.strftime('%H:%M:%S')})", flush=True)
            
            print(f"{node_name_log} [DEBUG]  importing DiarizeConfig...", flush=True)
            from nemo.collections.asr.parts.mixins.diarization import DiarizeConfig 
            print(f"{node_name_log} [DEBUG]  DiarizeConfig imported OK ({_t.strftime('%H:%M:%S')})", flush=True)
            
            print(f"{node_name_log} 成功导入 NeMo 类。")
        except ImportError as e_import_model:
            return self._create_error_output(f"导入 NeMo 类失败 ({e_import_model})")

        actual_device = torch.device(device)
        
        with tempfile.TemporaryDirectory(prefix="aiia_gs_nemo_") as runtime_temp_dir:
            print(f"{node_name_log} 运行时临时目录: {runtime_temp_dir}")
            try:
                waveform_tensor = audio["waveform"][0] 
                waveform_to_save = None
                if waveform_tensor.ndim == 1: waveform_to_save = waveform_tensor.cpu().numpy()
                elif waveform_tensor.ndim == 2 and waveform_tensor.shape[0] == 1: waveform_to_save = waveform_tensor.squeeze(0).cpu().numpy()
                elif waveform_tensor.ndim == 2 and waveform_tensor.shape[0] > 1: 
                    print(f"{node_name_log} 音频有 {waveform_tensor.shape[0]} 个通道。仅使用第一个通道。")
                    waveform_to_save = waveform_tensor[0].cpu().numpy()
                else: 
                    return self._create_error_output(f"不支持的音频波形维度 {waveform_tensor.ndim}")
                
                unique_suffix = os.path.basename(runtime_temp_dir) 
                temp_wav_path = os.path.join(runtime_temp_dir, f"input_{unique_suffix}.wav")
                sf.write(temp_wav_path, waveform_to_save, audio["sample_rate"])
                print(f"{node_name_log} 已保存临时音频到 {temp_wav_path}")

                print(f"{node_name_log} 加载 E2E 模型: {model_path}")
                
                # --- Diagnostic: trace hang point in NeMo restore_from ---
                import time as _time
                _t0 = _time.time()
                
                # Patch SortformerEncLabelModel.__init__ to trace progress
                _orig_init = SortformerEncLabelModel.__init__
                def _traced_init(self_model, cfg, trainer=None):
                    print(f"[AIIA DEBUG] SortformerEncLabelModel.__init__ START ({_time.time()-_t0:.1f}s)")
                    import nemo.core.classes.modelPT as _mpt
                    _orig_mpt_init = _mpt.ModelPT.__init__
                    def _traced_mpt_init(self2, **kwargs2):
                        print(f"[AIIA DEBUG]   ModelPT.__init__ START ({_time.time()-_t0:.1f}s)")
                        _orig_mpt_init(self2, **kwargs2)
                        print(f"[AIIA DEBUG]   ModelPT.__init__ DONE ({_time.time()-_t0:.1f}s)")
                    _mpt.ModelPT.__init__ = _traced_mpt_init
                    try:
                        _orig_init(self_model, cfg, trainer)
                    finally:
                        _mpt.ModelPT.__init__ = _orig_mpt_init
                    print(f"[AIIA DEBUG] SortformerEncLabelModel.__init__ DONE ({_time.time()-_t0:.1f}s)")
                SortformerEncLabelModel.__init__ = _traced_init
                
                try:
                    print(f"[AIIA DEBUG] restore_from START ({_time.time()-_t0:.1f}s)")
                    diar_model = SortformerEncLabelModel.restore_from(restore_path=model_path, map_location=actual_device)
                    print(f"[AIIA DEBUG] restore_from DONE ({_time.time()-_t0:.1f}s)")
                finally:
                    SortformerEncLabelModel.__init__ = _orig_init
                
                diar_model.eval()

                file_duration = sf.info(temp_wav_path).duration
                manifest_content = {'audio_filepath': temp_wav_path, 'offset': 0, 'duration': file_duration, 'label': 'infer', 'text': '-'}
                temp_manifest_path_for_diarize = os.path.join(runtime_temp_dir, f"manifest_for_diarize_{unique_suffix}.json")
                with open(temp_manifest_path_for_diarize, 'w', encoding='utf-8') as f:
                    json.dump(manifest_content, f); f.write('\n')
                print(f"{node_name_log} 已创建供 diarize() 使用的临时 manifest: {temp_manifest_path_for_diarize}")
                                
                with open_dict(diar_model.cfg):
                    if not hasattr(diar_model.cfg, 'test_ds'): 
                        diar_model.cfg.test_ds = OmegaConf.create({})
                    diar_model.cfg.test_ds.num_workers = 0    
                    diar_model.cfg.test_ds.batch_size = 1   
                    diar_model.cfg.output_dir = runtime_temp_dir 
                    if hasattr(diar_model.cfg, 'verbose'): diar_model.cfg.verbose = True 
                print(f"{node_name_log} 模型配置准备完成。")

                # +++ 根据选择的 profile 定义后处理参数字典 +++
                postprocessing_params_dict = {}
                if postprocessing_profile == "very_permissive":
                    postprocessing_params_dict = {"onset": 0.3, "offset": 0.1, "min_duration_on": 0.02, "min_duration_off": 0.02, "pad_onset": 0.0, "pad_offset": 0.0}
                elif postprocessing_profile == "permissive":
                    postprocessing_params_dict = {"onset": 0.4, "offset": 0.2, "min_duration_on": 0.05, "min_duration_off": 0.05, "pad_onset": 0.0, "pad_offset": 0.0}
                elif postprocessing_profile == "balanced":
                    postprocessing_params_dict = {"onset": 0.6, "offset": 0.4, "min_duration_on": 0.1, "min_duration_off": 0.2, "pad_onset": 0.05, "pad_offset": 0.05}
                elif postprocessing_profile == "strict":
                    postprocessing_params_dict = {"onset": 0.8, "offset": 0.6, "min_duration_on": 0.2, "min_duration_off": 0.4, "pad_onset": 0.1, "pad_offset": 0.1}
                elif postprocessing_profile == "very_strict":
                    postprocessing_params_dict = {"onset": 0.9, "offset": 0.7, "min_duration_on": 0.3, "min_duration_off": 0.5, "pad_onset": 0.15, "pad_offset": 0.15}
                elif postprocessing_profile == "custom":
                    postprocessing_params_dict = {"onset": custom_onset, "offset": custom_offset, "min_duration_on": custom_min_duration_on, "min_duration_off": custom_min_duration_off, "pad_onset": custom_pad_onset, "pad_offset": custom_pad_offset}
                else: # 默认 (例如，如果 profile 字符串意外错误，回退到 balanced)
                    postprocessing_params_dict = {"onset": 0.6, "offset": 0.4, "min_duration_on": 0.1, "min_duration_off": 0.2, "pad_onset": 0.05, "pad_offset": 0.05}
                    print(f"警告: [{node_name_log}] 未知的后处理配置 '{postprocessing_profile}'，已回退到 Balanced。")

                # +++ 修改日志，直接使用 postprocessing_profile +++
                print(f"{node_name_log} 当前生效的后处理 Profile: {postprocessing_profile.capitalize()}") # 使用 capitalize() 使首字母大写
                if postprocessing_profile == "custom": # 仅在 custom 时打印详细参数，因为 permissive/strict 是预设的
                    print(f"{node_name_log} 使用的自定义后处理参数字典: {postprocessing_params_dict}")

                # 准备 DiarizeConfig，并将后处理参数字典直接传递给 postprocessing_params 字段
                # 这是基于你之前的反馈，即这种方式在你环境中被接受并且解决了下游类型问题
                diar_override_config = DiarizeConfig(
                    postprocessing_params=postprocessing_params_dict, # 直接传递字典
                    postprocessing_yaml=None # 确保 YAML 路径为 None
                )
                print(f"{node_name_log} 已创建 DiarizeConfig，并将后处理参数字典直接赋给 postprocessing_params。")

                diarize_call_params = {
                    "audio": temp_manifest_path_for_diarize,
                    "batch_size": 1,            
                    "num_workers": 0,           
                    "verbose": True, 
                    "include_tensor_outputs": False,
                    "override_config": diar_override_config 
                }
                
                if 0 < num_speakers_hint <= 4:
                    print(f"{node_name_log} 用户提示期望 {num_speakers_hint} 个说话人。Sortformer 将自行估计。")
                
                # 在这里也加入 profile 信息
                print(f"{node_name_log} 开始运行 Sortformer diarization (使用 {postprocessing_profile.capitalize()} 后处理配置)...") 
                list_of_rttm_lines_for_each_file = diar_model.diarize(**diarize_call_params)

                speaker_segments_for_json_chunks = [] 
                actual_rttm_lines = []
                if isinstance(list_of_rttm_lines_for_each_file, list) and len(list_of_rttm_lines_for_each_file) > 0:
                    if isinstance(list_of_rttm_lines_for_each_file[0], list): actual_rttm_lines = list_of_rttm_lines_for_each_file[0]
                    elif isinstance(list_of_rttm_lines_for_each_file[0], str) : actual_rttm_lines = list_of_rttm_lines_for_each_file
                
                if actual_rttm_lines:
                    print(f"{node_name_log} Diarization 返回了 {len(actual_rttm_lines)} RTTM 行。")
                    for line_idx, line_content in enumerate(actual_rttm_lines):
                        line = line_content.strip(); parts = line.split()
                        if not parts : continue
                        if len(parts) == 3: 
                            try: 
                                start_time, end_time, raw_id = float(parts[0]), float(parts[1]), parts[2]
                                duration = end_time - start_time
                                if duration <= 0.01: continue
                                speaker_segments_for_json_chunks.append({"timestamp": [round(start_time,3), round(end_time,3)], "text": "", "speaker": self._format_speaker_label(raw_id)})
                            except (ValueError, IndexError) as e_rttm_simple: 
                                print(f"警告: [{self.NODE_NAME}] 解析简易 RTTM 行 {line_idx+1} 失败: '{line}', 错误: {e_rttm_simple}")
                                continue
                        elif parts[0] == "SPEAKER" and len(parts) >= 8:
                            try:
                                start_time, duration, raw_id = float(parts[3]), float(parts[4]), parts[7]
                                if duration <= 0.01: continue
                                speaker_segments_for_json_chunks.append({"timestamp": [round(start_time,3), round(start_time + duration,3)], "text": "", "speaker": self._format_speaker_label(raw_id)})
                            except (ValueError, IndexError) as e_rttm_std:
                                print(f"警告: [{self.NODE_NAME}] 解析标准 RTTM 行 {line_idx+1} 失败: '{line}', 错误: {e_rttm_std}")
                                continue
                        else:
                            print(f"警告: [{self.NODE_NAME}] 未知 RTTM 行格式 {line_idx+1}，跳过: '{line}'")
                    print(f"{node_name_log} 从返回的 RTTM 内容解析了 {len(speaker_segments_for_json_chunks)} 个分段。")
                else: 
                    # 此处不再尝试从磁盘后备读取，因为 diarize() 应该直接返回内容
                    print(f"{node_name_log} diarize() 未返回 RTTM 行或返回为空。")
                
                if not speaker_segments_for_json_chunks:
                     print(f"警告: [{self.NODE_NAME}] 最终未能获取任何说话人分段。")
                
                output_data_structure = {"text": "", "chunks": speaker_segments_for_json_chunks, "language": ""}
                
                # Cleanup: Move model to CPU and delete
                try:
                    if 'diar_model' in locals() and diar_model is not None:
                        print(f"{node_name_log} Cleaning up NeMo model (Moving to CPU)...")
                        diar_model.to("cpu")
                        if hasattr(diar_model, 'encoder'): diar_model.encoder.to("cpu")
                        if hasattr(diar_model, 'decoder'): diar_model.decoder.to("cpu")
                        del diar_model
                    torch.cuda.empty_cache()
                except Exception as cleanup_err:
                    print(f"Warning: Cleanup failed: {cleanup_err}")

                print(f"{node_name_log} 流程结束。")
                return (output_data_structure,)

            except Exception as e:
                return self._create_error_output(f"NeMo E2E 处理过程中发生意外: {e}")
            # finally 块中不再需要删除 temp_postprocessing_yaml_path

    # --- 辅助方法 _get_model_path, _format_speaker_label, _create_error_output ---
    # 【确保这些方法与你之前成功运行的版本一致】

# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_GenerateSpeakerSegments": AIIA_GenerateSpeakerSegments
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_GenerateSpeakerSegments": "AIIA Generate Speaker Segments"
}