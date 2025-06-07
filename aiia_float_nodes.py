# --- START OF FILE aiia_float_nodes.py ---

import torch
import os
import tempfile
import torchaudio
import torchvision.utils as vutils
import numpy as np
import folder_paths
import time
import types
from comfy.utils import ProgressBar  # ComfyUI 进度条
from PIL import Image
import traceback
from tqdm import tqdm # 导入 tqdm

# ----------------------------------------------------------------------------------
# 辅助函数：打过补丁的解码逻辑
# ----------------------------------------------------------------------------------

def _patched_decode_for_in_memory_stack(
    self_float_model, # FLOATModel 实例 (float_pipe.G)
    s_r: torch.Tensor, 
    s_r_feats: list, 
    r_d: torch.Tensor
) -> dict: # 返回 {'d_hat': cpu_stacked_tensor_tchw}
    
    T_prime = r_d.shape[1]
    B = r_d.shape[0] # 应该总是 1
    
    comfy_pbar = ProgressBar(T_prime) # ComfyUI 进度条
    
    console_pbar_desc = "[FLOAT In-Memory] Processing Frames"
    # 尝试获取更具体的描述，如果 self_float_model 是可预期的类型
    if hasattr(self_float_model, '__class__') and hasattr(self_float_model.__class__, '__name__'):
        model_name = self_float_model.__class__.__name__
        if model_name != "FLOATModel": # 如果不是通用的 FLOATModel，则使用更具体的名称
             console_pbar_desc = f"[{model_name} In-Memory] Processing Frames"


    with tqdm(total=T_prime, desc=console_pbar_desc, unit="frame") as console_pbar:
        processed_frames_cpu_list = []
        opt = self_float_model.opt 
        FRAMES_PER_GPU_CHUNK = getattr(opt, 'decode_gpu_chunk_size', 32) 
        
        gpu_frame_buffer = []

        for t_idx in range(T_prime):
            current_motion_vector = r_d[:, t_idx] 
            s_r_plus_motion = s_r + current_motion_vector 
            img_t_gpu_raw, _ = self_float_model.motion_autoencoder.dec(s_r_plus_motion, alpha=None, feats=s_r_feats)
            img_t_gpu_clamped = torch.clamp(img_t_gpu_raw, -1, 1) # 值域 [-1, 1]

            gpu_frame_buffer.append(img_t_gpu_clamped.squeeze(0) if B == 1 else img_t_gpu_clamped[0])

            if len(gpu_frame_buffer) >= FRAMES_PER_GPU_CHUNK or \
               (t_idx == T_prime - 1 and len(gpu_frame_buffer) > 0):
                for frame_gpu in gpu_frame_buffer:
                    processed_frames_cpu_list.append(frame_gpu.cpu())
                gpu_frame_buffer = [] # 清空 buffer
            
            comfy_pbar.update(1)
            console_pbar.update(1) # 更新 tqdm 进度条

    del r_d, s_r, s_r_feats 

    if not processed_frames_cpu_list:
        print(f"警告: [PatchedDecodeInMemory] 未生成任何帧。")
        img_c = getattr(opt, 'output_nc', 3); img_h = getattr(opt, 'input_size', 64); img_w = getattr(opt, 'input_size', 64)
        return {'d_hat': torch.empty((0, img_c, img_h, img_w), device='cpu')}
    try:
        d_hat_stacked_cpu = torch.stack(processed_frames_cpu_list, dim=0) # (T, C, H, W)
        return {'d_hat': d_hat_stacked_cpu} 
    except RuntimeError as e_cpu_stack: 
        print(f"错误: [PatchedDecodeInMemory] CPU堆叠错误: {e_cpu_stack}")
        raise


def _patched_decode_and_save_to_disk(
    self_float_model, 
    s_r: torch.Tensor, 
    s_r_feats: list, 
    r_d: torch.Tensor,
    output_frames_dir: str, 
    node_name_log_prefix: str # 这个可以用作 tqdm 的 desc
) -> dict:
    
    T_prime = r_d.shape[1]
    B = r_d.shape[0] # 应该总是 1 (由原始代码的 squeeze(0) 暗示)
    
    comfy_pbar = ProgressBar(T_prime) # ComfyUI 进度条
    
    # 使用传入的 node_name_log_prefix 作为基础描述，并添加操作说明
    console_pbar_desc = f"{node_name_log_prefix} Saving Frames"

    with tqdm(total=T_prime, desc=console_pbar_desc, unit="frame") as console_pbar:
        opt = self_float_model.opt
        FRAMES_PER_GPU_CHUNK_FOR_PROCESSING = getattr(opt, 'frames_per_gpu_chunk_for_processing', 16)
        
        gpu_frame_buffer = []; saved_frame_count = 0
        self_float_model._last_run_saved_frames = 0

        for t_idx in range(T_prime):
            current_motion_vector = r_d[:, t_idx] 
            s_r_plus_motion = s_r + current_motion_vector 
            img_t_gpu_raw, _ = self_float_model.motion_autoencoder.dec(s_r_plus_motion, alpha=None, feats=s_r_feats)
            img_t_gpu_clamped = torch.clamp(img_t_gpu_raw, -1, 1)
            gpu_frame_buffer.append(img_t_gpu_clamped.squeeze(0) if B == 1 else img_t_gpu_clamped[0])

            if len(gpu_frame_buffer) >= FRAMES_PER_GPU_CHUNK_FOR_PROCESSING or \
               (t_idx == T_prime - 1 and len(gpu_frame_buffer) > 0):
                if gpu_frame_buffer: # 确保 buffer 不为空
                    current_gpu_chunk_to_process = torch.stack(gpu_frame_buffer, dim=0) if len(gpu_frame_buffer) > 1 else gpu_frame_buffer[0].unsqueeze(0)
                    gpu_frame_buffer = [] 
                    
                    chunk_cpu_chw = current_gpu_chunk_to_process.cpu(); del current_gpu_chunk_to_process 
                    chunk_cpu_hwc_float_0_1 = ((chunk_cpu_chw.permute(0, 2, 3, 1).clamp(-1,1) + 1.0) / 2.0)
                    
                    for frame_idx_in_chunk in range(chunk_cpu_hwc_float_0_1.shape[0]):
                        frame_to_save_np = (chunk_cpu_hwc_float_0_1[frame_idx_in_chunk].numpy() * 255).astype(np.uint8)
                        filename = f"frame_{saved_frame_count:06d}.png"
                        filepath = os.path.join(output_frames_dir, filename)
                        try: 
                            Image.fromarray(frame_to_save_np).save(filepath)
                            saved_frame_count += 1
                        except Exception as e_save: 
                            # 对于保存单个帧的错误，可以选择不打断 tqdm，只打印警告
                            # console_pbar.write(f"警告: 保存帧 {filepath} 失败: {e_save}")
                            # 或者如果希望错误更明显，可以打印到标准错误流
                            print(f"警告: [{node_name_log_prefix}] 保存帧 {filepath} 失败: {e_save}", file=sys.stderr if 'sys' in globals() else None) # pylint: disable=undefined-variable
                    del chunk_cpu_chw, chunk_cpu_hwc_float_0_1 
            
            comfy_pbar.update(1)
            console_pbar.update(1) # 更新 tqdm 进度条
        
    del r_d, s_r, s_r_feats
    # tqdm 会在 with 块结束时自动清理/关闭
    # 如果有任何最终信息要在进度条下方打印，确保在 with 块之后
    # （tqdm 默认会这样做，除非 leave=False 且没有其他输出）
    
    print(f"信息: [{node_name_log_prefix}] 已处理并尝试保存 {saved_frame_count} 帧到 {output_frames_dir}")
    self_float_model._last_run_saved_frames = saved_frame_count
    
    img_c = getattr(opt, 'output_nc', 3); img_h = getattr(opt, 'input_size', 64); img_w = getattr(opt, 'input_size', 64)
    batch_size_for_placeholder = B if B > 0 else 1 # 确保批次大小至少为1
    d_hat_placeholder_btchw = torch.empty((batch_size_for_placeholder, 0, img_c, img_h, img_w), device='cpu') 
    return {'d_hat': d_hat_placeholder_btchw}


class AIIA_FloatProcess_InMemory:
    NODE_NAME = "AIIA Float Process (In-Memory Output)"
    CATEGORY = "AIIA/FLOAT"
    FUNCTION = "floatprocess_in_memory"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float_pipe": ("FLOAT_PIPE",),"ref_image": ("IMAGE",),"ref_audio": ("AUDIO",),"a_cfg_scale": ("FLOAT", {"default": 2.0,"min": 0.0, "max": 10.0, "step": 0.1}),"r_cfg_scale": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.1}),"e_cfg_scale": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.1}),"fps": ("FLOAT", {"default": 25.0, "min":1.0, "max": 60.0, "step": 0.5}),"emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none"}),"crop_input_image": ("BOOLEAN",{"default":False},),"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),"nfe": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}), },"optional": {"device_override": (["default", "cuda", "cpu"], {"default": "default"}), "decode_gpu_chunk_size": ("INT", {"default": 32, "min":1, "max":128, "step":1, "tooltip":"(In-Memory) GPU解码后一次转移多少帧到CPU。影响显存和速度。"}),}}

    def _create_error_image(self, error_message_text: str, log_message: bool = True) -> tuple:
        if log_message:
            print(f"错误: [{self.__class__.NODE_NAME}] {error_message_text}")
        return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

    def floatprocess_in_memory(self, float_pipe, ref_image, ref_audio, 
                               a_cfg_scale, r_cfg_scale, e_cfg_scale, 
                               fps, emotion, crop_input_image, seed, nfe, 
                               device_override: str = "default", 
                               decode_gpu_chunk_size: int = 32):
        node_name_log = f"[{self.__class__.NODE_NAME}]"
        print(f"{node_name_log} 流程开始 (内存输出模式)。")
        start_time_process = time.time()

        _default_error_tuple = self._create_error_image("未知错误 (初始化或预处理失败)", log_message=False)
        return_value = _default_error_tuple 

        if float_pipe is None or not hasattr(float_pipe, 'opt') or not hasattr(float_pipe.G, 'decode_latent_into_image'):
            return self._create_error_image("float_pipe 无效或不完整", log_message=True)
        
        processing_device = torch.device(device_override) if device_override != "default" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{node_name_log} 本次运行将在设备上: {processing_device}")
        
        original_decode_method = None
        original_opt_rank_backup = getattr(float_pipe.opt, 'rank', None)
        original_opt_fps_backup = getattr(float_pipe.opt, 'fps', None)  
        original_opt_decode_chunk_backup = getattr(float_pipe.opt, 'decode_gpu_chunk_size', None)

        with tempfile.TemporaryDirectory(prefix="aiia_fp_inmem_") as temp_run_dir:
            try:
                audio_waveform_to_save = ref_audio['waveform'].squeeze(0);
                if audio_waveform_to_save.ndim > 1 and audio_waveform_to_save.shape[0] > 1: audio_waveform_to_save = audio_waveform_to_save[0]
                audio_save_path = os.path.join(temp_run_dir, "temp_audio.wav"); torchaudio.save(audio_save_path, audio_waveform_to_save.cpu(), ref_audio["sample_rate"])
                ref_image_chw = ref_image[0].permute(2, 0, 1).cpu(); image_save_path = os.path.join(temp_run_dir, "temp_ref_image.png")
                vutils.save_image(ref_image_chw, image_save_path, normalize=False)

                if hasattr(float_pipe.opt, 'rank'):
                    float_pipe.opt.rank = processing_device.index if processing_device.type == 'cuda' and processing_device.index is not None else (0 if processing_device.type == 'cuda' else -1)
                if hasattr(float_pipe.opt, 'fps'): float_pipe.opt.fps = float(fps)
                float_pipe.opt.decode_gpu_chunk_size = decode_gpu_chunk_size
                print(f"{node_name_log} opt 更新: rank={getattr(float_pipe.opt, 'rank', 'N/A')}, fps={getattr(float_pipe.opt, 'fps', 'N/A')}, decode_chunk={getattr(float_pipe.opt, 'decode_gpu_chunk_size', 'N/A')}")
                
                model_current_device_before_move = next(float_pipe.G.parameters()).device
                if model_current_device_before_move != processing_device: float_pipe.G.to(processing_device)
                print(f"{node_name_log} 模型移至: {processing_device}")

                original_decode_method = float_pipe.G.decode_latent_into_image
                float_pipe.G.decode_latent_into_image = types.MethodType(_patched_decode_for_in_memory_stack, float_pipe.G)
                print(f"信息: {node_name_log} 已替换 decode_latent_into_image 为内存堆叠版本。")
                
                print(f"{node_name_log} 开始运行推理...")
                images_thwc_cpu_float01 = float_pipe.run_inference(
                    res_video_path=None, ref_path=image_save_path, audio_path=audio_save_path,
                    a_cfg_scale=a_cfg_scale, r_cfg_scale=r_cfg_scale, e_cfg_scale=e_cfg_scale,
                    emo=None if emotion == "none" else emotion,
                    no_crop=not crop_input_image, nfe=nfe, seed=seed, verbose=False 
                )
                if not isinstance(images_thwc_cpu_float01, torch.Tensor):
                    images_thwc_cpu_float01 = torch.from_numpy(images_thwc_cpu_float01.astype(np.float32))
                
                print(f"信息: {node_name_log} 推理完成。输出图像序列形状: {images_thwc_cpu_float01.shape if images_thwc_cpu_float01 is not None else 'None'}")
                return_value = (images_thwc_cpu_float01,)
            except Exception as e_proc_inner:
                print(f"错误: {node_name_log} 内部处理错误: {e_proc_inner}"); traceback.print_exc()
                return_value = self._create_error_image(f"内部处理错误: {e_proc_inner}", log_message=True)
            finally:
                if original_decode_method and hasattr(float_pipe.G, 'decode_latent_into_image'):
                    current_decode_method = getattr(float_pipe.G, 'decode_latent_into_image', None)
                    # 检查当前绑定方法的 __func__ 是否是我们的 patch 函数
                    if hasattr(current_decode_method, '__func__') and current_decode_method.__func__ is _patched_decode_for_in_memory_stack: 
                        float_pipe.G.decode_latent_into_image = original_decode_method
                        print(f"信息: {node_name_log} (finally) 已恢复 float_pipe.G.decode_latent_into_image 为原始方法。")
                    # 添加一个 elif 以防万一我们的 patch 方式改变了或者上面没匹配上，但方法确实不是原始的了
                    elif current_decode_method is not original_decode_method:
                        print(f"警告: {node_name_log} (finally) decode_latent_into_image 不是预期中的 patch 方法，但也不是原始方法。仍尝试恢复为原始方法。")
                        float_pipe.G.decode_latent_into_image = original_decode_method

                if original_opt_rank_backup is not None: float_pipe.opt.rank = original_opt_rank_backup
                if original_opt_fps_backup is not None: float_pipe.opt.fps = original_opt_fps_backup
                if original_opt_decode_chunk_backup is not None : float_pipe.opt.decode_gpu_chunk_size = original_opt_decode_chunk_backup
                elif hasattr(float_pipe.opt, 'decode_gpu_chunk_size'): 
                    try:
                        del float_pipe.opt.decode_gpu_chunk_size
                    except AttributeError: # 以防万一 opt 不是 OmegaConf 或普通 dict
                        pass 
                
                current_g_device_after_proc = next(float_pipe.G.parameters()).device
                if current_g_device_after_proc.type == 'cuda':
                    try: 
                        float_pipe.G.to(torch.device("cpu"))
                        torch.cuda.empty_cache()
                    except Exception as e_to_cpu: 
                        print(f"{node_name_log} (finally) 模型移至CPU或清空缓存时出错: {e_to_cpu}")
        
        end_time_process = time.time()
        print(f"{node_name_log} 方法总执行耗时: {end_time_process - start_time_process:.2f} 秒。")
        return return_value


class AIIA_FloatProcess_ToDisk:
    NODE_NAME = "AIIA Float Process (To Disk)"
    CATEGORY = "AIIA/FLOAT"
    FUNCTION = "floatprocess_to_disk"
    RETURN_TYPES = ("STRING", "INT") 
    RETURN_NAMES = ("frames_output_directory", "saved_frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        # ... (INPUT_TYPES 不变) ...
        base_inputs = AIIA_FloatProcess_InMemory.INPUT_TYPES()
        base_inputs["optional"]["output_subdir_name"] = ("STRING", {"default": "float_frames_AIIA", "tooltip": "在ComfyUI输出目录下创建的子目录名"})
        if "decode_gpu_chunk_size" not in base_inputs["optional"]:
             base_inputs["optional"]["decode_gpu_chunk_size"] = ("INT", {"default": 32, "min":1, "max":128, "step":1, "tooltip":"GPU解码后一次处理并保存多少帧。影响显存和IO。"})
        else: 
            base_inputs["optional"]["decode_gpu_chunk_size"][1]["default"] = 32 
            base_inputs["optional"]["decode_gpu_chunk_size"][1]["tooltip"] = "(To Disk) GPU解码后一次处理并保存多少帧。影响显存和IO。"
        return base_inputs
    
    def _create_error_string_count(self, error_message_text: str, log_message: bool = True) -> tuple:
        # ... (不变) ...
        if log_message:
            print(f"错误: [{self.__class__.NODE_NAME}] {error_message_text}")
        return (f"错误: {error_message_text}", 0)

    def floatprocess_to_disk(self, float_pipe, ref_image, ref_audio, 
                             a_cfg_scale, r_cfg_scale, e_cfg_scale, 
                             fps, emotion, crop_input_image, seed, nfe, 
                             device_override: str = "default",
                             output_subdir_name: str = "float_frames_AIIA",
                             decode_gpu_chunk_size: int = 16):
        
        node_name_log = f"[{self.__class__.NODE_NAME}]"
        print(f"{node_name_log} 流程开始 (输出到磁盘模式)。")
        start_time_process = time.time()
        
        _default_error_tuple = self._create_error_string_count("未知错误 (初始化或预处理失败)", log_message=False)
        return_value = _default_error_tuple

        if float_pipe is None or not hasattr(float_pipe, 'opt') or not hasattr(float_pipe.G, 'decode_latent_into_image'):
             return self._create_error_string_count("float_pipe 无效或不完整", log_message=True)

        processing_device = torch.device(device_override) if device_override != "default" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{node_name_log} 本次运行将在设备上: {processing_device}")

        original_decode_method = None
        _intermediate_wrapper_for_patch = None # 声明以便 finally 块可以引用

        original_opt_rank_backup = getattr(float_pipe.opt, 'rank', None)
        original_opt_fps_backup = getattr(float_pipe.opt, 'fps', None)  
        original_opt_frames_per_gpu_chunk_backup = getattr(float_pipe.opt, 'frames_per_gpu_chunk_for_processing', None)

        output_node_main_dir = folder_paths.get_output_directory()
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        run_unique_folder_name = f"{output_subdir_name}_{timestamp_str}_{int(torch.randint(0,10000,(1,)).item())}"
        frames_output_directory_final = os.path.join(output_node_main_dir, run_unique_folder_name)
        try:
            os.makedirs(frames_output_directory_final, exist_ok=True)
        except Exception as e_mkdir:
            return self._create_error_string_count(f"无法创建输出目录 {frames_output_directory_final}: {e_mkdir}", log_message=True)

        with tempfile.TemporaryDirectory(prefix="aiia_fp_disk_input_") as input_temp_dir:
            try:
                audio_waveform_to_save = ref_audio['waveform'].squeeze(0);
                if audio_waveform_to_save.ndim > 1 and audio_waveform_to_save.shape[0] > 1: audio_waveform_to_save = audio_waveform_to_save[0]
                audio_save_path = os.path.join(input_temp_dir, "temp_audio.wav"); torchaudio.save(audio_save_path, audio_waveform_to_save.cpu(), ref_audio["sample_rate"])
                ref_image_chw = ref_image[0].permute(2, 0, 1).cpu(); image_save_path = os.path.join(input_temp_dir, "temp_ref_image.png")
                vutils.save_image(ref_image_chw, image_save_path, normalize=False)

                if hasattr(float_pipe.opt, 'rank'):
                    float_pipe.opt.rank = processing_device.index if processing_device.type == 'cuda' and processing_device.index is not None else (0 if processing_device.type == 'cuda' else -1)
                if hasattr(float_pipe.opt, 'fps'): float_pipe.opt.fps = float(fps)
                float_pipe.opt.frames_per_gpu_chunk_for_processing = decode_gpu_chunk_size
                print(f"{node_name_log} opt 更新: rank={getattr(float_pipe.opt, 'rank', 'N/A')}, fps={getattr(float_pipe.opt, 'fps', 'N/A')}, frames_chunk_for_processing={getattr(float_pipe.opt, 'frames_per_gpu_chunk_for_processing', 'N/A')}")
                
                model_current_device_before_move = next(float_pipe.G.parameters()).device 
                if model_current_device_before_move != processing_device: float_pipe.G.to(processing_device)
                print(f"{node_name_log} 模型移至: {processing_device}")

                original_decode_method = float_pipe.G.decode_latent_into_image
                
                # _intermediate_wrapper_for_patch_local 需要定义在这里，以便捕获 frames_output_directory_final 和 node_name_log
                # 修改签名以明确接受关键字参数
                def _intermediate_wrapper_for_patch_local(actual_self, *, s_r, s_r_feats, r_d):
                    # actual_self 在这里就是 float_pipe.G
                    # frames_output_directory_final 和 node_name_log 是从外部作用域捕获的 (闭包)
                    return _patched_decode_and_save_to_disk(
                        actual_self,         # 传递 self (模型实例)
                        s_r=s_r,             # 以关键字参数传递
                        s_r_feats=s_r_feats, # 以关键字参数传递
                        r_d=r_d,             # 以关键字参数传递
                        output_frames_dir=frames_output_directory_final, # 我们自己的参数
                        node_name_log_prefix=node_name_log               # 我们自己的参数
                    )
                
                _intermediate_wrapper_for_patch = _intermediate_wrapper_for_patch_local

                float_pipe.G.decode_latent_into_image = types.MethodType(_intermediate_wrapper_for_patch, float_pipe.G)
                print(f"信息: {node_name_log} 已替换 float_pipe.G.decode_latent_into_image 为磁盘保存版本。")
                
                print(f"{node_name_log} 开始运行推理 (帧将保存到磁盘)...")
                _ = float_pipe.run_inference( # 这里会调用被 patch 的方法
                    res_video_path=None, ref_path=image_save_path, audio_path=audio_save_path,
                    a_cfg_scale=a_cfg_scale, r_cfg_scale=r_cfg_scale, e_cfg_scale=e_cfg_scale,
                    emo=None if emotion == "none" else emotion,
                    no_crop=not crop_input_image, nfe=nfe, seed=seed, verbose=False 
                )
                
                actual_saved_frames = getattr(float_pipe.G, '_last_run_saved_frames', 0)
                if hasattr(float_pipe.G, '_last_run_saved_frames'): 
                    delattr(float_pipe.G, '_last_run_saved_frames') 
                
                if actual_saved_frames > 0:
                    print(f"信息: {node_name_log} 推理完成。{actual_saved_frames} 帧已保存到 {frames_output_directory_final}")
                    return_value = (frames_output_directory_final, actual_saved_frames)
                else:
                    print(f"警告: {node_name_log} 推理似乎已完成，但未报告任何已保存的帧。")
                    return_value = self._create_error_string_count("未生成或保存任何帧", log_message=True)
            except Exception as e_proc_inner:
                print(f"错误: {node_name_log} 内部处理错误: {e_proc_inner}"); traceback.print_exc()
                return_value = self._create_error_string_count(f"内部处理错误: {e_proc_inner}", log_message=True)
            finally:
                if original_decode_method and hasattr(float_pipe.G, 'decode_latent_into_image'):
                    current_decode_method = getattr(float_pipe.G, 'decode_latent_into_image', None)
                    # 检查 _intermediate_wrapper_for_patch 是否已被定义 (即 try 块中 patch 代码已执行)
                    if _intermediate_wrapper_for_patch is not None and \
                       hasattr(current_decode_method, '__func__') and \
                       current_decode_method.__func__ is _intermediate_wrapper_for_patch: # 比较函数对象本身
                        float_pipe.G.decode_latent_into_image = original_decode_method
                        print(f"信息: {node_name_log} (finally) 已恢复 float_pipe.G.decode_latent_into_image 为原始方法。")
                    elif current_decode_method is not original_decode_method:
                        print(f"警告: {node_name_log} (finally) decode_latent_into_image 不是预期中的 patch 方法，但也不是原始方法。仍尝试恢复为原始方法。")
                        float_pipe.G.decode_latent_into_image = original_decode_method
                
                if original_opt_rank_backup is not None: float_pipe.opt.rank = original_opt_rank_backup
                if original_opt_fps_backup is not None: float_pipe.opt.fps = original_opt_fps_backup
                if original_opt_frames_per_gpu_chunk_backup is not None : 
                    float_pipe.opt.frames_per_gpu_chunk_for_processing = original_opt_frames_per_gpu_chunk_backup
                elif hasattr(float_pipe.opt, 'frames_per_gpu_chunk_for_processing'): 
                    try:
                        del float_pipe.opt.frames_per_gpu_chunk_for_processing
                    except AttributeError:
                        pass
                
                current_g_device_after_proc = next(float_pipe.G.parameters()).device
                if current_g_device_after_proc.type == 'cuda':
                    try: 
                        float_pipe.G.to(torch.device("cpu"))
                        torch.cuda.empty_cache()
                    except Exception as e_to_cpu: 
                        print(f"{node_name_log} (finally) 模型移至CPU或清空缓存时出错: {e_to_cpu}")
        
        end_time_process = time.time()
        print(f"{node_name_log} 方法总执行耗时: {end_time_process - start_time_process:.2f} 秒。")
        return return_value

# --- ComfyUI 节点注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_FloatProcess_InMemory": AIIA_FloatProcess_InMemory,
    "AIIA_FloatProcess_ToDisk": AIIA_FloatProcess_ToDisk,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_FloatProcess_InMemory": "Float Process (AIIA In-Memory)",
    "AIIA_FloatProcess_ToDisk": "Float Process (AIIA To-Disk for Long Audio)",
}
print(f"--- AIIA FLOAT Process Nodes (InMemory & ToDisk) Loaded ---")

# --- END OF FILE aiia_float_nodes.py ---