# --- START OF FILE aiia_float_nodes.py (FIXED for Motion & Audio Continuity) ---

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
from tqdm import tqdm

# ----------------------------------------------------------------------------------
# 辅助函数：通用的分块包装逻辑
# ----------------------------------------------------------------------------------

def _patched_decode_chunked_in_memory(self, s_r, s_r_feats, r_d):
    """
    In-Memory 模式的解码补丁：调用原始解码方法，但对 r_d 进行分块。
    """
    orig_decode = getattr(self, '_original_decode', None)
    if orig_decode is None:
        raise RuntimeError("[AIIA] Original decode method not found on model instance.")

    T_total = r_d.shape[1]
    # 使用之前设置的块大小
    chunk_size = getattr(self.opt, 'decode_gpu_chunk_size', 32)
    
    all_d_hat = []
    comfy_pbar = ProgressBar(T_total)
    
    print(f"Info: [AIIA] Decoding {T_total} frames in chunks of {chunk_size} (In-Memory)...")

    for i in range(0, T_total, chunk_size):
        end_idx = min(i + chunk_size, T_total)
        r_d_chunk = r_d[:, i:end_idx]
        
        # 调用原始方法处理这个块
        with torch.no_grad():
            res = orig_decode(s_r, s_r_feats, r_d_chunk)
        
        # 提取结果 (处理可能是 dict 或 tensor 的情况)
        d_hat_chunk = res['d_hat'] if isinstance(res, dict) else res
        
        # 转移到 CPU 并存储
        all_d_hat.append(d_hat_chunk.cpu())
        
        # 更新进度
        comfy_pbar.update(end_idx - i)
        torch.cuda.empty_cache()

    # 合并结果
    final_d_hat = torch.cat(all_d_hat, dim=0) # 原始方法返回通常是 (T, C, H, W)
    return {'d_hat': final_d_hat}


def _patched_decode_chunked_to_disk(self, s_r, s_r_feats, r_d):
    """
    To-Disk 模式的解码补丁：调用原始解码方法，分块处理并立即存盘。
    """
    orig_decode = getattr(self, '_original_decode', None)
    output_dir = getattr(self, '_aiia_output_dir', None)
    if orig_decode is None or output_dir is None:
        raise RuntimeError("[AIIA] Required attributes for to-disk patching not found.")

    T_total = r_d.shape[1]
    chunk_size = getattr(self.opt, 'frames_per_gpu_chunk_for_processing', 16)
    
    comfy_pbar = ProgressBar(T_total)
    saved_count = 0
    
    print(f"Info: [AIIA] Decoding {T_total} frames in chunks of {chunk_size} (To-Disk)...")

    for i in range(0, T_total, chunk_size):
        end_idx = min(i + chunk_size, T_total)
        r_d_chunk = r_d[:, i:end_idx]
        
        with torch.no_grad():
            res = orig_decode(s_r, s_r_feats, r_d_chunk)
        
        d_hat_chunk = res['d_hat'] if isinstance(res, dict) else res
        
        # 处理并保存该块中的每一帧
        # d_hat_chunk 期望形状: (T_chunk, C, H, W)，值域通常 [-1, 1] 或 [0, 1]
        chunk_cpu = d_hat_chunk.cpu()
        # 转换为 HWC 并映射到 [0, 255]
        # 注意：FLOAT 原始 decode 出来可能还在 [-1, 1]，我们需要根据其值域处理
        # 这里的处理逻辑参考原始代码的 clamp(-1,1)
        chunk_images = ((chunk_cpu.permute(0, 2, 3, 1).clamp(-1, 1) + 1.0) / 2.0 * 255).byte().numpy()
        
        for frame_np in chunk_images:
            filename = f"frame_{saved_count:06d}.png"
            filepath = os.path.join(output_dir, filename)
            Image.fromarray(frame_np).save(filepath)
            saved_count += 1
            
        comfy_pbar.update(end_idx - i)
        del d_hat_chunk, chunk_cpu, res
        torch.cuda.empty_cache()

    self._last_run_saved_frames = saved_count
    # 返回一个空的占位符，因为帧已经存盘了
    return {'d_hat': torch.empty((1, 0, 3, 64, 64), device='cpu')}


def _patched_predict_emotion_chunked(self, audio_input):
    """
    音频情感分块补丁：调用原始预测方法，增加重叠上下文。
    """
    orig_predict = getattr(self, '_original_predict_emotion', None)
    if orig_predict is None:
        return self.wav2vec2_for_emotion(audio_input).logits # 回退

    MODEL_STRIDE = 320
    CHUNK_SEC = 30
    OVERLAP_SEC = 2
    
    chunk_samples = CHUNK_SEC * 16000
    overlap_samples = OVERLAP_SEC * 16000
    
    # 对齐步长
    chunk_samples = (chunk_samples // MODEL_STRIDE) * MODEL_STRIDE
    overlap_samples = (overlap_samples // MODEL_STRIDE) * MODEL_STRIDE
    
    total_len = audio_input.shape[1]
    
    if total_len <= chunk_samples:
        return orig_predict(audio_input)
        
    outputs = []
    print(f"Info: [AIIA] Processing Audio Emotion via Original Method in chunks...")
    
    for start_idx in range(0, total_len, chunk_samples):
        end_idx = min(start_idx + chunk_samples, total_len)
        
        expanded_start = max(0, start_idx - overlap_samples)
        expanded_end = min(total_len, end_idx + overlap_samples)
        
        chunk = audio_input[:, expanded_start : expanded_end]
        if chunk.shape[1] == 0: continue
        
        # 调用原始方法（保留所有内部处理）
        with torch.no_grad():
            out = orig_predict(chunk)
        
        # 计算裁剪范围
        input_offset_samples = start_idx - expanded_start
        input_core_len_samples = end_idx - start_idx
        
        start_frame = input_offset_samples // MODEL_STRIDE
        keep_frames = input_core_len_samples // MODEL_STRIDE
        
        if start_frame < out.shape[1]:
             end_frame = min(start_frame + keep_frames, out.shape[1])
             core_out = out[:, start_frame:end_frame, :]
             outputs.append(core_out)
        
        torch.cuda.empty_cache()
        
    return torch.cat(outputs, dim=1)


# ----------------------------------------------------------------------------------
# 节点类
# ----------------------------------------------------------------------------------

class AIIA_FloatProcess_InMemory:
    NODE_NAME = "AIIA Float Process (In-Memory Output)"
    CATEGORY = "AIIA/FLOAT"
    FUNCTION = "floatprocess_in_memory"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"float_pipe": ("FLOAT_PIPE",),"ref_image": ("IMAGE",),"ref_audio": ("AUDIO",),"a_cfg_scale": ("FLOAT", {"default": 2.0,"min": 0.0, "max": 10.0, "step": 0.1}),"r_cfg_scale": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.1}),"e_cfg_scale": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.1}),"fps": ("FLOAT", {"default": 25.0, "min":1.0, "max": 60.0, "step": 0.5}),"emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none"}),"crop_input_image": ("BOOLEAN",{"default":False},),"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),"nfe": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}), },"optional": {"device_override": (["default", "cuda", "cpu"], {"default": "default"}), "decode_gpu_chunk_size": ("INT", {"default": 32, "min":1, "max":128, "step":1}),}}

    def floatprocess_in_memory(self, float_pipe, ref_image, ref_audio, **kwargs):
        node_name_log = f"[{self.__class__.NODE_NAME}]"
        processing_device = torch.device(kwargs.get("device_override", "cuda") if kwargs.get("device_override") != "default" else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 1. 准备临时文件
        with tempfile.TemporaryDirectory(prefix="aiia_fp_mem_") as temp_dir:
            # 音频保存 (固定采样率 16k)
            audio_save_path = os.path.join(temp_dir, "audio.wav")
            waveform = ref_audio['waveform'].squeeze(0)
            if waveform.shape[0] > 1: waveform = waveform[0:1, :]
            torchaudio.save(audio_save_path, waveform.cpu(), ref_audio["sample_rate"], encoding="PCM_S", bits_per_sample=16)
            
            # 图片保存
            image_save_path = os.path.join(temp_dir, "ref.png")
            vutils.save_image(ref_image[0].permute(2, 0, 1).cpu(), image_save_path, normalize=False)

            # 2. 应用 Patch
            G = float_pipe.G
            G.to(processing_device)
            
            # 备份并替换解码方法
            G._original_decode = G.decode_latent_into_image
            G.decode_latent_into_image = types.MethodType(_patched_decode_chunked_in_memory, G)
            
            # 备份并替换情感预测方法
            if hasattr(G, 'emotion_encoder'):
                G.emotion_encoder._original_predict_emotion = G.emotion_encoder.predict_emotion
                G.emotion_encoder.predict_emotion = types.MethodType(_patched_predict_emotion_chunked, G.emotion_encoder)

            try:
                # 3. 运行推理
                # float_pipe.opt 需要更新
                float_pipe.opt.decode_gpu_chunk_size = kwargs.get("decode_gpu_chunk_size", 32)
                float_pipe.opt.fps = float(kwargs.get("fps", 25.0))
                
                print(f"{node_name_log} 推理开始...")
                images = float_pipe.run_inference(
                    res_video_path=None, ref_path=image_save_path, audio_path=audio_save_path,
                    a_cfg_scale=kwargs.get("a_cfg_scale"), r_cfg_scale=kwargs.get("r_cfg_scale"), 
                    e_cfg_scale=kwargs.get("e_cfg_scale"), emo=None if kwargs.get("emotion") == "none" else kwargs.get("emotion"),
                    no_crop=not kwargs.get("crop_input_image"), nfe=kwargs.get("nfe"), seed=kwargs.get("seed"), verbose=False
                )
                
                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(images.astype(np.float32))
                return (images,)

            finally:
                # 4. 恢复原始方法
                if hasattr(G, '_original_decode'):
                    G.decode_latent_into_image = G._original_decode
                if hasattr(G, 'emotion_encoder') and hasattr(G.emotion_encoder, '_original_predict_emotion'):
                    G.emotion_encoder.predict_emotion = G.emotion_encoder._original_predict_emotion
                G.to(torch.device("cpu"))
                torch.cuda.empty_cache()


class AIIA_FloatProcess_ToDisk:
    NODE_NAME = "AIIA Float Process (To Disk)"
    CATEGORY = "AIIA/FLOAT"
    FUNCTION = "floatprocess_to_disk"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("frames_output_directory", "saved_frame_count")

    @classmethod
    def INPUT_TYPES(cls):
        base = AIIA_FloatProcess_InMemory.INPUT_TYPES()
        base["required"]["output_subdir_name"] = ("STRING", {"default": "float_frames_AIIA"})
        return base

    def floatprocess_to_disk(self, float_pipe, ref_image, ref_audio, **kwargs):
        node_name_log = f"[{self.__class__.NODE_NAME}]"
        processing_device = torch.device(kwargs.get("device_override", "cuda") if kwargs.get("device_override") != "default" else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        output_dir = os.path.join(folder_paths.get_output_directory(), f"{kwargs.get('output_subdir_name')}_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="aiia_fp_disk_") as temp_dir:
            audio_save_path = os.path.join(temp_dir, "audio.wav")
            waveform = ref_audio['waveform'].squeeze(0)
            if waveform.shape[0] > 1: waveform = waveform[0:1, :]
            torchaudio.save(audio_save_path, waveform.cpu(), ref_audio["sample_rate"], encoding="PCM_S", bits_per_sample=16)
            
            image_save_path = os.path.join(temp_dir, "ref.png")
            vutils.save_image(ref_image[0].permute(2, 0, 1).cpu(), image_save_path, normalize=False)

            G = float_pipe.G
            G.to(processing_device)
            G._aiia_output_dir = output_dir
            G._original_decode = G.decode_latent_into_image
            G.decode_latent_into_image = types.MethodType(_patched_decode_chunked_to_disk, G)
            
            if hasattr(G, 'emotion_encoder'):
                G.emotion_encoder._original_predict_emotion = G.emotion_encoder.predict_emotion
                G.emotion_encoder.predict_emotion = types.MethodType(_patched_predict_emotion_chunked, G.emotion_encoder)

            try:
                float_pipe.opt.frames_per_gpu_chunk_for_processing = kwargs.get("decode_gpu_chunk_size", 32)
                float_pipe.opt.fps = float(kwargs.get("fps", 25.0))
                
                print(f"{node_name_log} 推理开始 (To-Disk)...")
                _ = float_pipe.run_inference(
                    res_video_path=None, ref_path=image_save_path, audio_path=audio_save_path,
                    a_cfg_scale=kwargs.get("a_cfg_scale"), r_cfg_scale=kwargs.get("r_cfg_scale"), 
                    e_cfg_scale=kwargs.get("e_cfg_scale"), emo=None if kwargs.get("emotion") == "none" else kwargs.get("emotion"),
                    no_crop=not kwargs.get("crop_input_image"), nfe=kwargs.get("nfe"), seed=kwargs.get("seed"), verbose=False
                )
                
                saved_count = getattr(G, '_last_run_saved_frames', 0)
                return (output_dir, saved_count)

            finally:
                if hasattr(G, '_original_decode'):
                    G.decode_latent_into_image = G._original_decode
                if hasattr(G, 'emotion_encoder') and hasattr(G.emotion_encoder, '_original_predict_emotion'):
                    G.emotion_encoder.predict_emotion = G.emotion_encoder._original_predict_emotion
                G.to(torch.device("cpu"))
                torch.cuda.empty_cache()

# --- 注册 ---
NODE_CLASS_MAPPINGS = {
    "AIIA_FloatProcess_InMemory": AIIA_FloatProcess_InMemory,
    "AIIA_FloatProcess_ToDisk": AIIA_FloatProcess_ToDisk,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_FloatProcess_InMemory": "Float Process (AIIA In-Memory)",
    "AIIA_FloatProcess_ToDisk": "Float Process (AIIA To-Disk for Long Audio)",
}
print(f"--- AIIA FLOAT Process Nodes (Safe Chunking) Loaded ---")
