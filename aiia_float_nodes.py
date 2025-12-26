# --- START OF FILE aiia_float_nodes.py (REVERTED TO STABLE MOTION LOGIC) ---

import torch
import os
import tempfile
import torchaudio
import torchvision.utils as vutils
import numpy as np
import folder_paths
import time
import types
from comfy.utils import ProgressBar
from PIL import Image
import traceback

# ----------------------------------------------------------------------------------
# 核心补丁：仅针对音频情感编码器的 OOM 修复 (带上下文重叠)
# ----------------------------------------------------------------------------------

def _patched_predict_emotion_chunked(self, audio_input):
    """
    仅分块处理音频，防止 wav2vec2 在计算 Attention 时爆显存。
    这是安全的，因为它不影响后续的视频生成逻辑。
    """
    MODEL_STRIDE = 320
    CHUNK_SEC = 30
    OVERLAP_SEC = 2
    
    chunk_samples = CHUNK_SEC * 16000
    overlap_samples = OVERLAP_SEC * 16000
    
    total_len = audio_input.shape[1]
    
    if total_len <= chunk_samples + overlap_samples:
        return self.wav2vec2_for_emotion(audio_input).logits
        
    outputs = []
    print(f"Info: [AIIA] Processing Long Audio in chunks to prevent OOM...")
    
    for start_idx in range(0, total_len, chunk_samples):
        end_idx = min(start_idx + chunk_samples, total_len)
        expanded_start = max(0, start_idx - overlap_samples)
        expanded_end = min(total_len, end_idx + overlap_samples)
        
        chunk = audio_input[:, expanded_start : expanded_end]
        if chunk.shape[1] == 0: continue
        
        with torch.no_grad():
            out = self.wav2vec2_for_emotion(chunk).logits
        
        # 裁剪掉重叠部分
        input_offset_samples = start_idx - expanded_start
        input_core_len_samples = end_idx - start_idx
        start_frame = input_offset_samples // MODEL_STRIDE
        keep_frames = input_core_len_samples // MODEL_STRIDE
        
        if start_frame < out.shape[1]:
             end_frame = min(start_frame + keep_frames, out.shape[1])
             outputs.append(out[:, start_frame:end_frame, :])
        
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
        return {"required": {
            "float_pipe": ("FLOAT_PIPE",),
            "ref_image": ("IMAGE",),
            "ref_audio": ("AUDIO",),
            "a_cfg_scale": ("FLOAT", {"default": 2.0,"min": 0.0, "max": 10.0, "step": 0.1}),
            "r_cfg_scale": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.1}),
            "e_cfg_scale": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.1}),
            "fps": ("FLOAT", {"default": 25.0, "min":1.0, "max": 60.0, "step": 0.5}),
            "emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none"}),
            "crop_input_image": ("BOOLEAN",{"default":False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "nfe": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
        }}

    def floatprocess_in_memory(self, float_pipe, ref_image, ref_audio, **kwargs):
        G = float_pipe.G
        device = next(G.parameters()).device
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")
            waveform = ref_audio['waveform'].squeeze(0)
            if waveform.shape[0] > 1: waveform = waveform[0:1, :]
            torchaudio.save(audio_path, waveform.cpu(), ref_audio["sample_rate"], encoding="PCM_S", bits_per_sample=16)
            
            image_path = os.path.join(temp_dir, "ref.png")
            vutils.save_image(ref_image[0].permute(2, 0, 1).cpu(), image_path, normalize=False)

            # 备份原始方法
            orig_predict = None
            if hasattr(G, 'emotion_encoder'):
                orig_predict = G.emotion_encoder.predict_emotion
                G.emotion_encoder.predict_emotion = types.MethodType(_patched_predict_emotion_chunked, G.emotion_encoder)

            try:
                float_pipe.opt.fps = float(kwargs.get("fps"))
                # 直接调用原始推理，不干扰解码过程
                images = float_pipe.run_inference(
                    res_video_path=None, ref_path=image_path, audio_path=audio_path,
                    a_cfg_scale=kwargs.get("a_cfg_scale"), r_cfg_scale=kwargs.get("r_cfg_scale"), 
                    e_cfg_scale=kwargs.get("e_cfg_scale"), emo=None if kwargs.get("emotion") == "none" else kwargs.get("emotion"),
                    no_crop=not kwargs.get("crop_input_image"), nfe=kwargs.get("nfe"), seed=kwargs.get("seed"), verbose=False
                )
                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(images.astype(np.float32))
                return (images,)
            finally:
                if orig_predict: G.emotion_encoder.predict_emotion = orig_predict
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
        output_dir = os.path.join(folder_paths.get_output_directory(), f"{kwargs.get('output_subdir_name')}_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)

        G = float_pipe.G
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")
            waveform = ref_audio['waveform'].squeeze(0)
            if waveform.shape[0] > 1: waveform = waveform[0:1, :]
            torchaudio.save(audio_path, waveform.cpu(), ref_audio["sample_rate"], encoding="PCM_S", bits_per_sample=16)
            
            image_path = os.path.join(temp_dir, "ref.png")
            vutils.save_image(ref_image[0].permute(2, 0, 1).cpu(), image_path, normalize=False)

            orig_predict = None
            if hasattr(G, 'emotion_encoder'):
                orig_predict = G.emotion_encoder.predict_emotion
                G.emotion_encoder.predict_emotion = types.MethodType(_patched_predict_emotion_chunked, G.emotion_encoder)

            try:
                float_pipe.opt.fps = float(kwargs.get("fps"))
                # 运行完整推理
                images = float_pipe.run_inference(
                    res_video_path=None, ref_path=image_path, audio_path=audio_path,
                    a_cfg_scale=kwargs.get("a_cfg_scale"), r_cfg_scale=kwargs.get("r_cfg_scale"), 
                    e_cfg_scale=kwargs.get("e_cfg_scale"), emo=None if kwargs.get("emotion") == "none" else kwargs.get("emotion"),
                    no_crop=not kwargs.get("crop_input_image"), nfe=kwargs.get("nfe"), seed=kwargs.get("seed"), verbose=False
                )
                
                # 即使是 To-Disk 模式，我们也先通过原始方法拿到完整 Tensor，然后存盘
                # 这样可以保证运动完全正确。如果 Tensor 太大导致系统内存崩，我们再考虑下一步。
                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(images.astype(np.float32))
                
                print(f"Info: [AIIA] Saving {len(images)} frames to disk...")
                for i, img_tensor in enumerate(images):
                    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_np).save(os.path.join(output_dir, f"frame_{i:06d}.png"))
                
                return (output_dir, len(images))
            finally:
                if orig_predict: G.emotion_encoder.predict_emotion = orig_predict
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
print(f"--- AIIA FLOAT Process Nodes (Motion Stabilized) Loaded ---")