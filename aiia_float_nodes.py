# --- START OF FILE aiia_float_nodes.py (GENUINE ORIGINAL VERSION) ---

import torch
import os
import tempfile
import torchaudio
import torchvision.utils as vutils
import numpy as np
import folder_paths
import time
from PIL import Image
import traceback

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
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")
            waveform = ref_audio['waveform'].squeeze(0)
            if waveform.shape[0] > 1: waveform = waveform[0:1, :]
            torchaudio.save(audio_path, waveform.cpu(), ref_audio["sample_rate"], encoding="PCM_S", bits_per_sample=16)
            
            image_path = os.path.join(temp_dir, "ref.png")
            vutils.save_image(ref_image[0].permute(2, 0, 1).cpu(), image_path, normalize=False)

            try:
                float_pipe.opt.fps = float(kwargs.get("fps"))
                # 直接调用原始推理，不进行任何动态方法替换(Patch)
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

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.wav")
            waveform = ref_audio['waveform'].squeeze(0)
            if waveform.shape[0] > 1: waveform = waveform[0:1, :]
            torchaudio.save(audio_path, waveform.cpu(), ref_audio["sample_rate"], encoding="PCM_S", bits_per_sample=16)
            
            image_path = os.path.join(temp_dir, "ref.png")
            vutils.save_image(ref_image[0].permute(2, 0, 1).cpu(), image_path, normalize=False)

            try:
                float_pipe.opt.fps = float(kwargs.get("fps"))
                images = float_pipe.run_inference(
                    res_video_path=None, ref_path=image_path, audio_path=audio_path,
                    a_cfg_scale=kwargs.get("a_cfg_scale"), r_cfg_scale=kwargs.get("r_cfg_scale"), 
                    e_cfg_scale=kwargs.get("e_cfg_scale"), emo=None if kwargs.get("emotion") == "none" else kwargs.get("emotion"),
                    no_crop=not kwargs.get("crop_input_image"), nfe=kwargs.get("nfe"), seed=kwargs.get("seed"), verbose=False
                )
                
                if not isinstance(images, torch.Tensor):
                    images = torch.from_numpy(images.astype(np.float32))
                
                for i, img_tensor in enumerate(images):
                    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(img_np).save(os.path.join(output_dir, f"frame_{i:06d}.png"))
                
                return (output_dir, len(images))
            finally:
                torch.cuda.empty_cache()

NODE_CLASS_MAPPINGS = {
    "AIIA_FloatProcess_InMemory": AIIA_FloatProcess_InMemory,
    "AIIA_FloatProcess_ToDisk": AIIA_FloatProcess_ToDisk,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_FloatProcess_InMemory": "Float Process (AIIA In-Memory)",
    "AIIA_FloatProcess_ToDisk": "Float Process (AIIA To-Disk for Long Audio)",
}
print(f"--- AIIA FLOAT Process Nodes (Genuine Original) Loaded ---")
