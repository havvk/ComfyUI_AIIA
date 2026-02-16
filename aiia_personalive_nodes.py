import os
import sys
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection
import mediapipe as mp
import folder_paths
from huggingface_hub import snapshot_download
import time

from .personalive.models.unet_2d_condition import UNet2DConditionModel
from .personalive.models.unet_3d import UNet3DConditionModel
from .personalive.models.motion_encoder.encoder import MotEncoder
from .personalive.liveportrait.motion_extractor import MotionExtractor
from .personalive.models.pose_guider import PoseGuider
from .personalive.scheduler.scheduler_ddim import DDIMScheduler
from .personalive.pipelines.pipeline_pose2vid import Pose2VideoPipeline
from .personalive.utils.util import crop_face
from diffusers.utils.import_utils import is_xformers_available

def get_folder_list():
    base_dir = folder_paths.models_dir
    if not os.path.exists(base_dir):
        return ["persona_live"]
        
    candidates = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            if os.path.exists(os.path.join(full_path, "pretrained_weights")):
                candidates.append(name)
    
    if "persona_live" not in candidates:
        candidates.append("persona_live") 
        
    return sorted(candidates)

def download_models_if_missing(root_dir):
    """Auto-download models from HuggingFace if they don't exist."""
    base_model_path = os.path.join(root_dir, "sd-image-variations-diffusers")
    vae_path = os.path.join(root_dir, "sd-vae-ft-mse")
    personalive_path = os.path.join(root_dir, "persona_live")
    
    models_to_download = [
        {
            "repo_id": "lambdalabs/sd-image-variations-diffusers",
            "local_dir": base_model_path,
            "name": "Base Model (sd-image-variations-diffusers)"
        },
        {
            "repo_id": "stabilityai/sd-vae-ft-mse",
            "local_dir": vae_path,
            "name": "VAE (sd-vae-ft-mse)"
        },
        {
            "repo_id": "huaichang/PersonaLive",
            "local_dir": personalive_path,
            "name": "PersonaLive Weights"
        }
    ]
    
    for model_info in models_to_download:
        if not os.path.exists(model_info["local_dir"]) or not os.listdir(model_info["local_dir"]):
            print(f"\n{'='*60}")
            print(f"Downloading {model_info['name']}...")
            print(f"From: {model_info['repo_id']}")
            print(f"To: {model_info['local_dir']}")
            print(f"This may take a while (several GB)...")
            print(f"{ '='*60}\n")
            
            try:
                snapshot_download(
                    repo_id=model_info["repo_id"],
                    local_dir=model_info["local_dir"],
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print(f"\n✓ Successfully downloaded {model_info['name']}\n")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {model_info['name']} from {model_info['repo_id']}: {e}\n"
                    f"Please check your internet connection or download manually."
                )
        else:
            print(f"✓ {model_info['name']} already exists at {model_info['local_dir']}")

class AIIA_PersonaLive_CheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_dir": (get_folder_list(), ),
            }
        }

    RETURN_TYPES = ("PERSONALIVE_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "AIIA/PersonaLive"

    def load_checkpoint(self, model_dir):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight_dtype = torch.float16 if device == "cuda" else torch.float32
        
        root_dir = os.path.join(folder_paths.models_dir, model_dir)
        download_models_if_missing(root_dir)
        
        base_model_path = os.path.join(root_dir, "sd-image-variations-diffusers")
        vae_path = os.path.join(root_dir, "sd-vae-ft-mse")
        personalive_path = os.path.join(root_dir, "persona_live")
        image_encoder_path = os.path.join(base_model_path, "image_encoder")

        try:
            vae_model = AutoencoderKL.from_pretrained(vae_path).to(device, dtype=weight_dtype)
        except Exception as e:
             vae_model = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae").to(device, dtype=weight_dtype)


        reference_unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
        ).to(device=device, dtype=weight_dtype)

        unet_additional_kwargs = {
            "use_inflated_groupnorm": True,
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "use_motion_module": True,
            "motion_module_resolutions": [1, 2, 4, 8],
            "motion_module_mid_block": True,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "cross_attention_dim": 16,
                "attention_block_types": ["Spatial_Cross", "Spatial_Cross"],
                "temporal_position_encoding": False,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
            },
            "use_temporal_module": True,
            "temporal_module_type": "Vanilla",
            "temporal_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
            }
        }

        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs=unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)

        motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device).eval()
        pose_guider = PoseGuider().to(device=device, dtype=weight_dtype)
        pose_encoder = MotionExtractor(num_kp=21).to(device=device, dtype=weight_dtype).eval()

        image_enc = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path
        ).to(dtype=weight_dtype, device=device)

        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.02,
            beta_schedule="scaled_linear",
            clip_sample=False,
            steps_offset=1,
            prediction_type="epsilon",
            timestep_spacing="trailing"
        )

        print(f"Loading weights from {personalive_path}")
        
        def load_w(model, filename, strict=True):
             p = os.path.join(personalive_path, "pretrained_weights", "personalive", filename)
             if os.path.exists(p):
                 print(f"Loading {filename} from {p}")
                 model.load_state_dict(torch.load(p, map_location="cpu"), strict=strict)
             else:
                 print(f"WARNING: Could not find {filename} in {personalive_path}")

        load_w(denoising_unet, "denoising_unet.pth", strict=False)
        load_w(reference_unet, "reference_unet.pth", strict=True)
        load_w(motion_encoder, "motion_encoder.pth", strict=True)
        load_w(pose_guider, "pose_guider.pth", strict=True)
        load_w(denoising_unet, "temporal_module.pth", strict=False) 
        load_w(pose_encoder, "motion_extractor.pth", strict=False)

        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()

        pipe = Pose2VideoPipeline(
            vae=vae_model,
            image_encoder=image_enc,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            motion_encoder=motion_encoder,
            pose_encoder=pose_encoder,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        pipe = pipe.to(device)

        return (pipe,)

# --- Shared Helpers ---

def _prepare_ref_inputs(ref_image, width, height, face_mesh):
    ref_pil = Image.fromarray(np.clip(255. * ref_image[0].cpu().numpy(), 0, 255).astype(np.uint8))
    ref_input = ref_pil.resize((width, height))
    try:
        ref_patch = crop_face(ref_pil, face_mesh, scale=1.1)
        ref_face = Image.fromarray(ref_patch).convert("RGB")
    except Exception as e:
        print(f"Ref face detection failed: {e}. Using full image.")
        ref_face = ref_input
    return ref_input, ref_face

def _prepare_chunk_inputs(driving_image_tensor_batch, width, height, face_mesh):
    # driving_image_tensor_batch: (B, H, W, C)
    ori_pose_images = []
    dri_faces = []
    
    num_frames = driving_image_tensor_batch.shape[0]
    
    for i in range(num_frames):
        frame_tensor = driving_image_tensor_batch[i]
        frame_pil = Image.fromarray(np.clip(255. * frame_tensor.cpu().numpy(), 0, 255).astype(np.uint8))
        frame_resized = frame_pil.resize((width, height))
        ori_pose_images.append(frame_resized)
        
        try:
            frame_patch = crop_face(frame_pil, face_mesh, scale=1.1)
            frame_face = Image.fromarray(frame_patch).convert("RGB")
        except Exception as e:
            frame_face = frame_resized
        
        dri_faces.append(frame_face)
        
    return ori_pose_images, dri_faces

def _run_inference(pipe, chunk_ori, chunk_dri, input_ref, input_ref_face, width, height, guidance_scale, generator, temporal_window_size=4):
    current_len = len(chunk_ori)
    remainder = current_len % temporal_window_size
    pad_frames = 0
    if remainder != 0:
         pad_frames = temporal_window_size - remainder
         chunk_ori.extend([chunk_ori[-1]] * pad_frames)
         chunk_dri.extend([chunk_dri[-1]] * pad_frames)
         current_len += pad_frames
    
    if current_len == 0: return None

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    result = pipe(
        chunk_ori,
        input_ref,
        chunk_dri,
        input_ref_face,
        width,
        height,
        current_len,
        num_inference_steps=4,
        guidance_scale=guidance_scale,
        generator=generator,
        temporal_window_size=temporal_window_size,
        temporal_adaptive_step=4,
    )
    
    gen_video = result.videos # (B, C, F, H, W)
    
    if isinstance(gen_video, np.ndarray):
        gen_video = torch.from_numpy(gen_video)
        
    if len(gen_video.shape) == 5:
        gen_video = gen_video.squeeze(0) # (C, F, H, W)
    
    if pad_frames > 0:
        gen_video = gen_video[:, :-pad_frames, :, :]
        
    gen_video = gen_video.permute(1, 2, 3, 0) # (F, H, W, C)
    return gen_video

class AIIA_PersonaLive_PhotoSampler_InMemory:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PERSONALIVE_PIPE",),
                "ref_image": ("IMAGE",),
                "driving_image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "chunk_size": ("INT", {"default": 16, "min": 4, "max": 128, "step": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("generated_image",)
    FUNCTION = "generate"
    CATEGORY = "AIIA/PersonaLive"

    def generate(self, pipe, ref_image, driving_image, width, height, guidance_scale, seed, chunk_size):
        device = pipe.device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        input_ref, input_ref_face = _prepare_ref_inputs(ref_image, width, height, face_mesh)

        if len(driving_image.shape) == 3:
             driving_image = driving_image.unsqueeze(0)
        
        num_frames = driving_image.shape[0]
        temporal_window_size = 4
        chunk_size = (chunk_size // temporal_window_size) * temporal_window_size
        if chunk_size < temporal_window_size: chunk_size = temporal_window_size
        
        print(f"Processing {num_frames} frames (In-Memory). Chunk size: {chunk_size}")

        all_generated_frames = []

        for start_idx in range(0, num_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, num_frames)
            
            chunk_tensor = driving_image[start_idx:end_idx]
            chunk_ori, chunk_dri = _prepare_chunk_inputs(chunk_tensor, width, height, face_mesh)
            
            print(f"Generating chunk {start_idx}-{end_idx}...")
            
            gen_video = _run_inference(pipe, chunk_ori, chunk_dri, input_ref, input_ref_face, width, height, guidance_scale, generator)
            
            if gen_video is not None:
                all_generated_frames.append(gen_video)
        
        if not all_generated_frames:
             return (torch.zeros((1, height, width, 3)),)

        final_video = torch.cat(all_generated_frames, dim=0)
        return (final_video,)

class AIIA_PersonaLive_PhotoSampler_ToDisk:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PERSONALIVE_PIPE",),
                "ref_image": ("IMAGE",),
                "driving_image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "chunk_size": ("INT", {"default": 16, "min": 4, "max": 128, "step": 4}),
                "output_subdir_name": ("STRING", {"default": "PersonaLive_Frames"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("frames_directory", "frame_count")
    FUNCTION = "generate"
    CATEGORY = "AIIA/PersonaLive"

    def generate(self, pipe, ref_image, driving_image, width, height, guidance_scale, seed, chunk_size, output_subdir_name):
        device = pipe.device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

        input_ref, input_ref_face = _prepare_ref_inputs(ref_image, width, height, face_mesh)

        if len(driving_image.shape) == 3:
             driving_image = driving_image.unsqueeze(0)
        
        num_frames = driving_image.shape[0]
        temporal_window_size = 4
        chunk_size = (chunk_size // temporal_window_size) * temporal_window_size
        if chunk_size < temporal_window_size: chunk_size = temporal_window_size
        
        output_node_main_dir = folder_paths.get_output_directory()
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
        run_unique_folder_name = f"{output_subdir_name}_{timestamp_str}_{int(torch.randint(0,10000,(1,)).item())}"
        frames_output_directory = os.path.join(output_node_main_dir, run_unique_folder_name)
        os.makedirs(frames_output_directory, exist_ok=True)
        # 写入标记文件，供下游 Video Combine 的 cleanup_frames 安全识别
        from pathlib import Path
        Path(frames_output_directory, ".aiia_temp").touch()
        
        print(f"Processing {num_frames} frames (To-Disk). Output: {frames_output_directory}")

        saved_frame_count = 0

        for start_idx in range(0, num_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, num_frames)
            
            chunk_tensor = driving_image[start_idx:end_idx]
            chunk_ori, chunk_dri = _prepare_chunk_inputs(chunk_tensor, width, height, face_mesh)
            
            print(f"Generating chunk {start_idx}-{end_idx}...")
            
            # gen_video is (F, H, W, C) tensor
            gen_video = _run_inference(pipe, chunk_ori, chunk_dri, input_ref, input_ref_face, width, height, guidance_scale, generator)
            
            if gen_video is not None:
                # Save frames
                for i in range(gen_video.shape[0]):
                    frame = gen_video[i] # (H, W, C)
                    frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                    filename = f"frame_{saved_frame_count:08d}.png"
                    filepath = os.path.join(frames_output_directory, filename)
                    Image.fromarray(frame_np).save(filepath)
                    saved_frame_count += 1
                
                # Cleanup memory
                del gen_video
                import gc
                gc.collect()

        return (frames_output_directory, saved_frame_count)

NODE_CLASS_MAPPINGS = {
    "PersonaLiveCheckpointLoader": AIIA_PersonaLive_CheckpointLoader,
    "PersonaLivePhotoSampler": AIIA_PersonaLive_PhotoSampler_InMemory,
    "AIIA_PersonaLive_PhotoSampler_InMemory": AIIA_PersonaLive_PhotoSampler_InMemory,
    "AIIA_PersonaLive_PhotoSampler_ToDisk": AIIA_PersonaLive_PhotoSampler_ToDisk
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PersonaLiveCheckpointLoader": "PersonaLive Checkpoint Loader",
    "PersonaLivePhotoSampler": "PersonaLive Photo Sampler (AIIA In-Memory)",
    "AIIA_PersonaLive_PhotoSampler_InMemory": "PersonaLive Photo Sampler (AIIA In-Memory)",
    "AIIA_PersonaLive_PhotoSampler_ToDisk": "PersonaLive Photo Sampler (AIIA To-Disk for Long Video)"
}