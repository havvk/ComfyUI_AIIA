#!/app/miniconda3/envs/comfyui/bin/python
import os
import sys
import torch
import numpy as np
import math
import gc
import logging
from PIL import Image
from omegaconf import OmegaConf

# Mock ComfyUI environment
class MockFolderPaths:
    models_dir = "/app/ComfyUI/models"

folder_paths = MockFolderPaths()

# Configure Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # ComfyUI_AIIA root
echomimic_v3_root = os.path.join(root_dir, "libs", "EchoMimicV3")

sys.path.insert(0, root_dir)
sys.path.insert(0, echomimic_v3_root)

# Imports from EchoMimicV3
from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from diffusers import FlowMatchEulerDiscreteScheduler
from echomimic_v3_src.wan_vae import AutoencoderKLWan
from echomimic_v3_src.wan_image_encoder import CLIPModel
from echomimic_v3_src.wan_text_encoder import WanT5EncoderModel
from echomimic_v3_src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from echomimic_v3_src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
from echomimic_v3_src.utils import get_image_to_video_latent3, filter_kwargs
from echomimic_v3_src.face_detect import get_mask_coord
import torchvision.transforms.functional as TF
from echomimic_v3_src.fm_solvers import FlowDPMSolverMultistepScheduler

# Config
ECHOMIMIC_MODELS_DIR = "EchoMimicV3"
MODEL_SUBFOLDER = "EchoMimicV3" # or Wan2.1...
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
SAMPLE_IMAGE_PATH = "/app/ComfyUI/input/xuerOneCyanTenColor_fluxV10--20241112-194257-00001.png"
SAMPLE_AUDIO_PATH = "/app/ComfyUI/custom_nodes/ComfyUI_AIIA/assets/seed_male.wav"

def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    mask = mask.reshape(-1)
    return mask.float()

def main():
    print(f"=== Starting EchoMimicV3 Manual Test ===")
    print(f"Device: {DEVICE}, DType: {DTYPE}")
    
    # 1. Load Config
    config_path = os.path.join(echomimic_v3_root, "config", "config.yaml")
    cfg = OmegaConf.load(config_path)
    
    # 2. Paths
    model_root = os.path.join(folder_paths.models_dir, ECHOMIMIC_MODELS_DIR, "Wan2.1-Fun-V1.1-1.3B-InP")
    if not os.path.exists(model_root):
        model_root = os.path.join(folder_paths.models_dir, ECHOMIMIC_MODELS_DIR, "EchoMimicV3")
        
    print(f"Model Root: {model_root}")

    # 3. Load Models
    print("Loading Transformer...")
    transformer = WanTransformerAudioMask3DModel.from_pretrained(
        os.path.join(model_root, "transformer"),
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        torch_dtype=torch.float32, # Load as float32 then move/cast
        low_cpu_mem_usage=True
    ).to("cpu").to(DTYPE) # Keep on CPU first
    
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_root, "Wan2.1_VAE.pth"),
        additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
    ).to(dtype=torch.float32, device="cpu") # VAE on CPU initially

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root, "google/umt5-xxl"))

    print("Loading Text Encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_root, "text_encoder"),
        additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True
    ).to(dtype=DTYPE, device="cpu").eval()

    print("Loading Image Encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_root, "image_encoder")
    ).to(dtype=DTYPE, device="cpu").eval()
    
    print("Loading Scheduler...")
    scheduler_kwargs = OmegaConf.to_container(cfg['scheduler_kwargs'])
    scheduler = FlowDPMSolverMultistepScheduler(**filter_kwargs(FlowDPMSolverMultistepScheduler, scheduler_kwargs))

    print("Loading Audio Encoder...")
    wav2vec_path = os.path.join(folder_paths.models_dir, ECHOMIMIC_MODELS_DIR, "wav2vec2-base-960h")
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
    wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path).to(dtype=DTYPE, device="cpu").eval()

    # 4. Pipeline
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    ).to("cpu")

    # 5. Prepare Inputs
    print("Preparing Inputs...")
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"Image not found at {SAMPLE_IMAGE_PATH}, creating dummy...")
        ref_img_pil = Image.new("RGB", (768, 768), (100, 100, 200))
    else:
        ref_img_pil = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
        ref_img_pil = ref_img_pil.resize((768, 768)) # Force resize

    # Audio
    if not os.path.exists(SAMPLE_AUDIO_PATH):
        raise FileNotFoundError(f"Audio not found: {SAMPLE_AUDIO_PATH}")
        
    audio_wav, sr = torchaudio.load(SAMPLE_AUDIO_PATH)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_wav = resampler(audio_wav)
    
    # Process Audio
    # Simplified audio feature extraction
    audio_inputs = wav2vec_processor(audio_wav[0].numpy(), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        audio_embeds = wav2vec_model(audio_inputs.to(dtype=DTYPE)).last_hidden_state
    audio_embeds = audio_embeds.to(DEVICE, dtype=DTYPE)

    # Face Mask (Simplified - full face)
    # EchoMimicV3 usually detects face, here we just make a centered mask
    print("Generating Mask...")
    # mask_coord = get_mask_coord(ref_img_pil, ...) # Skip face detection for simplicity
    # Use a dummy center crop mask
    h, w = 768, 768
    y1, y2, x1, x2 = h//4, h*3//4, w//4, w*3//4
    ip_mask = get_ip_mask((y1, y2, x1, x2, h, w)).to(DEVICE, dtype=DTYPE).unsqueeze(0)

    # 6. Run Generation (Short chunk)
    print("Starting Generation...")
    
    # Manual Memory Mgmt (Mimic Node)
    
    # Encode Prompt
    print("Encoding Prompt...")
    pipeline.text_encoder.to(DEVICE)
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        "a talking head video", "bad quality", True, 1, 512, device=DEVICE
    )
    pipeline.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Move Models
    pipeline.transformer.to(DEVICE)
    pipeline.vae.to(DEVICE)
    
    # Prepare Latents
    input_video, input_video_mask, clip_image = get_image_to_video_latent3(
        ref_img_pil, None, video_length=25, sample_size=[768, 768]
    )
    
    # CLIP Context
    print("Computing CLIP Context...")
    pipeline.clip_image_encoder.to(DEVICE)
    clip_image_t = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(DEVICE, dtype=DTYPE)
    clip_context = pipeline.clip_image_encoder([clip_image_t[:, None, :, :]])
    pipeline.clip_image_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Run
    print("Running Pipeline Loop...")
    partial_audio_embeds = audio_embeds[:, :50] # 25 frames * 2
    
    with torch.no_grad():
        sample = pipeline(
            prompt=None,
            num_frames=25,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            audio_embeds=partial_audio_embeds,
            audio_scale=1.0,
            ip_mask=ip_mask,
            use_un_ip_mask=False,
            height=768,
            width=768,
            generator=torch.Generator(device="cpu").manual_seed(42),
            clip_context=clip_context,
            neg_scale=1.5,
            neg_steps=2,
            use_dynamic_cfg=True,
            use_dynamic_acfg=True,
            guidance_scale=2.5,
            audio_guidance_scale=1.0,
            num_inference_steps=20, # Short run
            video=input_video,
            mask_video=input_video_mask,
            clip_image=clip_image,
        ).videos

    print("Generation/Inference successful!")
    print(f"Output shape: {sample.shape}")

if __name__ == "__main__":
    main()
