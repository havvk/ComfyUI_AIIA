
import os
import sys
import torch
from omegaconf import OmegaConf

# Add libs to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # ComfyUI_AIIA root
echomimic_v3_root = os.path.join(root_dir, "libs", "EchoMimicV3")
sys.path.insert(0, echomimic_v3_root)
sys.path.insert(0, root_dir)

# Import Config from infer.py (or recreate it if imports fail)
# We might need to mock some things if infer.py has hardcoded paths
# Let's import the pipeline and run parts of main logic from infer.py instead of running it directly
# because infer.py has hardcoded paths in Config class.

from transformers import AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor
from echomimic_v3_src.wan_vae import AutoencoderKLWan
from echomimic_v3_src.wan_image_encoder import CLIPModel
from echomimic_v3_src.wan_text_encoder import WanT5EncoderModel
from echomimic_v3_src.wan_transformer3d_audio import WanTransformerAudioMask3DModel
from echomimic_v3_src.pipeline_wan_fun_inpaint_audio import WanFunInpaintAudioPipeline
from echomimic_v3_src.utils import get_image_to_video_latent3, filter_kwargs
from echomimic_v3_src.fm_solvers import FlowDPMSolverMultistepScheduler
from echomimic_v3_src.face_detect import get_mask_coord
import torchvision.transforms.functional as TF
from PIL import Image
import torchaudio

# Setup Paths
MODELS_DIR = "/app/ComfyUI/models/EchoMimicV3"
WAN_ROOT = os.path.join(MODELS_DIR, "Wan2.1-Fun-V1.1-1.3B-InP")
ECHOMIMIC_ROOT = os.path.join(MODELS_DIR, "EchoMimicV3")
SAMPLE_AUDIO = "/app/ComfyUI/custom_nodes/ComfyUI_AIIA/assets/seed_male.wav"
SAMPLE_IMAGE = "/app/ComfyUI/input/xuerOneCyanTenColor_fluxV10--20241112-194257-00001.png"

def run_official_style_inference():
    print("=== EchoMimicV3 Official-Style Test ===")
    device = "cuda"
    dtype = torch.bfloat16
    
    # 1. Load Config
    config_path = os.path.join(echomimic_v3_root, "config", "config.yaml")
    print(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)

    # 2. Load Models (Using logic similar to infer.py but with correct paths)
    print("Loading Models...")
    
    # Transformer
    transformer_path = os.path.join(ECHOMIMIC_ROOT, "transformer")
    print(f"Transformer path: {transformer_path}")
    if not os.path.exists(transformer_path):
        print("Transformer missing, trying Wan root")
        transformer_path = WAN_ROOT
        
    transformer = WanTransformerAudioMask3DModel.from_pretrained(
        transformer_path,
        transformer_additional_kwargs=OmegaConf.to_container(cfg['transformer_additional_kwargs']),
        torch_dtype=torch.float32, # Force float32 loading as per previous fixes
        low_cpu_mem_usage=True
    ).to(device).to(dtype) # Move to GPU immediately like infer.py does (usually)

    # VAE
    vae_path = os.path.join(WAN_ROOT, "Wan2.1_VAE.pth")
    print(f"VAE path: {vae_path}")
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        additional_kwargs=OmegaConf.to_container(cfg['vae_kwargs']),
    ).to(device, dtype=torch.float32)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(WAN_ROOT, "google/umt5-xxl"))

    # Text Encoder
    text_enc_path = os.path.join(WAN_ROOT, "models_t5_umt5-xxl-enc-bf16.pth")
    print(f"Text Encoder path: {text_enc_path}")
    text_encoder = WanT5EncoderModel.from_pretrained(
        text_enc_path,
        additional_kwargs=OmegaConf.to_container(cfg['text_encoder_kwargs']),
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device, dtype=dtype)

    # Image Encoder
    img_enc_path = os.path.join(WAN_ROOT, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    print(f"Image Encoder path: {img_enc_path}")
    clip_image_encoder = CLIPModel.from_pretrained(img_enc_path).to(device, dtype=dtype)

    # Audio Encoder
    wav2vec_path = os.path.join(MODELS_DIR, "wav2vec2-base-960h")
    print(f"Wav2Vec path: {wav2vec_path}")
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_path)
    wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path).to(device, dtype=dtype)

    # Scheduler
    scheduler = FlowDPMSolverMultistepScheduler(**filter_kwargs(FlowDPMSolverMultistepScheduler, OmegaConf.to_container(cfg['scheduler_kwargs'])))

    # Pipeline
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    ).to(device) # Move entire pipeline to device

    # 3. Inputs
    ref_img = Image.open(SAMPLE_IMAGE).convert("RGB").resize((768, 768))
    audio_wav, sr = torchaudio.load(SAMPLE_AUDIO)
    if sr != 16000:
        audio_wav = torchaudio.transforms.Resample(sr, 16000)(audio_wav)
    
    # Simple fake mask
    h, w = 768, 768
    y1, y2, x1, x2 = h//4, h*3//4, w//4, w*3//4
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y >= y1) & (Y < y2) & (X >= x1) & (X < x2)
    ip_mask = mask.float().to(device, dtype=dtype).view(-1).unsqueeze(0)

    audio_inputs = wav2vec_processor(audio_wav[0].numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device, dtype=dtype)
    with torch.no_grad():
        audio_embeds = wav2vec_model(audio_inputs).last_hidden_state

    # 4. Run
    print("Encoding Prompt...")
    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt="a talking head video",
        negative_prompt="bad quality",
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device
    )

    print("Pre-processing Video Latents...")
    latents, mask_latents, clip_image_pixel_values = get_image_to_video_latent3(
        ref_img, None, video_length=25, sample_size=[768, 768]
    )
    # Ensure all inputs are on device
    latents = latents.to(device, dtype=dtype)
    mask_latents = mask_latents.to(device, dtype=dtype) if mask_latents is not None else None
    
    # Explicitly handle clip image context
    clip_image_pixel_values = TF.to_tensor(clip_image_pixel_values).sub_(0.5).div_(0.5).to(device, dtype=dtype)
    clip_context = clip_image_encoder([clip_image_pixel_values.unsqueeze(0)])
    
    print("Starting generation loop...")
    video = pipeline(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        audio_embeds=audio_embeds[:, :50], # Shorten for test
        ip_mask=ip_mask,
        video=latents,
        mask_video=mask_latents,
        clip_image=clip_image_pixel_values,
        clip_context=clip_context,
        num_inference_steps=20,
        guidance_scale=4.0,
        audio_guidance_scale=2.0,
        height=768, width=768,
        num_frames=25,
        generator=torch.Generator(device="cpu").manual_seed(42) # Ensure generator is valid
    ).videos

    print(f"Success! Output shape: {video.shape}")

if __name__ == "__main__":
    run_official_style_inference()
