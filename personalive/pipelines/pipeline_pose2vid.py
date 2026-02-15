import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from diffusers import DiffusionPipeline
from diffusers import (
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor
from ComfyUI_AIIA.personalive.pipelines.utils import get_tensor_interpolation_method
from ComfyUI_AIIA.personalive.models.mutual_self_attention import ReferenceAttentionControl
from ComfyUI_AIIA.personalive.utils.util import draw_keypoints

@dataclass
class Pose2VideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class Pose2VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        # vae_tiny,
        image_encoder,
        reference_unet,
        denoising_unet,
        motion_encoder,
        pose_encoder,
        pose_guider,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            # vae_tiny=vae_tiny,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            motion_encoder=motion_encoder,
            pose_encoder=pose_encoder,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=True,
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device
    
    def decode_latents_tiny(self, latents, decode_chunk_size=64):
        video_length = latents.shape[2]
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(0, latents.shape[0], decode_chunk_size)):
            video.append(self.vae_tiny.decode(latents[frame_idx : frame_idx + decode_chunk_size]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def decode_latents(self, latents, decode_chunk_size=16):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in range(0, latents.shape[0], decode_chunk_size):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + decode_chunk_size]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def decode_latents_svd(
        self, latents: torch.FloatTensor, decode_chunk_size: int = 5
    ):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            frame = self.vae.decode(
                latents[i : i + decode_chunk_size], num_frames_in
            ).sample
            frames.append(frame)
        video = torch.cat(frames, dim=0)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_videos_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_videos_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def downgrade_input(self, init_latents, generator, device, dtype):
        mask = (
            torch.rand(
                *init_latents.shape,
                generator=generator,
                device=device,
            )
            >= 0.5
        ).to(dtype=dtype)

        blur = transforms.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
        # first_frame_latents = init_latents[:, :, 0:1, :, :].clone()
        video_length = init_latents.shape[2]
        init_latents = rearrange(init_latents, "b c f h w -> (b f) c h w")
        init_latents_blur = blur(init_latents)
        init_latents = rearrange(init_latents, "(b f) c h w -> b c f h w", f=video_length)
        init_latents_blur = rearrange(init_latents_blur, "(b f) c h w -> b c f h w", f=video_length)
        # init_latents = init_latents * mask + init_latents_blur * (1 - mask)
        # init_latents[:, :, 0:1, :, :] = first_frame_latents

        return init_latents_blur

    def interpolate_latents(
        self, latents: torch.Tensor, interpolation_factor: int, device
    ):
        if interpolation_factor < 2:
            return latents

        new_latents = torch.zeros(
            (
                latents.shape[0],
                latents.shape[1],
                ((latents.shape[2] - 1) * interpolation_factor) + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            device=latents.device,
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:
                v = get_tensor_interpolation_method()(
                    v0.to(device=device), v1.to(device=device), f
                )
                new_latents[:, :, new_index, :, :] = v.to(latents.device)
                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    def interpolate_tensors(self, a: torch.Tensor, b: torch.Tensor, num: int = 10) -> torch.Tensor:
        """
        Linear interpolation between tensors a and b.
        input shape: (B, 1, D1, D2, ...)
        output shape: (B, num, D1, D2, ...)
        """
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}")

        B, _, *rest = a.shape
        alphas = torch.linspace(0, 1, num, device=a.device, dtype=a.dtype)
        view_shape = (1, num) + (1,) * len(rest)
        alphas = alphas.view(view_shape)  # (1, num, 1, 1, ...)

        # (B, num, D1, D2, ...)
        result = (1 - alphas) * a + alphas * b
        return result
    
    def calculate_dis(self, A, B, threshold=10.):
        """
        A: (b, f1, c1, c2)  bank
        B: (b, f2, c1, c2)  new data
        """

        A_flat = A.view(A.size(1), -1).clone()
        B_flat = B.view(B.size(1), -1).clone()

        dist = torch.cdist(B_flat.to(torch.float32), A_flat.to(torch.float32), p=2)

        min_dist, min_idx = dist.min(dim=1)  # (f2,)

        idx_to_add = torch.nonzero(min_dist[:1] > threshold, as_tuple=False).squeeze(1).tolist()

        if len(idx_to_add) > 0:
            B_to_add = B[:, idx_to_add]  # (1, k, c1, c2)
            A_new = torch.cat([A, B_to_add], dim=1)  # (1, f1+k, c1, c2)
            add_flag = True
        else:
            A_new = A
            add_flag = False

        return add_flag, A_new

    @torch.no_grad()
    def __call__(
        self,
        tgt_images,
        ref_image,
        face_images,
        ref_face_image,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        temporal_window_size = 4,
        temporal_adaptive_step = 4,
        temporal_kv_cache=True,
        init_latents=None,
        **kwargs,
    ):
        assert num_inference_steps % temporal_adaptive_step == 0, "temporal_adaptive_step should be divisor of num_inference_steps"
        assert video_length % temporal_window_size == 0, "temporal_window_size should be divisor of video_length"
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        timesteps = torch.tensor([999, 666, 333, 0], device=device).long()
        self.scheduler.set_step_length(333)
        jump = num_inference_steps // temporal_adaptive_step
        windows = video_length // temporal_window_size

        batch_size = 1

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_encoder(
            clip_image.to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        image_prompt_embeds = clip_image_embeds.unsqueeze(1)

        uncond_image_prompt_embeds = torch.zeros_like(image_prompt_embeds)

        if do_classifier_free_guidance:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
            )

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
            cache_kv=temporal_kv_cache,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(
            ref_image, height=height, width=width
        )  # (bs, c, width, height)
        ref_image_tensor = ref_image_tensor.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # initialize latents
        padding_num = (temporal_adaptive_step - 1) * temporal_window_size
        noisy_latents_first = []
        init_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, padding_num, 1, 1)
        init_timesteps = reversed(timesteps[0::jump]).repeat_interleave(temporal_window_size, dim=0)
        noise = torch.randn_like(init_latents)
        noisy_latents_first.append(self.scheduler.add_noise(init_latents, noise, init_timesteps[:padding_num]))
        
        repeated_latents = ref_image_latents.unsqueeze(2).repeat(1, 1, video_length+padding_num, 1, 1)
        noise = torch.randn_like(repeated_latents)
        noisy_latents_first.append(self.scheduler.add_noise(repeated_latents, noise, timesteps[:1]))
        latents = torch.cat(noisy_latents_first, dim=2)

        # 3D implicit keypoints
        ref_cond_tensor = self.cond_image_processor.preprocess(
            ref_image, height=256, width=256
        ).to(device=device, dtype=self.pose_encoder.dtype)  # (1, c, h, w)
        ref_cond_tensor = ref_cond_tensor / 2 + 0.5 # to [0, 1]

        tgt_cond_tensor = self.cond_image_processor.preprocess(
            tgt_images, height=256, width=256
        ).to(device=device, dtype=self.pose_encoder.dtype)  # (1, c, h, w)
        tgt_cond_tensor = tgt_cond_tensor / 2 + 0.5 # to [0, 1]
        
        mot_bbox_param = self.pose_encoder.interpolate_kps(ref_cond_tensor, tgt_cond_tensor, num_interp=padding_num+1)  # (t, c)
        keypoints = draw_keypoints(mot_bbox_param, height=height, width=width, device=device).unsqueeze(2)
        keypoints = rearrange(keypoints, 'f c b h w -> b c f h w')
        keypoints = keypoints.to(device=device, dtype=self.pose_guider.dtype)
        
        pose_feas = []
        for i in range(0, keypoints.shape[2], 256):
            pose_fea = self.pose_guider(keypoints[:,:, i:i+256, :, :])  # (b, c, f)
            pose_feas.append(pose_fea)
        pose_feas = torch.cat(pose_feas, dim=2)

        # motion embeddings
        face_cond_tensor = self.cond_image_processor.preprocess(
            face_images, height=224, width=224
        ).transpose(0, 1)
        face_cond_tensor = face_cond_tensor.unsqueeze(0) # (1, c, t, h, w)
        face_cond_tensor = face_cond_tensor.to(
            device=device, dtype=self.motion_encoder.dtype
        )
        
        motion_hidden_states = []
        for i in range(0, face_cond_tensor.shape[2], 256):
            motion_hidden_state = self.motion_encoder(face_cond_tensor[:,:, i:i+256, :, :])  # [b, c]
            motion_hidden_states.append(motion_hidden_state)
        motion_hidden_states = torch.cat(motion_hidden_states, dim=1)  # [b, f, c]

        ref_face_cond_tensor = self.cond_image_processor.preprocess(
            ref_face_image, height=224, width=224
        ).to(device=device, dtype=self.motion_encoder.dtype)
        neg_motion_hidden_states = self.motion_encoder(ref_face_cond_tensor.unsqueeze(2))    # [b, c]

        if do_classifier_free_guidance:
            ref_motion_emb = mot_bbox_param[:1]
            ref_keypoints = draw_keypoints(ref_motion_emb, height=height, width=width, device=device).unsqueeze(2)
            ref_keypoints = ref_keypoints.to(device=device, dtype=self.pose_guider.dtype)
            ref_pose_fea = self.pose_guider(ref_keypoints)
        
        init_motion_hidden_states = self.interpolate_tensors(neg_motion_hidden_states, motion_hidden_states[:,:1], num=padding_num+1)[:,:-1]
        motion_hidden_states = torch.cat([init_motion_hidden_states, motion_hidden_states, motion_hidden_states[:,-1:].repeat(1, padding_num, 1, 1)], dim=1)
        pose_feas = torch.cat([pose_feas, pose_feas[:,:,-1:].repeat(1,1,padding_num,1,1)], dim=2)

        motion_bank = neg_motion_hidden_states

        # denoising loop
        with self.progress_bar(total=windows + temporal_adaptive_step - 1) as progress_bar:
            self.reference_unet(
                ref_image_latents.repeat((2 if do_classifier_free_guidance else 1), 1, 1, 1),
                torch.zeros((batch_size,),dtype=torch.float32,device=ref_image_latents.device),
                encoder_hidden_states=image_prompt_embeds,
                return_dict=False,
            )
            reference_control_reader.update(reference_control_writer)

            for i in range(windows + temporal_adaptive_step - 1):
                l = i * temporal_window_size
                r = (i + temporal_adaptive_step) * temporal_window_size

                motion_hidden_state = motion_hidden_states[:, l:r]
                pose_fea = pose_feas[:, :, l:r]

                add_flag = False
                if l > temporal_adaptive_step * temporal_window_size * 2 and motion_bank.shape[1] < 4:
                    add_flag, motion_bank = self.calculate_dis(motion_bank, motion_hidden_state, threshold=17.)
                
                if do_classifier_free_guidance:
                    motion_hidden_state = torch.cat(
                        [
                            neg_motion_hidden_states.expand_as(motion_hidden_state),
                            motion_hidden_state,
                        ],
                        dim=0,
                    )

                    pose_fea = torch.cat(
                        [
                            ref_pose_fea.expand_as(pose_fea),
                            pose_fea,
                        ],
                        dim=0,
                    )

                for j in range(jump):
                    latent = latents[:,:,l:r,:,:]
                    latent_model_input = (torch.cat([latent] * 2) if do_classifier_free_guidance else latent)
                    ut = reversed(timesteps[j::jump]).repeat_interleave(temporal_window_size, dim=0)
                    ut = torch.stack([ut] * batch_size * (2 if do_classifier_free_guidance else 1)).to(device)
                    ut = rearrange(ut, 'b f -> (b f)')

                    noise_pred = self.denoising_unet(
                        latent_model_input,
                        ut,
                        encoder_hidden_states=[
                            image_prompt_embeds,
                            motion_hidden_state,
                        ],
                        pose_cond_fea=pose_fea,
                        return_dict=False,
                    )[0]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    clip_length = noise_pred.shape[2]
                    mid_noise_pred = rearrange(noise_pred, 'b c f h w -> (b f) c h w')
                    mid_latents = rearrange(latents[:,:,l:r,:,:], 'b c f h w -> (b f) c h w')
                    ts = ut.chunk(2)[0] if do_classifier_free_guidance else ut
                    mid_latents, pred_original_sample = self.scheduler.step(
                        mid_noise_pred, ts, mid_latents, **extra_step_kwargs, return_dict=False
                    )
                    mid_latents = rearrange(mid_latents, '(b f) c h w -> b c f h w', f=clip_length)
                    pred_original_sample = rearrange(pred_original_sample, '(b f) c h w -> b c f h w', f=clip_length)
                    mid_latents = torch.cat([
                        pred_original_sample[:,:,:temporal_window_size],
                        mid_latents[:,:,temporal_window_size:]], dim=2)
                    latents[:,:,l:r,:,:] = mid_latents
                
                # history keyframe mechanism
                if add_flag:
                    reference_control_writer.clear()
                    self.reference_unet(
                        pred_original_sample[:,:,0].repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ).to(self.reference_unet.dtype),
                        torch.zeros((batch_size,),dtype=torch.float32,device=ref_image_latents.device),
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )
                    reference_control_reader.update_hkf(reference_control_writer)
                
                progress_bar.update()
            reference_control_reader.clear()
            reference_control_writer.clear()

        latents = latents[:, :, padding_num:video_length+padding_num, :, :]
        # Post-processing
        if isinstance(self.vae, AutoencoderKL):
            # images = self.decode_latents_tiny(latents)  # (b, c, f, h, w)
            images = self.decode_latents(latents)  # (b, c, f, h, w)
        elif isinstance(self.vae, AutoencoderKLTemporalDecoder):
            images = self.decode_latents_svd(latents)

        # Convert to tensor
        if output_type == "tensor":
            images = torch.from_numpy(images)

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)
