import torch
from torch import nn
from einops import rearrange
from polygraphy.backend.trt import Profile

class unet_work(nn.Module): # Ugly Power Strip
    def __init__(self, pose_guider, motion_encoder, unet, vae, scheduler, timestep):
        super().__init__()
        self.pose_guider = pose_guider
        self.motion_encoder = motion_encoder
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.timesteps = timestep

    def decode_slice(self, vae, x):
        x = x / 0.18215
        x = vae.decode(x).sample
        x = rearrange(x, "b c h w -> b h w c")
        x = (x / 2 + 0.5).clamp(0, 1)
        return x
    
    def forward(self, sample, encoder_hidden_states, motion_hidden_states, motion, pose_cond_fea, pose, new_noise, 
        d00, d01, d10, d11, d20, d21, m, u10, u11, u12, u20, u21, u22, u30, u31, u32
        ):
        new_pose_cond_fea = self.pose_guider(pose)
        pose_cond_fea = torch.cat([pose_cond_fea, new_pose_cond_fea], dim=2)
        new_motion_hidden_states = self.motion_encoder(motion)
        motion_hidden_states = torch.cat([motion_hidden_states, new_motion_hidden_states], dim=1)
        encoder_hidden_states = [encoder_hidden_states, motion_hidden_states]
        score = self.unet(sample, self.timesteps, encoder_hidden_states, pose_cond_fea, d00, d01, d10, d11, d20, d21, m, u10, u11, u12, u20, u21, u22, u30, u31, u32)
        score = rearrange(score, 'b c f h w -> (b f) c h w')
        sample = rearrange(sample, 'b c f h w -> (b f) c h w')
        latents_model_input, pred_original_sample = self.scheduler.step(
            score, self.timesteps, sample, return_dict=False
        )
        latents_model_input = latents_model_input.to(sample.dtype)
        pred_original_sample = pred_original_sample.to(sample.dtype)
        latents_model_input = rearrange(latents_model_input, '(b f) c h w -> b c f h w', f=16)
        pred_video = self.decode_slice(self.vae, pred_original_sample[:4])
        latents = torch.cat([latents_model_input[:, :, 4:, :, :], new_noise], dim=2)
        pose_cond_fea_out = pose_cond_fea[:, :, 4:, :, :]
        motion_hidden_states_out = motion_hidden_states[:, 4:, :, :]
        motion_out = motion_hidden_states[:, :1, :, :]
        return pred_video, latents, pose_cond_fea_out, motion_hidden_states_out, motion_out, pred_original_sample[:1]
    
    def get_sample_input(self, batchsize, height, width, dtype, device):
        tw, ts, tb = 4, 4, 16 # temporal window size| temporal adaptive steps | temporal batch size
        ml, mc, mh, mw= 32, 16, 224, 224 # motion latent size | motion channels
        b, h, w = batchsize, height, width
        lh, lw = height // 8, width // 8 # latent height | width
        cd0, cd1, cd2, cm, cu1, cu2, cu3 = 320, 640, 1280, 1280, 1280, 640, 320 # unet channels
        emb = 768 # CLIP Embedding Dims | TAESDV Channels
        lc, ic = 4, 3 # latent | image channels
        profile = {
            "sample" : [b, lc, tb, lh, lw],
            "encoder_hidden_states" : [b, 1, emb],
            "motion_hidden_states" : [b, tw * (ts - 1), ml, mc],
            "motion": [b, ic, tw, mh, mw],
            "pose_cond_fea" : [b, cd0, tw * (ts - 1), lh, lw],
            "pose" : [b, ic, tw, h, w],
            "new_noise" : [b, lc, tw, lh, lw],
            "d00" : [b, lh * lw, cd0],
            "d01" : [b, lh * lw, cd0],
            "d10" : [b, lh * lw // 4, cd1],
            "d11" : [b, lh * lw // 4, cd1],
            "d20" : [b, lh * lw // 16, cd2],
            "d21" : [b, lh * lw // 16, cd2],
            "m" : [b, lh * lw // 64, cm],
            "u10" : [b, lh * lw // 16, cu1],
            "u11" : [b, lh * lw // 16, cu1],
            "u12" : [b, lh * lw // 16, cu1],
            "u20" : [b, lh * lw // 4, cu2],
            "u21" : [b, lh * lw // 4, cu2],
            "u22" : [b, lh * lw // 4, cu2],
            "u30" : [b, lh * lw, cu3],
            "u31" : [b, lh * lw, cu3],
            "u32" : [b, lh * lw, cu3],
        }
        return {k: torch.randn(profile[k], dtype=dtype, device=device) for k in profile}
    
    def get_input_names(self):
        return ["sample", "encoder_hidden_states", "motion_hidden_states", 
                "motion", "pose_cond_fea", "pose", "new_noise", 
                "d00", "d01", "d10", "d11", "d20", "d21", "m", "u10", "u11", "u12", 
                "u20", "u21", "u22", "u30", "u31", "u32"]
   
    def get_output_names(self):
        return ["pred_video", "latents", "pose_cond_fea_out", 
                "motion_hidden_states_out", "motion_out", "latent_first"]

    def get_dynamic_axes(self):
        dynamic_axes = {
            "sample": {3:"h_64", 4:"w_64"},
            "pose_cond_fea": {3:"h_64", 4:"w_64"},
            "pose": {3:"h_512", 4:"h_512"},
            "new_noise": {3: "h_64", 4: "w_64"},
            "d00" : {1: "len_4096"},
            "d01" : {1: "len_4096"},
            "u30" : {1: "len_4096"},
            "u31" : {1: "len_4096"},
            "u32" : {1: "len_4096"},
            "d10" : {1: "len_1024"},
            "d11" : {1: "len_1024"},
            "u20" : {1: "len_1024"},
            "u21" : {1: "len_1024"},
            "u22" : {1: "len_1024"},
            "d20" : {1: "len_256"},
            "d21" : {1: "len_256"},
            "u10" : {1: "len_256"},
            "u11" : {1: "len_256"},
            "u12" : {1: "len_256"},
            "m"   : {1: "len_64"},
        }
        return dynamic_axes
    
    def get_dynamic_map(self, batchsize, height, width):
        tw, ts, tb = 4, 4, 16 # temporal window size| temporal adaptive steps | temporal batch size
        ml, mc, mh, mw= 32, 16, 224, 224 # motion latent size | motion channels
        b, h, w = batchsize, height, width
        lh, lw = height // 8, width // 8 # latent height | width
        cd0, cd1, cd2, cm, cu1, cu2, cu3 = 320, 640, 1280, 1280, 1280, 640, 320 # unet channels
        emb = 768 # CLIP Embedding Dims | TAESDV Channels
        lc, ic = 4, 3 # latent | image channels
        
        fixed_inputs_map = {
            "sample":                (b, lc, tb, lh, lw),
            "encoder_hidden_states": (b, 1, emb),
            "motion_hidden_states":  (b, tw * (ts - 1), ml, mc),
            "motion":                (b, ic, tw, mh, mw),
            "pose_cond_fea":         (b, cd0, tw * (ts - 1), lh, lw),
            "pose":                  (b, ic, tw, h, w),
            "new_noise":             (b, lc, tw, lh, lw),
        }
        
        dynamic_inputs_map = {
            "d00": (b, lh * lw, cd0),
            "d01": (b, lh * lw, cd0),
            "d10": (b, lh * lw // 4, cd1),
            "d11": (b, lh * lw // 4, cd1),
            "d20": (b, lh * lw // 16, cd2),
            "d21": (b, lh * lw // 16, cd2),
            "m":   (b, lh * lw // 64, cm),
            "u10": (b, lh * lw // 16, cu1),
            "u11": (b, lh * lw // 16, cu1),
            "u12": (b, lh * lw // 16, cu1),
            "u20": (b, lh * lw // 4, cu2),
            "u21": (b, lh * lw // 4, cu2),
            "u22": (b, lh * lw // 4, cu2),
            "u30": (b, lh * lw, cu3),
            "u31": (b, lh * lw, cu3),
            "u32": (b, lh * lw, cu3),
        }
        
        profile = Profile()
        
        for name, shape in fixed_inputs_map.items():
            shape_tuple = tuple(shape)
            profile.add(name, min=shape_tuple, opt=shape_tuple, max=shape_tuple)
            
        for name, base_shape in dynamic_inputs_map.items():
            
            dim0, dim1_base, dim2 = base_shape
            
            val_1x = dim1_base * 1
            val_2x = dim1_base * 2
            val_4x = dim1_base * 4
            
            min_shape = (dim0, val_1x, dim2)
            opt_shape = (dim0, val_2x, dim2) 
            max_shape = (dim0, val_4x, dim2)
            
            profile.add(name, min=min_shape, opt=opt_shape, max=max_shape)
            
            print(f"Dynamic: {name:<5} | Base(1x): {dim1_base:<5} | Range: {val_1x} ~ {val_4x} | Opt: {val_2x}")
            
        return profile