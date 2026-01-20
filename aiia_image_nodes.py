import torch
import numpy as np
from PIL import Image

class AIIA_ImageSmartCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 8}),
                "crop_basis": (["custom_size", "fixed_width", "fixed_height", "fixed_long_side", "fixed_short_side"],),
                "aspect_ratio": (["original", "custom", "1:1", "4:3", "3:4", "16:9", "9:16", "21:9", "9:21"], {"default": "original"}),
                "custom_aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "position": (["center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"],),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "AIIA/Image"

    def crop(self, image, width, height, crop_basis, aspect_ratio, custom_aspect_ratio, position, offset_x, offset_y):
        # Image is typically [B, H, W, C]
        batch_results = []
        
        for i in range(image.shape[0]):
            img_tensor = image[i]
            # Convert to PIL for easier handling
            img_np = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            src_w, src_h = img_pil.size
            tgt_w, tgt_h = width, height

            # --- 0. Determine Target Aspect Ratio ---
            target_ratio = None # w / h
            
            if aspect_ratio == "original":
                target_ratio = src_w / src_h
            elif aspect_ratio == "custom":
                target_ratio = custom_aspect_ratio
            elif aspect_ratio == "1:1":
                target_ratio = 1.0
            elif aspect_ratio == "4:3":
                target_ratio = 4.0 / 3.0
            elif aspect_ratio == "3:4":
                target_ratio = 3.0 / 4.0
            elif aspect_ratio == "16:9":
                target_ratio = 16.0 / 9.0
            elif aspect_ratio == "9:16":
                target_ratio = 9.0 / 16.0
            elif aspect_ratio == "21:9":
                target_ratio = 21.0 / 9.0
            elif aspect_ratio == "9:21":
                target_ratio = 9.0 / 21.0
                
            # --- 1. Determine Target Dimensions ---
            if crop_basis == "custom_size":
                 # In custom_size mode, if aspect_ratio is NOT original, we override height? 
                 # Usually custom_size means strict Width x Height.
                 # Let's say: If Aspect Ratio is standard (original), use WxH.
                 # If user Explicitly selects a ratio (e.g. 1:1), we respect Width and recalc Height?
                 if aspect_ratio != "original":
                     # Treat 'width' as the primary dimension
                     tgt_w = width
                     tgt_h = int(width / target_ratio)
                 else:
                     tgt_w = width
                     tgt_h = height
            
            elif crop_basis == "fixed_width":
                tgt_w = width
                tgt_h = int(width / target_ratio)
                
            elif crop_basis == "fixed_height":
                tgt_h = height
                tgt_w = int(height * target_ratio)
                
            elif crop_basis == "fixed_long_side":
                # Determine which side of the RESULT should be the long side based on ratio
                # Logic: We force the LONGEST side of the CROP to be 'width' pixels.
                
                # If target_ratio > 1 (Landscape), Width is long side.
                if target_ratio >= 1.0:
                    tgt_w = width
                    tgt_h = int(width / target_ratio)
                else:
                    # Portrait, Height is long side.
                    tgt_h = width # Using 'width' input as the constraint value
                    tgt_w = int(tgt_h * target_ratio)
                    
            elif crop_basis == "fixed_short_side":
                # Logic: We force the SHORTEST side of the CROP to be 'width' pixels.
                
                # If target_ratio > 1 (Landscape), Height is short side.
                if target_ratio >= 1.0:
                    tgt_h = width # Using 'width' input as the constraint value
                    tgt_w = int(tgt_h * target_ratio)
                else:
                    # Portrait, Width is short side.
                    tgt_w = width
                    tgt_h = int(tgt_w / target_ratio)

            # --- Logic End ---
            
            cw, ch = tgt_w, tgt_h
            
            # Limit crop size to source size
            cw = min(cw, src_w)
            ch = min(ch, src_h)

            # --- 3. Determine Position ---
            # Center coordinates of the crop box
            
            # Base anchors
            left = 0
            top = 0
            
            if position == "center":
                left = (src_w - cw) / 2
                top = (src_h - ch) / 2
            elif position == "top":
                left = (src_w - cw) / 2
                top = 0
            elif position == "bottom":
                left = (src_w - cw) / 2
                top = src_h - ch
            elif position == "left":
                left = 0
                top = (src_h - ch) / 2
            elif position == "right":
                left = src_w - cw
                top = (src_h - ch) / 2
            elif position == "top_left":
                left = 0
                top = 0
            elif position == "top_right":
                left = src_w - cw
                top = 0
            elif position == "bottom_left":
                left = 0
                top = src_h - ch
            elif position == "bottom_right":
                left = src_w - cw
                top = src_h - ch
                
            # Apply offsets (relative to source size)
            # offset_x = 0.1 means shift right by 10% of source width
            left += offset_x * src_w
            top += offset_y * src_h
            
            # Clamp to boundaries
            left = max(0, min(left, src_w - cw))
            top = max(0, min(top, src_h - ch))
            
            box = (int(left), int(top), int(left + cw), int(top + ch))
            
            crop_pil = img_pil.crop(box)
            
            # Convert back to Tensor
            crop_np = np.array(crop_pil).astype(np.float32) / 255.0
            crop_tensor = torch.from_numpy(crop_np)
            batch_results.append(crop_tensor)

        return (torch.stack(batch_results),)

NODE_CLASS_MAPPINGS = {
    "AIIA_ImageSmartCrop": AIIA_ImageSmartCrop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_ImageSmartCrop": "AIIA Image Smart Crop"
}
