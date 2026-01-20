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
                "position": (["center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"],),
                "offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = "AIIA/Image"

    def crop(self, image, width, height, crop_basis, position, offset_x, offset_y):
        # Image is typically [B, H, W, C]
        batch_results = []
        
        for i in range(image.shape[0]):
            img_tensor = image[i]
            # Convert to PIL for easier handling
            img_np = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            src_w, src_h = img_pil.size
            tgt_w, tgt_h = width, height

            # --- 1. Determine Target Dimensions ---
            if crop_basis == "fixed_width":
                tgt_w = width
                tgt_h = int(src_h * (width / src_w))
            elif crop_basis == "fixed_height":
                tgt_h = height
                tgt_w = int(src_w * (height / src_h))
            elif crop_basis == "fixed_long_side":
                if src_w > src_h:
                    tgt_w = width
                    tgt_h = int(src_h * (width / src_w))
                else:
                    tgt_h = width  # treat 'width' input as the long side length
                    tgt_w = int(src_w * (width / src_h))
            elif crop_basis == "fixed_short_side":
                if src_w < src_h:
                    tgt_w = width
                    tgt_h = int(src_h * (width / src_w))
                else:
                    tgt_h = width  # treat 'width' input as the short side length
                    tgt_w = int(src_w * (width / src_h))
            else: # custom_size
                pass # Use provided width and height directly

            # --- 2. Resize Logic (if preserving aspect ratio logic implies resizing first, or just cropping?)
            # The user asked for "crop out designated size". 
            # If the target size is strictly smaller than source, we crop.
            # If target size logic implies scaling (like "fixed width" Usually implies resize to that width), then we resize.
            # BUT, standard "Crop" nodes usually just cut. Smart Crop often implies finding the best area.
            # Given the "crop_basis" names, "fixed_width" usually means "Resize image so width is X".
            # However, prompt says "crop out designated proportion".
            # Let's assume standard Comfy behavior: 
            # "Smart Crop" implies resizing the *Shortest* side to fill the target frame, then cropping the rest?
            # OR, does it mean "Cut a 512x512 box out of a 1024x1024 image"?
            # Re-reading prompt: "from image designated position (top, left...) cut out designated proportion picture".
            # And "crop_scale... changes action area...".
            
            # Let's interpret "crop_basis" as "How to calculate the CROP BOX size".
            # If "fixed_width", the crop box has width=InputWidth, Height=InputHeight (or derived?).
            # Actually, standard "Resize" behavior is more useful here usually.
            # But let's stick to strict "Crop" logic if the user wants to isolate mouth etc.
            # However, if user wants to prevent "mouth too wide" (resolution mismatch), they likely want to RESIZE the image to 512x512 (centering or cropping).
            
            # Let's implement a hybrid "Resize then Crop" if needed, OR just "Crop".
            # For "custom_size" (512x512) on a 1920x1080 image:
            # We cut a 512x512 box.
            
            # Handling "fixed_width" etc for CROP size:
            # If I select "fixed_width" and width=512. Why do I need height? 
            # Maybe I want a crop of width 512, and height covering the whole image?
            
            # Let's allow Target W/H to be defined.
            # BUT, if we want to support "Resizing", that's a different node usually.
            # The prompt mentions "crop_scale" issues in Ditto.
            # Let's implement strict Cropping for now.
            
            # Refined Target Dimensions Force Constraint:
            # If "fixed_width", we ignore input height and use source height? 
            # No, that's just passing through.
            # Let's stick to the interpretation: "Calculate the rectangle size to crop".
            
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
