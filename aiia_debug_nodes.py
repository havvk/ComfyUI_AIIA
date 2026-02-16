import os
import time
import folder_paths

class AIIA_TextDebugSplicer:
    NODE_NAME = "Text Debug Splicer (AIIA)"
    CATEGORY = "AIIA/Debug"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "title_1": ("STRING", {"default": "Text 1", "multiline": False}),
                "title_2": ("STRING", {"default": "Text 2", "multiline": False}),
                "title_3": ("STRING", {"default": "Text 3", "multiline": False}),
                "separator": (["New Line (\\n)", "Double New Line (\\n\\n)", "Dash Line (---)", "Hash Line (###)", "Star Line (***)"], {"default": "Dash Line (---)"}),
                "save_prefix": ("STRING", {"default": "debug_spliced"}),
                "save_file": ("BOOLEAN", {"default": True}),
                "return_text": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "text_1": ("STRING", {"forceInput": True}),
                "text_2": ("STRING", {"forceInput": True}),
                "text_3": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_text",)
    FUNCTION = "splice_text"
    OUTPUT_NODE = True

    def splice_text(self, title_1, title_2, title_3, separator, save_prefix, save_file, return_text, text_1=None, text_2=None, text_3=None):
        
        # 1. Parsing Separator
        sep_map = {
            "New Line (\\n)": "\n",
            "Double New Line (\\n\\n)": "\n\n",
            "Dash Line (---)": "\n---\n",
            "Hash Line (###)": "\n###\n",
            "Star Line (***)": "\n***\n"
        }
        
        actual_sep = sep_map.get(separator, "\n")
        
        parts = []
        
        # 2. Building Parts
        if text_1:
            if title_1.strip():
                parts.append(f"[{title_1}]\n{text_1}")
            else:
                parts.append(text_1)
                
        if text_2:
            if title_2.strip():
                parts.append(f"[{title_2}]\n{text_2}")
            else:
                parts.append(text_2)
                
        if text_3:
            if title_3.strip():
                parts.append(f"[{title_3}]\n{text_3}")
            else:
                parts.append(text_3)
                
        combined_text = actual_sep.join(parts)
        
        # 3. Saving to File
        if save_file and combined_text.strip():
            output_dir = folder_paths.get_output_directory()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{save_prefix}_{timestamp}.txt"
            full_path = os.path.join(output_dir, filename)
            
            try:
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(combined_text)
                print(f"[AIIA Debug] Saved text to {full_path}")
            except Exception as e:
                print(f"[AIIA Debug] Failed to save text: {e}")
                
        return (combined_text if return_text else "",)

NODE_CLASS_MAPPINGS = {
    "AIIA_TextDebugSplicer": AIIA_TextDebugSplicer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_TextDebugSplicer": "Text Debug Splicer (AIIA)"
}
