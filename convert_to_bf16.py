import torch
import sys
import os

def convert_to_bf16(path):
    print(f"Loading {path}...")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Load failed: {e}. Trying without weights_only...")
        data = torch.load(path, map_location="cpu")
    
    new_data = {}
    
    def cast_recursive(d):
        if isinstance(d, torch.Tensor):
            if d.dtype == torch.float16 or d.dtype == torch.float32:
                return d.to(torch.bfloat16)
            return d
        elif isinstance(d, dict):
            return {k: cast_recursive(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [cast_recursive(v) for v in d]
        elif isinstance(d, tuple):
            return tuple(cast_recursive(v) for v in d)
        return d

    new_data = cast_recursive(data)
    
    new_path = path.replace(".pt", "_bf16.pt")
    print(f"Saving to {new_path}...")
    torch.save(new_data, new_path)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_to_bf16(sys.argv[1])
    else:
        print("Usage: python convert_to_bf16.py <preset_path>")
