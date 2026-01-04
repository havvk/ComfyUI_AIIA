
import torch
import sys

def inspect_data(name, v, detail=False):
    if isinstance(v, torch.Tensor):
        info = f"{name}: Tensor {v.shape} {v.dtype} device={v.device}"
        
        # Check for NaNs and Infs
        if v.numel() > 0:
            v_float = v.float() # Cast to float for stats
            has_nan = torch.isnan(v_float).any().item()
            has_inf = torch.isinf(v_float).any().item()
            min_val = v_float.min().item()
            max_val = v_float.max().item()
            mean_val = v_float.mean().item()
            
            info += f" range=[{min_val:.2e}, {max_val:.2e}] mean={mean_val:.2e}"
            if has_nan: info += " [HAS NAN]"
            if has_inf: info += " [HAS INF]"
            
        if detail and v.numel() < 20:
            info += f" val={v.tolist()}"
        elif detail:
             # Print start and end
             if v.dim() == 1:
                 info += f" head={v[:5].tolist()} tail={v[-5:].tolist()}"
             elif v.dim() == 2:
                 info += f" head={v[0, :5].tolist()} tail={v[0, -5:].tolist()}"
        print("    " + info)
    elif isinstance(v, (list, tuple)):
        print(f"    {name}: {type(v).__name__} len={len(v)}")
        if len(v) > 0:
            inspect_data(f"{name}[0]", v[0], detail=False) # Check first item type
            if isinstance(v[0], tuple): # Nested tuple (e.g. KV Cache)
                print(f"      {name}[0] is tuple len={len(v[0])}")
                if len(v[0]) > 0:
                     inspect_data(f"{name}[0][0]", v[0][0], detail=False)

def inspect_preset(path):
    print(f"\n--- Inspecting {path} ---")
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            # Sort keys for consistent output
            for k in sorted(data.keys()):
                v = data[k]
                print(f"Key: {k}")
                if k in ["neg_input_ids", "tts_lm_input_ids", "tts_text_masks", "neg_tts_text_masks"]:
                    inspect_data(k, v, detail=True)
                elif isinstance(v, dict):
                    print(f"  Type: Dict with keys: {list(v.keys())}")
                    for sub_k, sub_v in v.items():
                        inspect_data(sub_k, sub_v)
                else:
                    inspect_data(k, v)
        else:
            print(f"Root Type: {type(data)}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if __name__ == "__main__":
    files = [
        "/app/ComfyUI/models/vibevoice/voices/streaming_model/en-Carter_man.pt",
        "/app/ComfyUI/models/vibevoice/voices/streaming_model/my_new_voice_preset.pt"
    ]
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    
    for f in files:
        inspect_preset(f)
