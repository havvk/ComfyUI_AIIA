import torch
import os
import sys
import subprocess
import folder_paths
import soundfile as sf
import numpy as np
import tempfile
from pathlib import Path

# Lazy-loaded globals
enhance = None
denoise = None

def _mock_deepspeed():
    """
    Mock DeepSpeed via sys.meta_path to prevent import errors.
    DeepSpeed is a heavy training dependency that is hard to install.
    We use a custom Importer to intercept ALL deepspeed submodules dynamically.
    """
    if "deepspeed" in sys.modules:
        return

    # Check if we already registered the finder
    # We can check sys.meta_path, but it's hard to identify our specific instance class properly if we reload.
    # We'll just define the class and check if any instance of it is in meta_path? 
    # Or simpler: just proceed.
    
    print("[AIIA] Mocking DeepSpeed via sys.meta_path...")
    from importlib.abc import MetaPathFinder, Loader
    from importlib.machinery import ModuleSpec
    from unittest.mock import MagicMock

    class DeepSpeedImportMocker(MetaPathFinder, Loader):
        def find_spec(self, fullname, path, target=None):
            if fullname.startswith("deepspeed"):
                return ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            m = MagicMock()
            m.__path__ = []
            m.__file__ = "mock_deepspeed_dynamic.py"
            m.__loader__ = self
            m.__spec__ = spec
            return m

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, DeepSpeedImportMocker())
    
    # Also inject main module just in case, but make it a PACKAGE
    m = MagicMock()
    m.__path__ = [] # This makes it a package
    m.__file__ = "mock_deepspeed_root.py"
    m.__name__ = "deepspeed"
    m.__package__ = "deepspeed"
    m.__spec__ = None
    
    sys.modules["deepspeed"] = m

def _install_resemble_if_needed():
    global enhance, denoise
    if enhance is not None:
        return

    # 1. Pre-emptively mock DeepSpeed
    # This ensures that if the library is installed, the import succeeds immediately
    # avoiding the "Not Found" -> "Install" -> "Mock" -> "Retry" loop.
    _mock_deepspeed()

    # 2. Try import first
    try:
        from resemble_enhance.enhancer.inference import enhance as eh, denoise as dn, load_enhancer, inference
        # Inject into global namespace so we can use them
        globals()["load_enhancer"] = load_enhancer
        globals()["inference"] = inference
        enhance = eh
        denoise = dn
        return
    except ImportError:
        pass
        
    try:
        # Fallback import
        from resemble_enhance.inference import enhance as eh, denoise as dn, load_enhancer, inference
        globals()["load_enhancer"] = load_enhancer
        globals()["inference"] = inference
        enhance = eh
        denoise = dn
        return
    except ImportError:
        pass

    print("[AIIA] resemble-enhance not found (or broken). Installing without dependencies to protect environment...")
    try:
        # 3. Install resemble-enhance without dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "resemble-enhance", "--no-deps"])
        
        # 4. Check critical deps
        try:
            import hjson
        except ImportError:
             subprocess.check_call([sys.executable, "-m", "pip", "install", "hjson"])
            
        print("[AIIA] resemble-enhance installed.")
        
        # 5. Retry import
        try:
            from resemble_enhance.enhancer.inference import enhance as eh, denoise as dn, load_enhancer, inference
            globals()["load_enhancer"] = load_enhancer
            globals()["inference"] = inference
            enhance = eh
            denoise = dn
        except ImportError:
             try:
                from resemble_enhance.inference import enhance as eh, denoise as dn, load_enhancer, inference
                globals()["load_enhancer"] = load_enhancer
                globals()["inference"] = inference
                enhance = eh
                denoise = dn
             except ImportError as e:
                 print(f"[AIIA] Failed to import resemble_enhance even after install. Error: {e}")
                 import traceback
                 traceback.print_exc()
                 enhance = None
                 denoise = None

    except Exception as e:
        print(f"[AIIA] Error installing resemble-enhance: {e}")


def setup_resemble_model_path():
    """
    Redirect resemble-enhance model download to ComfyUI/models/resemble_enhance
    Resemble Enhance uses: ~/.cache/resemble-enhance/
    """
    try:
        # 1. Target Directory (ComfyUI/models/resemble_enhance)
        comfy_models_dir = folder_paths.models_dir
        target_dir = os.path.join(comfy_models_dir, "resemble_enhance")
        
        # 2. Source Cache Directory
        cache_dir = Path.home() / ".cache" / "resemble-enhance"
        
        # If cache_dir is already a symlink pointing to target_dir, we are good.
        if os.path.islink(cache_dir):
            if os.readlink(cache_dir) == target_dir:
                return
            else:
                os.unlink(cache_dir)

        # Prepare target directory
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
            print(f"[AIIA] Created Resemble Enhance model directory: {target_dir}")

        # If cache dir exists and is a real folder
        if os.path.exists(cache_dir) and not os.path.islink(cache_dir):
            print(f"[AIIA] Migrating Resemble Enhance models from {cache_dir}...")
            import shutil
            for item in os.listdir(cache_dir):
                s = os.path.join(cache_dir, item)
                d = os.path.join(target_dir, item)
                if not os.path.exists(d):
                    shutil.move(s, d)
                else:
                    if os.path.isdir(s): shutil.rmtree(s)
                    else: os.remove(s)
            try:
                os.rmdir(cache_dir)
            except:
                pass

        # Create symlink
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        if not os.path.exists(cache_dir):
            try:
                os.symlink(target_dir, cache_dir)
                print(f"[AIIA] Linked {cache_dir} -> {target_dir}")
            except OSError:
                print(f"[AIIA] Warning: Failed to create symlink for Resemble Enhance.")

        # Trigger download if empty to ensure files are there (it normally downloads on first run)
        # But we let the inference call handle it via the library's internal check
        
    except Exception as e:
        print(f"[AIIA] Error setting up model path: {e}")

class AIIA_Audio_Enhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["Enhance (Denoise + Bandwidth Ext)", "Denoise Only"], {"default": "Enhance (Denoise + Bandwidth Ext)"}),
                "solver": (["Midpoint", "RK4", "Euler"],),
                "nfe": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
                "tau": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0.0 = Keep Original Noise, 1.0 = Full Denoise. Increase to remove artifacts/hum."}),
                "chunk_seconds": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            }, "optional": {
                "splice_info": ("SPLICE_INFO",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SPLICE_INFO")
    RETURN_NAMES = ("audio", "splice_info")
    FUNCTION = "process_audio"
    CATEGORY = "AIIA/Audio"

    def process_audio(self, audio, mode, solver, nfe, tau, denoise_strength, chunk_seconds, overlap_seconds, use_cuda, splice_info=None):
        _install_resemble_if_needed()
        if enhance is None:
             raise ImportError("resemble-enhance library is not available.")

        setup_resemble_model_path()
        
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        
        # Prepare Input
        waveform = audio["waveform"] # [B, C, T]
        sample_rate = audio["sample_rate"]
        
        if waveform.ndim == 3:
            wav_tensor = waveform[0] # [C, T]
        else:
            wav_tensor = waveform
            
        # Resemble Enhance expects torch tensor on device, shape [C, T] or [1, T]?
        # Looking at library: run(dwav, sr, device) -> dwav is torch tensor
        # It handles resampling internally? 
        # Actually inference.py takes (process_audio_file) or (waveform, sample_rate)
        # We need to make sure we pass the right thing.
        
        # Convert to mono if needed?
        # The library seems to handle it, but typically it expects [1, T] or [T]
        # ERROR FIXED: Library expects [T] (1D), got 2D.
        if wav_tensor.shape[0] > 1:
            # Simple mixdown for enhancement stability
             wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=False)
        elif wav_tensor.dim() == 2 and wav_tensor.shape[0] == 1:
             wav_tensor = wav_tensor.squeeze(0)
        
        wav_tensor = wav_tensor.to(device)
        
        print(f"[AIIA] Running Resemble Enhance ({mode}) on {device}...")
        
        try:
            if mode == "Denoise Only":
                # denoise(waveform: Tensor, sample_rate: int, device: str)
                # It returns (waveform, new_sr)
                processed_wav, new_sr = denoise(wav_tensor, sample_rate, device)
            else:
                # 1. FIX CUDA Device Mismatch (Monkey Patch)
                try:
                    import resemble_enhance.inference as inference_mod
                    import torch.nn.functional as F
                    
                    # Store original if not already patched
                    if not hasattr(inference_mod, "original_inference_chunk"):
                        inference_mod.original_inference_chunk = inference_mod.inference_chunk
                        
                    # Define safe chunk inference (Based on inspected source)
                    def safe_inference_chunk(model, dwav, sr, device, npad=441):
                        # 1. Calculate abs_max as FLOAT (CPU Scalar)
                        # This prevents "Expected all tensors to be on same device" when multiplying later
                        abs_max = float(dwav.abs().max().clamp(min=1e-7))
                        
                        # 2. Normalize
                        dwav = dwav.to(device)
                        dwav = dwav / abs_max
                        
                        # 3. Valid Padding (from original source)
                        dwav = F.pad(dwav, (0, npad))
                        
                        # 4. Run Model
                        # We need to reconstruct the ODE solver call manually because we are bypassing the library's chunk runner.
                        # The library does: hwav = model.ode_solve(dwav, t, solver=..., tau=...)
                        
                        # Retrieve parameters
                        # Priority: 1. model.hp (if we injected it), 2. model.config, 3. Defaults
                        nfe = 64
                        solver = "midpoint"
                        tau = 0.5
                        
                        if hasattr(model, "hp"):
                             nfe = getattr(model.hp, "nfe", nfe)
                             solver = getattr(model.hp, "solver", solver)
                             tau = getattr(model.hp, "tau", tau)
                        elif hasattr(model, "config"):
                             nfe = getattr(model.config, "nfe", nfe)
                             solver = getattr(model.config, "solver", solver)
                             tau = getattr(model.config, "tau", tau)
                             
                        # DEBUG: Trace Execution
                        print(f"[AIIA DEBUG] safe_inference_chunk: NFE={nfe}, Solver={solver}, Tau={tau}")
                        
                        t = torch.linspace(0, 1, nfe + 1, device=device)
                        
                        # EXECUTE
                        # Note: Inspection confirms this model uses .forward() which internally calls ode_solve
                        # with the injected parameters (nfe, solver, tau).
                        
                        if hasattr(model, "ode_solve"):
                             # If explicitly present, use it.
                             hwav = model.ode_solve(dwav, t, solver=solver, tau=tau)
                        else:
                             # Standard Path for EnhancerStage2
                             print(f"[AIIA DEBUG] Running model forward pass (NFE={nfe})...")
                             out = model(dwav.unsqueeze(0))
                             if isinstance(out, tuple): out = out[0]
                             hwav = out[0]

                        # 6. Unnormalize with FLOAT
                        # hwav is on GPU
                        hwav = hwav * abs_max
                        
                        # 7. Strict Length Trimming (Fix Tensor Mismatch)
                        # We must return exactly 'length' samples (original length before padding)
                        # dwav was padded by npad.
                        # Original logic: length = dwav.shape[-1] - npad; hwav = hwav[:length]
                        target_length = dwav.shape[-1] - npad
                        if hwav.shape[-1] > target_length:
                            hwav = hwav[..., :target_length]
                        elif hwav.shape[-1] < target_length:
                             # Pad if too short (rare but possible with some models)
                             hwav = F.pad(hwav, (0, target_length - hwav.shape[-1]))
                        
                        return hwav.cpu()

                    # Apply Patch 1: Inference Chunk
                    inference_mod.inference_chunk = safe_inference_chunk
                    print("[AIIA] Monkey-patched inference_chunk for CUDA safety (v4).")
                    
                    # Apply Patch 2: Merge Chunks (Safety Net)
                    if not hasattr(inference_mod, "original_merge_chunks"):
                        inference_mod.original_merge_chunks = inference_mod.merge_chunks
                        
                    def safe_merge_chunks(chunks, *args, **kwargs):
                        # Force CPU
                        cpu_chunks = [c.cpu() for c in chunks]
                        return inference_mod.original_merge_chunks(cpu_chunks, *args, **kwargs)
                        
                    inference_mod.merge_chunks = safe_merge_chunks
                    print("[AIIA] Monkey-patched merge_chunks for CPU safety.")
                    
                except ImportError:
                    pass

                global _cached_enhancer
                if "_cached_enhancer" not in globals() or _cached_enhancer is None:
                     print(f"[AIIA] Loading Enhancer model...")
                     
                     # 2. LOCATE MODEL (to skip download)
                     # Priority 1: ComfyUI/models/resemble_enhance
                     paths_to_check = [
                         Path(folder_paths.models_dir) / "resemble_enhance" / "model_repo" / "enhancer_stage2",
                     ]
                     
                     # Priority 2: site-packages/resemble_enhance/model_repo (Standard Install)
                     try:
                         import resemble_enhance
                         site_pkg_path = Path(resemble_enhance.__file__).parent / "model_repo" / "enhancer_stage2"
                         paths_to_check.append(site_pkg_path)
                     except:
                         pass
                         
                     run_dir = None
                     for p in paths_to_check:
                         if p.exists():
                             run_dir = p
                             print(f"[AIIA] Found Resume Enhance model at: {run_dir}")
                             break
                     
                     # ERROR FIXED: load_enhancer expects Path object, not string (it uses / operator)
                     _cached_enhancer = load_enhancer(run_dir if run_dir else None, device)
                
                # Force move to device
                _cached_enhancer.to(device)
                _cached_enhancer.eval()
                
                # Configure Params: FORCE INJECT EVERYWHERE (to fix Quality/Stripes)

                # 5. Inject Parameters into the Model (Official Method)
                # Found 'configurate_' in Enhancer source. Much cleaner than manual attribute hacking.
                if hasattr(_cached_enhancer, "configurate_"):
                    try:
                         # lambd defaults to 0.5 in 'enhance' function, but we use tau for flow matching guidance.
                         # The enhance function uses lambd=0.5, tau=0.5.
                         # We passed 'tau' in argument. Using it for both if consistent?
                         # Actually, 'lambd' controls denoiser strength (0-1).
                         # 'tau' controls CFM prior temperature (0-1).
                         # The node inputs say 'tau' (CFM). 
                         # Let's set lambd to a reasonable default or expose it? 
                         # The user node implementation doesn't seem to expose 'lambd' (denoise strength) separately?
                         # Wait, the node HAS 'denoise' input? No, only 'nfe', 'solver', 'tau', 'chunk_seconds'...
                         # Let's check INPUT_TYPES.
                         # Ah, wait. If I look at my previous analysis, `lambd` is used for "denoiser strength".
                         # If `lambd=0`, denoiser is skipped.
                         # Generally we want denoiser + enhancer.
                         # For now, I'll set lambd=0.5 (default) or 1.0?
                         # The library default is 0.5.
                         
                         _cached_enhancer.configurate_(nfe=nfe, solver=solver.lower(), lambd=denoise_strength, tau=tau)
                         print(f"[AIIA DEBUG] Configured model: nfe={nfe}, solver={solver.lower()}, lambd={denoise_strength}, tau={tau}")
                    except Exception as e:
                         print(f"[AIIA WARNING] configurate_ failed: {e}")
                else:
                     print("[AIIA WARNING] Model has no configurate_ method. Using fallback injection.")
                     try:
                         if hasattr(_cached_enhancer, "config"):
                             object.__setattr__(_cached_enhancer.config, "nfe", nfe)
                             object.__setattr__(_cached_enhancer.config, "solver", solver)
                             object.__setattr__(_cached_enhancer.config, "tau", tau)
                         
                         if hasattr(_cached_enhancer, "hp"):
                             object.__setattr__(_cached_enhancer.hp, "nfe", nfe)
                             object.__setattr__(_cached_enhancer.hp, "solver", solver)
                             object.__setattr__(_cached_enhancer.hp, "tau", tau)
                             
                         # CRITICAL: Update internal state variables used by forward()
                         if hasattr(_cached_enhancer, "_eval_lambd"):
                              _cached_enhancer._eval_lambd = tau # Fallback behavior
    
                         if hasattr(_cached_enhancer, "lcfm") and hasattr(_cached_enhancer.lcfm, "_eval_tau"):
                              _cached_enhancer.lcfm._eval_tau = tau
                     except Exception as e:
                         print(f"[AIIA WARNING] Parameter injection failed: {e}")

                # 6. Monkey-Patch Inference (if not already)
                import resemble_enhance.inference as inference_mod

                target_len_samples = 0

                def safe_inference_chunk(model, dwav, sr, device):
                    nonlocal target_len_samples
                    
                    # 0. Normalization (Required for correct enhancement)
                    abs_max = dwav.abs().max()
                    dwav = dwav / (abs_max + 1e-8)
                    
                    # Record the "Standard Cache Size" (usually the first chunk)
                    current_len = dwav.shape[-1]
                    if target_len_samples == 0:
                        target_len_samples = current_len
                        
                    # DYNAMIC PADDING (AVOID RE-COMPILE)
                    # If this chunk is smaller than our standard cache (e.g. last chunk),
                    # pad it to match the standard size so we hit the JIT cache.
                    original_len = current_len
                    has_padded = False
                    
                    if current_len < target_len_samples:
                        pad_diff = target_len_samples - current_len
                        # Pad with silence at the end
                        dwav = F.pad(dwav, (0, pad_diff))
                        has_padded = True
                        print(f"[AIIA DEBUG] Padding last chunk to {target_len_samples} samples to hit JIT cache...")

                    # 1. Padding (Alignment for Model)
                    # PAD to be divisible by 64 (or whatever the model needs)
                    n_fft = 1024
                    hop_length = 256
                    
                    # Pad logic from original
                    pad_len = (dwav.shape[-1] // hop_length + 1) * hop_length - dwav.shape[-1]
                    npad = pad_len
                    dwav = F.pad(dwav, (0, npad), mode='constant', value=0)

                    # 2. To Device
                    dwav = dwav.to(device)
                    
                    # 3. Create Time Steps
                    t = torch.linspace(0, 1, nfe + 1, device=device)
                    
                    # EXECUTE
                    # Use standard forward pass (verified to trigger CFM Solver if configured correctly)
                    # We injected _eval_lambd/tau above to ensure it runs.
                    
                    if hasattr(model, "ode_solve"):
                         hwav = model.ode_solve(dwav, t, solver=solver, tau=tau)
                    else:
                         # print(f"[AIIA DEBUG] Running model forward pass (NFE={nfe}, len={dwav.shape[-1]})...")
                         out = model(dwav.unsqueeze(0))
                         if isinstance(out, tuple): out = out[0]
                         hwav = out[0]

                    # 6. Unnormalize with FLOAT
                    # hwav is on GPU here. 'abs_max' might be on CPU (float).
                    hwav = hwav * abs_max
                    
                    # 7. Strict Length Trimming (Fix Tensor Mismatch)
                    # First, undo alignment padding
                    aligned_len = dwav.shape[-1] - npad
                    if hwav.shape[-1] != aligned_len:
                        if hwav.shape[-1] > aligned_len:
                            hwav = hwav[..., :aligned_len]
                        else:
                            hwav = F.pad(hwav, (0, aligned_len - hwav.shape[-1]))
                            
                    # Second, undo Compiliation Padding (if applied)
                    if has_padded:
                        if hwav.shape[-1] > original_len:
                             hwav = hwav[..., :original_len]
                    
                    return hwav.cpu()


                # Apply Patch 1: Inference Chunk
                inference_mod.inference_chunk = safe_inference_chunk
                print("[AIIA] Monkey-patched inference_chunk for CUDA safety (v4).")
                
                # Apply Patch 2: Merge Chunks (Safety Net)
                if not hasattr(inference_mod, "original_merge_chunks"):
                    inference_mod.original_merge_chunks = inference_mod.merge_chunks

                def safe_merge_chunks(chunks, *args, **kwargs):
                    # Force all chunks to CPU before merging
                    try:
                        cpu_chunks = [c.cpu() if hasattr(c, 'cpu') else c for c in chunks]
                        return inference_mod.original_merge_chunks(cpu_chunks, *args, **kwargs)
                    except RuntimeError as e:
                         print(f"[AIIA ERROR] Merge chunks failed: {e}")
                         # Debug shapes
                         for i, c in enumerate(chunks):
                             if hasattr(c, 'shape'):
                                 print(f"Chunk {i}: shape={c.shape}, device={c.device}")
                         raise e

                inference_mod.merge_chunks = safe_merge_chunks
                print("[AIIA] Monkey-patched merge_chunks for CPU safety.")

                def run_inference_safe(active_device):
                    # Force move model
                    _cached_enhancer.to(active_device)
                    
                    # CLEANUP: Force mel_fn to CPU (ghost fix)
                    try:
                        import sys
                        if "resemble_enhance.inference" in sys.modules:
                             mod = sys.modules["resemble_enhance.inference"]
                             if hasattr(mod, "mel_fn") and hasattr(mod.mel_fn, "to"):
                                 mod.mel_fn.to("cpu")
                                 
                        if "resemble_enhance.audio" in sys.modules:
                             mod = sys.modules["resemble_enhance.audio"]
                             if hasattr(mod, "mel_fn") and hasattr(mod.mel_fn, "to"):
                                 mod.mel_fn.to("cpu")
                    except:
                        pass
                    
                    return inference(
                        model=_cached_enhancer,
                        dwav=wav_tensor.to(active_device), 
                        sr=sample_rate, 
                        device=active_device,
                    )

                try:
                    # Try on requested device
                    processed_wav, new_sr = run_inference_safe(device)
                except RuntimeError as e:
                    if "device" in str(e) and device != "cpu":
                        print(f"[AIIA] Device mismatch error on {device}. Retrying on CPU (fallback)...")
                        # Clear cache or just move?
                        try:
                            processed_wav, new_sr = run_inference_safe("cpu")
                        except Exception as e2:
                            raise e2
                    else:
                        raise e
        except Exception as e:
            print(f"Error in resemble-enhance: {e}")
            raise e

        # Post-process
        # processed_wav is typically on device
        processed_wav = processed_wav.cpu()
        
        # Output format [1, C, T]
        if processed_wav.ndim == 1:
            processed_wav = processed_wav.unsqueeze(0).unsqueeze(0)
        elif processed_wav.ndim == 2:
            processed_wav = processed_wav.unsqueeze(0)
            
        # Handle Output SR (usually 44100)
        result_sr = new_sr

        # Splice Info Scaling
        new_splice_info = None
        if splice_info is not None:
             import copy
             new_splice_info = copy.deepcopy(splice_info)
             old_rate = splice_info.get("sample_rate", sample_rate)
             if old_rate != result_sr:
                 scale = result_sr / old_rate
                 new_splice_info["splice_points"] = [int(p * scale) for p in new_splice_info.get("splice_points", [])]
                 new_splice_info["sample_rate"] = result_sr
                 new_splice_info["scale_factor"] = scale

        return ({"waveform": processed_wav, "sample_rate": result_sr}, new_splice_info)

NODE_CLASS_MAPPINGS = {
    "AIIA_Audio_Enhance": AIIA_Audio_Enhance
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIIA_Audio_Enhance": "Audio AI Enhance (Resemble)"
}
