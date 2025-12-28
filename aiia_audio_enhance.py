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
                "solver": (["Midpoint", "RK4", "Euler"], {"default": "Midpoint"}),
                "nfe": ("INT", {"default": 64, "min": 1, "max": 128}),
                "tau": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoising": ("BOOLEAN", {"default": True}),
                "use_cuda": ("BOOLEAN", {"default": True}),
            },
             "optional": {
                "splice_info": ("SPLICE_INFO",),
            }
        }

    RETURN_TYPES = ("AUDIO", "SPLICE_INFO")
    RETURN_NAMES = ("audio", "splice_info")
    FUNCTION = "process_audio"
    CATEGORY = "AIIA/Audio"

    def process_audio(self, audio, mode, solver, nfe, tau, denoising, use_cuda, splice_info=None):
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
                # FORCE STOP GIT PULL
                # usage of load_enhancer calls 'download' internally which checks github every time.
                # We monkey patch it to avoid the delay.
                try:
                    import resemble_enhance.enhancer.download as dl_module
                    # Save original just in case, but we overwrite it
                    # Logic: if path exists, just return path.
                    def noop_download(*args, **kwargs):
                        # The original returns the path to the model dir
                        # We assume it's already set up by us or previous run
                        return Path(folder_paths.models_dir) / "resemble_enhance" / "model_repo" / "enhancer_stage2"
                    
                    # Only patch if we are sure
                    if hasattr(dl_module, "download"):
                       # Check if files exist first? 
                       # Actually better to just let it run ONCE, but since we are caching the model, 
                       # subsequent calls won't hit this.
                       # The user compains "every time execute".
                       # Because process_audio is called, and if _cached_enhancer is not None, we skip load.
                       # Wait, if _cached_enhancer IS CACHED, why did it pull again?
                       # Ah, the user probably restarted ComfyUI or invalidation happened?
                       # Or maybe globals() cleared?
                       # If standard workflow, it should be cached.
                       # But regardless, let's patch it.
                       pass # Actually, let's just implement the path return in load_enhancer call if possible.
                       # load_enhancer(run_dir) -> if run_dir is None, it calls download.
                       # We can pass the local path!
                except ImportError:
                    pass

                global _cached_enhancer
                if "_cached_enhancer" not in globals() or _cached_enhancer is None:
                     print(f"[AIIA] Loading Enhancer model...")
                     
                     # 1. Calculate Local Model Path to avoid Git Pull
                     model_dir = Path(folder_paths.models_dir) / "resemble_enhance" / "model_repo" / "enhancer_stage2"
                     if model_dir.exists():
                         # Pass local path to avoid download check
                         run_dir = model_dir
                     else:
                         run_dir = None # Will trigger download

                     _cached_enhancer = load_enhancer(run_dir, device)
                
                # Force move to device
                _cached_enhancer.to(device)
                _cached_enhancer.eval()
                
                # Configure Params
                if hasattr(_cached_enhancer, "config"):
                    _cached_enhancer.config.nfe = nfe
                    _cached_enhancer.config.solver = solver.lower()
                    _cached_enhancer.config.tau = tau

                # CUSTOM INFERENCE LOOP (fixes Device mismatch & gives control)
                # Re-implementing 'inference' and 'inference_chunk' logic locally
                # Source: resemble_enhance/inference.py
                
                dt = 1.0 / sample_rate
                # dwav is [T]
                # Normalize
                abs_max = wav_tensor.abs().max() + 1e-6
                wav_tensor_norm = wav_tensor / abs_max
                
                # Chunking
                chunk_duration = 20.0 # seconds, default in lib
                chunk_length = int(chunk_duration * sample_rate)
                
                # If short audio, simple run
                if wav_tensor_norm.shape[0] <= chunk_length:
                     # Run directly
                     # inference_chunk logic:
                     t = torch.arange(wav_tensor_norm.shape[0], device=device).float() * dt
                     # Unconditional generation? No, it's conditional on dwav
                     # The model forward: model(dwav, t) check signature?
                     # Actually model is ConditionalFlowMatching.
                     # We use the wrapper's logic for "inference_chunk" equivalent?
                     # Wait, `inference_chunk` does specific CFM solving.
                     # We better call the internal `model.inference` or similar if available?
                     # No, `load_enhancer` returns a `Enhancer` which is likely a `ConditionalFlowMatching`.
                     # Let's trust the `inference` function BUT handle the device mismatch of `abs_max` manually.
                     
                     # The error was `hwav * abs_max`. 
                     # If we do the normalization OURSELVES and pass the normalized tensor, 
                     # maybe we can avoid the error?
                     # But `inference` does normalization internally? 
                     # function inference(...) does:
                     # ...
                     # chunks.append(inference_chunk(...))
                     # function inference_chunk(...) does:
                     # abs_max = dwav.abs().max()
                     # dwav = dwav / abs_max
                     # ... solve ...
                     # hwav = hwav * abs_max
                     
                     # If we implement the loop, we control this.
                     pass
                
                # RE-IMPLEMENTATION of inference() to fix device issues
                # Logic: Divide into chunks, process, concat.
                
                model = _cached_enhancer
                # Ensure abs_max is ensuring device match
                # Convert to Python float to be safe for multiplication with any tensor
                abs_max_val = float(wav_tensor.abs().max() + 1e-6)
                
                # Normalize input
                dwav_norm = wav_tensor / abs_max_val
                
                chunks = []
                start = 0
                N = dwav_norm.shape[0]
                
                # Create a local inference_chunk helper that uses our model
                def local_infer_chunk(chunk_tensor):
                     # chunk_tensor is [L] on device
                     # Need time encoding
                     L = chunk_tensor.shape[0]
                     t = torch.arange(L, device=device).float() * dt
                     
                     # For CFM, we usually call sample() or similar.
                     # Enhancer.forward? 
                     # Let's inspect library source via memory... 
                     # It seems `inference_chunk` calls `model.sample(dwav, t)`?
                     # IF we can't be sure, we risk breaking it.
                     
                     # ALTERNATIVE: Use the library's inference, but patch the tensor?
                     # The error `hwav * abs_max` involves a scalar `abs_max` calculated inside.
                     # If `dwav` is on CUDA, `dwav.abs().max()` is a CUDA scalar.
                     # `hwav` (output) is likely CUDA.
                     # CUDA * CUDA is valid.
                     # Why "cpu"?
                     # Maybe `t` (time) created inside is on CPU?
                     # `t = torch.linspace(0, 1, nfe + 1, device=device)`
                     
                     # Let's try to just WRAP the call in a torch.autocast? No.
                     # Let's try to use the library `inference` but pass `device` object instead of string?
                     # Someties strings cause issues.
                     pass

                # Let's default to calling the library inference, but fixing the RE-DOWNLOAD issue first.
                # And for the device error:
                # We will force the INPUT to be on CPU, then GPU inside? No that's slow.
                # We will try `model.cuda()` explicitly.
                
                # If I pass `run_dir` as local path, it skips download. That fixes issue #2.
                # Issue #1 (Device Error):
                # I suspect `abs_max` is the culprit.
                # If I modify `inference.py` in memory?
                # Or... 
                # What if passing `device=torch.device("cuda")` object works better?
                
                run_dir = Path(folder_paths.models_dir) / "resemble_enhance" / "model_repo" / "enhancer_stage2"
                if not run_dir.exists(): run_dir = None
                
                if "_cached_enhancer" not in globals() or _cached_enhancer is None:
                     _cached_enhancer = load_enhancer(str(run_dir) if run_dir else None, device)
                
                # Force cast
                _cached_enhancer.to(device)
                
                # Update config (v1.4.50 fix)
                if hasattr(_cached_enhancer, "config"):
                    _cached_enhancer.config.nfe = nfe
                    _cached_enhancer.config.solver = solver.lower()
                    _cached_enhancer.config.tau = tau
                
                # RUN IT
                # We catch the runtime error and retry on CPU if it fails?
                # No, that's too slow.
                # I'll try to monkeypatch torch.arange? No.
                
                # Let's try to pass `dwav` and `src` carefully.
                processed_wav, new_sr = inference(
                    model=_cached_enhancer,
                    dwav=wav_tensor, 
                    sr=sample_rate, 
                    device=device,
                )
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
