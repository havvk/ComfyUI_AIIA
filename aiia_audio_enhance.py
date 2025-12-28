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
                # 1. FIX CUDA Device Mismatch (Monkey Patch)
                # The library crashes on 'hwav * abs_max' because abs_max is a CUDA tensor scalar
                # and PyTorch sometimes dislikes mixing it with model output if types mismatch strictly?
                # Or simply making it a float is safer.
                try:
                    import resemble_enhance.inference as inference_mod
                    
                    # Store original if not already patched
                    if not hasattr(inference_mod, "original_inference_chunk"):
                        inference_mod.original_inference_chunk = inference_mod.inference_chunk
                        
                    # Define safe chunk inference
                    def safe_inference_chunk(model, chunk, sr, device):
                        # Chunk is [T]
                        # 1. Calculate abs_max as FLOAT (CPU Scalar)
                        # This prevents "Expected all tensors to be on same device" when multiplying later
                        abs_max = float(chunk.abs().max()) + 1e-6
                        
                        # 2. Normalize
                        chunk = chunk / abs_max
                        
                        # 3. Call original logic?
                        # Using original logic but we need to inject the float abs_max handling.
                        # Since we can't inject into a compiled function, we must reimplement the FEW lines 
                        # of inference_chunk from source.
                        # Source:
                        # def inference_chunk(model, dwav, sr, device):
                        #     abs_max = dwav.abs().max()
                        #     dwav = dwav / abs_max  <- we did this
                        #     dwav = dwav.to(device)
                        #     t = torch.linspace(0, 1, model.config.nfe + 1, device=device)
                        #     hwav = model.ode_solve(dwav, t, solver=model.config.solver, tau=model.config.tau)
                        #     hwav = hwav * abs_max
                        #     return hwav
                        
                        chunk = chunk.to(device)
                        # Fix: Handle missing config by providing defaults or injecting
                        nfe = getattr(model.config, "nfe", 64) if hasattr(model, "config") else 64
                        t = torch.linspace(0, 1, model.config.nfe + 1, device=device)
                        
                        # Solver
                        # Check if model has ode_solve (EnhancerStage2)
                        if hasattr(model, "ode_solve"):
                             hwav = model.ode_solve(chunk, t, solver=solver_val, tau=tau_val)
                        else:
                             # Fallback if API changes
                             hwav = chunk
                             
                        # 4. Unnormalize with FLOAT
                        hwav = hwav * abs_max
                        return hwav

                    # Apply Patch
                    inference_mod.inference_chunk = safe_inference_chunk
                    print("[AIIA] Monkey-patched resemble_enhance.inference_chunk for CUDA safety.")
                    
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
                
                # Configure Params: FORCE INJECT Config if missing
                if not hasattr(_cached_enhancer, "config"):
                    class Config: pass
                    _cached_enhancer.config = Config()
                    
                _cached_enhancer.config.nfe = nfe
                _cached_enhancer.config.solver = solver.lower()
                _cached_enhancer.config.tau = tau

                def run_inference_safe(active_device):
                    # Force move model
                    _cached_enhancer.to(active_device)
                    _cached_enhancer.eval()
                    
                    # Fix: Move global mel_fn if it exists (fixes 'stft input and window' error)
                    # The library uses a global mel_fn for chunk merging, which stays on CPU by default.
                    try:
                        import resemble_enhance.inference as inference_mod_inner
                        if hasattr(inference_mod_inner, "mel_fn") and hasattr(inference_mod_inner.mel_fn, "to"):
                            inference_mod_inner.mel_fn.to(active_device)
                    except:
                        pass
                        
                    try:
                        import resemble_enhance.audio as audio_mod
                        if hasattr(audio_mod, "mel_fn") and hasattr(audio_mod.mel_fn, "to"):
                             audio_mod.mel_fn.to(active_device)
                    except:
                        pass
                    
                    # Run
                    return inference(
                        model=_cached_enhancer,
                        dwav=wav_tensor.to(active_device), 
                        sr=sample_rate, 
                        device=active_device,
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
