"""
EchoMimicV3 distributed utilities stub.
parallel_magvit_vae is a multi-GPU decorator for VAE decoding.
In single-GPU / ComfyUI inference, it's a no-op passthrough.
"""

def parallel_magvit_vae(*args, **kwargs):
    """No-op decorator for single-GPU inference."""
    def decorator(fn):
        return fn
    return decorator
