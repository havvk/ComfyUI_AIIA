"""
EchoMimicV3 distributed utilities stub.
These are multi-GPU / sequence-parallel utilities not needed for single-GPU ComfyUI inference.
All functions are no-op stubs that return safe defaults.
"""


def parallel_magvit_vae(*args, **kwargs):
    """No-op decorator for single-GPU VAE decoding."""
    def decorator(fn):
        return fn
    return decorator


def get_sequence_parallel_rank():
    """Return rank 0 for single-GPU."""
    return 0


def get_sequence_parallel_world_size():
    """Return world size 1 for single-GPU."""
    return 1


def get_sp_group():
    """Return None (no sequence parallel group)."""
    return None


def set_multi_gpus_devices(*args, **kwargs):
    """No-op for single-GPU inference."""
    pass


class xFuserLongContextAttention:
    """Stub for xFuser long context attention (multi-GPU only)."""
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "xFuserLongContextAttention requires multi-GPU setup. "
            "Single-GPU inference uses standard attention instead."
        )
