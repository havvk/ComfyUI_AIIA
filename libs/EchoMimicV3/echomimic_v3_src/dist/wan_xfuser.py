"""
EchoMimicV3 xFuser attention stub.
usp_attn_forward is a multi-GPU sequence-parallel attention replacement.
Not needed for single-GPU ComfyUI inference.
"""


def usp_attn_forward(*args, **kwargs):
    """No-op stub. Single-GPU uses standard attention path."""
    raise NotImplementedError(
        "usp_attn_forward requires xFuser multi-GPU setup. "
        "Single-GPU inference uses standard attention instead."
    )
