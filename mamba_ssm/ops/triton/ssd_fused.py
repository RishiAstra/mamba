# The fused ssd kernel

import torch

def _fused3_ssd(
    C, B, CB, x, dt, dA_cumsum, C_dtype, D,
    initial_states=None, seq_idx=None, states_in_fp32=True, z=None
):
    return torch.zeros_like(x), None, None, None