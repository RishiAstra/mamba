from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor

def ssd_combined_fwd(
    x: Tensor,
    dt: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dt_softplus: bool,
    chunk_size: int,
    z: Optional[Tensor],
    dt_bias: Optional[Tensor],
    initial_states: Optional[Tensor],
    seq_idx: Optional[Tensor],
    cu_seqlens: Optional[Tensor],
    out: Optional[Tensor],
    out_x: Optional[Tensor],
    dt_out: Optional[Tensor],
    dA_cumsum: Optional[Tensor],
    states: Optional[Tensor],
    final_states: Optional[Tensor],
    /,
) -> Tuple[
    Optional[Tensor],    # out
    Optional[Tensor],    # out_x
    Optional[Tensor],   # dt
    Optional[Tensor],   # dA_cumsum
    Optional[Tensor],   # states
    Optional[Tensor],    # final_states
]:
    """
    Fused Mamba SSD forward (C++/CUDA), not fully implemented yet (some outputs will be None).
    """
