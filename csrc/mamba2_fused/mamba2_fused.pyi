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
    cb_force_fp32: bool = False,
    z: Optional[Tensor] = None,
    dt_bias: Optional[Tensor] = None,
    initial_states: Optional[Tensor] = None,
    seq_idx: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    out_x: Optional[Tensor] = None,
    dt_out: Optional[Tensor] = None,
    dA_cumsum: Optional[Tensor] = None,
    states: Optional[Tensor] = None,
    final_states: Optional[Tensor] = None,
    CB: Optional[Tensor] = None,
    /,
) -> Tuple[
    Optional[Tensor],    # out
    Optional[Tensor],    # out_x
    Optional[Tensor],    # dt
    Optional[Tensor],    # dA_cumsum
    Optional[Tensor],    # states
    Optional[Tensor],    # final_states
]:
    """
    Fused Mamba SSD forward (C++/CUDA), not fully implemented yet (some outputs will be None).
    """
