# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

from typing import Optional

import math
from packaging import version

import torch
import torch.nn.functional as F
from torch import Tensor
from mamba_ssm.utils.torch import custom_bwd, custom_fwd

import triton
import triton.language as tl

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, causal_conv1d_bwd_function, causal_conv1d_update_function
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None
    causal_conv1d_update_function = None

from mamba_ssm.ops.triton.ssd_fused5 import _fused5_ssd
from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_state_fwd, chunk_state_varlen
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn, _layer_norm_fwd, _layer_norm_bwd
from mamba_ssm.ops.triton.k_activations import _swiglu_fwd, _swiglu_bwd

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


def rearrange_and_update_stride(tensor, pattern=None, dim=2):
    # ensure tensor.stride(dim) is a multiple of eight after rearranging according to pattern,
    # if not call contiguous(), rearrange only if pattern is not None
    tensor_rearranged = rearrange(tensor, pattern) if pattern is not None else tensor
    return tensor_rearranged.contiguous() if tensor_rearranged.stride(dim) % 8 != 0 else tensor_rearranged


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, D_ptr,
    b_ptr, dstates_ptr,
    dx_ptr, ddt_ptr, dD_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_D_head,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        # scale = tl.exp(dA_cs_last - dA_cs_m)
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        # scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0)
    # Might be faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    # However, we're getting error with the Triton compiler 2.1.0 for that code path:
    # Unexpected mma -> mma layout conversion
    # Triton 2.2.0 fixes this
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate)
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    # x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    # x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    # dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    # dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    # ddt = tl.sum(acc * x, axis=1) * dt_m
    # ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    # tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        # For some reason setting mask to (offs_m[:, None] < chunk_size_limit) is much slower
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0).to(tl.float32)
        # cb *= tl.exp(dA_cs_k[None, :] - dA_cs_m[:, None])
        cb *= tl.exp(tl.minimum((dA_cs_k[None, :] - dA_cs_m[:, None]), 0.0))
        # If we don't have the (k + offs_k[None, :] < K_MAX) mask, for indices outside this range,
        # we might have dA_cs_m = 0.0 and dA_cs_k very negative, and tl.exp will return inf.
        # Multiplying with cb, which is 0.0 outside the range, will make the result NaN.
        # This will cause NaN in acc, and hence NaN in dx and ddt.
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


def _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                        batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x, CB, dout, dt, dA_cumsum, seq_idx, D, B, dstates, dx, ddt, dD,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(-1), CB.stride(-2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            D.stride(0) if D is not None else 0,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return dx, ddt.to(dtype=dt.dtype), dD


def _query_start_loc_to_chunk_indices_offsets(query_start_loc: torch.Tensor,
                                              chunk_size: int,
                                              total_seqlens: int):

    cu_seqlens = query_start_loc[1:]  # remove prepended 0

    # outputs will have length expansion of chunks that do not divide
    # chunk_size
    N = math.ceil(total_seqlens / chunk_size) + (cu_seqlens[:-1] % chunk_size
                                                 > 0).sum()
    chunk_indices = torch.arange(N,
                                 dtype=torch.int,
                                 device=query_start_loc.device)
    chunk_offsets = torch.zeros((N, ),
                                dtype=torch.int,
                                device=query_start_loc.device)

    p = 0  # num of insertions
    for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):

        # if does not divide chunk_size, then there is one chunk insertion
        p += (s % chunk_size > 0)

        # get the dimensions
        # - the + 1 for _e is to shift the boundary by one chunk
        # - this shifting is not needed if chunk_size divides e
        _s, _e = s // chunk_size + p, e // chunk_size + p + (e % chunk_size
                                                             > 0)

        # adjust indices and offsets
        chunk_indices[_s:_e] -= p
        chunk_offsets[_s] = s % chunk_size

    return chunk_indices, chunk_offsets


def _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), use_fused5_ssd=False):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    seq_idx = torch.zeros((1, seqlen), dtype=torch.int32, device='cuda')
    chunk_indices, chunk_offsets = \
            _query_start_loc_to_chunk_indices_offsets(
                torch.tensor((0, seqlen), dtype=torch.int32, device='cpu'), chunk_size, seqlen)
    
    chunk_indices = chunk_indices.to('cuda')
    chunk_offsets = chunk_offsets.to('cuda')

    if use_fused5_ssd: # all 5 kernels fused
        out, out_x, states, final_states, dA_cumsum, dt = _fused5_ssd(
            x, dt, A, B, C, D,
            chunk_size=chunk_size, initial_states=initial_states, seq_idx=seq_idx, z=z,
            states_in_fp32=False, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit,
        )
    else: # original
        assert batch == 1
        seq_idx = seq_idx.squeeze(0)
        x = x.squeeze(0)
        dt = dt.squeeze(0)
        A = A.squeeze(0)
        B = B.squeeze(0)
        C = C.squeeze(0)
        # # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
        # dA_cumsum_tmp0, dt_tmp0 = _chunk_cumsum_fwd(dt[:, :147], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
        # dA_cumsum_tmp1, dt_tmp1 = _chunk_cumsum_fwd(dt[:, 147:], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
        # dA_cumsum_tmp2, dt_tmp2 = _chunk_cumsum_fwd(dt[:, 147:256], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
        dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
        states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
        # states_tmp0 = _chunk_state_fwd(B[:, :147], x[:, :147], dt_tmp0, dA_cumsum_tmp0, states_in_fp32=True)
        # states_tmp1 = _chunk_state_fwd(B[:, 147:], x[:, 147:], dt_tmp1, dA_cumsum_tmp1, states_in_fp32=True)
        # states_tmp2 = _chunk_state_fwd(B[:, 147:256], x[:, 147:256], dt_tmp2, dA_cumsum_tmp2, states_in_fp32=True)
        # states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
        #                                         initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
        #                                         seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
        states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, -1],
                                    initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                    seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
        states = rearrange(states, "... (p n) -> ... p n", n=dstate)
        # states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
        # states_tmp0 = rearrange(_state_passing_fwd(rearrange(states_tmp0, "... p n -> ... (p n)"), dA_cumsum_tmp0[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
        # states_tmp1 = rearrange(_state_passing_fwd(rearrange(states_tmp1, "... p n -> ... (p n)"), dA_cumsum_tmp1[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
        CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
        out, _ = _chunk_scan_fwd( CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, initial_states=initial_states,)

    if cu_seqlens is None:
        return out, None, dt, dA_cumsum, states, None # TODO: final_states returned?
    else:
        assert batch == 1, "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        varlen_states = chunk_state_varlen(B.squeeze(0), x.squeeze(0), dt.squeeze(0), dA_cumsum.squeeze(0),
                                           cu_seqlens, states.squeeze(0))
        return out, None, dt, dA_cumsum, states, None, varlen_states



def selective_scan_bwd(dout, x, dt, A, B, C, D=None, z=None):
    """
    Argument:
        dout: (batch, seqlen, nheads, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size) or (batch, nheads, headdim, nchunks, chunk_size)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    import selective_scan

    batch, seqlen, nheads, headdim = x.shape
    chunk_size = dt.shape[-1]
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    x = rearrange(x, "b l h p -> b (h p) l")
    squeeze_dt = dt.dim() == 4
    if dt.dim() == 4:
        dt = repeat(dt, "b h c l -> b h p c l", p=headdim)
    dt = rearrange(dt, "b h p c l -> b (h p) (c l)", p=headdim)
    squeeze_A = A.dim() == 1
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    B = rearrange(B, "b l g n -> b g n l")
    C = rearrange(C, "b l g n -> b g n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")

    if x.stride(-1) != 1:
        x = x.contiguous()
    if dt.stride(-1) != 1:
        dt = dt.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    _, intermediate, *rest = selective_scan.fwd(x, dt.to(dtype=x.dtype), A, B, C, D, z, None, False)
    if z is not None:
        out = rest[0]
    else:
        out = None

    dout = rearrange(dout, "b l h p -> b (h p) l")

    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
    # backward of selective_scan with the backward of chunk).
    # Here we just pass in None and dz will be allocated in the C++ code.
    _, ddt, dA, *rest = selective_scan.bwd(
        x, dt.to(dtype=x.dtype), A, B, C, D, z, None, dout, intermediate, out, None, False,
        False  # option to recompute out_z, not used here
    )
    ddt = rearrange(ddt, "b (h p) (c l) -> b h p c l", p=headdim, l=chunk_size)
    if squeeze_dt:
        ddt = ddt.float().sum(dim=2)
    if squeeze_A:
        dA = rearrange(dA, "(h p) n -> h p n", p=headdim).sum(dim=(1, 2))
    return ddt, dA


class MambaChunkScanCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, return_varlen_states=False, use_fused5_ssd=False):
        ctx.dt_dtype = dt.dtype
        if not return_varlen_states:
            cu_seqlens = None
        else:
            assert cu_seqlens is not None, "cu_seqlens must be provided if return_varlen_states is True"
        out, out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit, use_fused5_ssd=use_fused5_ssd)
        ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx)
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.return_varlen_states = return_varlen_states
        if not return_varlen_states:
            return out if not return_final_states else (out, final_states)
        else:
            varlen_states = rest[0]
            return (out, varlen_states) if not return_final_states else (out, final_states, varlen_states)

    @staticmethod
    def backward(ctx, dout, *args):
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, return_varlen_states=False, use_fused5_ssd=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    return MambaChunkScanCombinedFn.apply(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, dt_limit, return_final_states, return_varlen_states, use_fused5_ssd)


def ssd_selective_scan(x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
        A: (nheads) or (dim, dstate)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,) or (nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    x = rearrange(x, "b l h p -> b (h p) l")
    if dt.dim() == 3:
        dt = repeat(dt, "b l h -> b l h p", p=headdim)
    dt = rearrange(dt, "b l h p -> b (h p) l")
    if A.dim() == 1:
        A = repeat(A, "h -> (h p) n", p=headdim, n=dstate).to(dtype=torch.float32)
    else:
        A = A.to(dtype=torch.float32)
    B = rearrange(B, "b l g n -> b g n l")
    C = rearrange(C, "b l g n -> b g n l")
    if D is not None:
        if D.dim() == 2:
            D = rearrange(D, "h p -> (h p)")
        else:
            D = repeat(D, "h -> (h p)", p=headdim)
    if z is not None:
        z = rearrange(z, "b l h p -> b (h p) l")
    if dt_bias is not None:
        if dt_bias.dim() == 1:
            dt_bias = repeat(dt_bias, "h -> h p", p=headdim)
        dt_bias = rearrange(dt_bias, "h p -> (h p)")
    if dt_limit != (0.0, float("inf")):
        if dt_bias is not None:
            dt = dt + rearrange(dt_bias, "d -> d 1")
        if dt_softplus:
            dt = F.softplus(dt)
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1]).to(x.dtype)
        dt_bias = None
        dt_softplus = None
    out = selective_scan_fn(x, dt, A, B, C, D=D, z=z, delta_bias=dt_bias, delta_softplus=dt_softplus)
    return rearrange(out, "b (h p) l -> b l h p", p=headdim)


def mamba_conv1d_scan_ref(xBC, conv1d_weight, conv1d_bias, dt, A, chunk_size, D=None, z=None,
                          dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")),
                          activation="silu", headdim=None, ngroups=1):
    """
    Argument:
        xBC: (batch, seqlen, dim + 2 * ngroups * dstate) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt: (batch, seqlen, nheads) or (batch, seqlen, nheads, headdim)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, dim)
        dt_bias: (nheads) or (nheads, headdim)
        headdim: if D is 1D and z is None, headdim must be passed in
    Return:
        out: (batch, seqlen, dim)
    """
    batch, seqlen, nheads = dt.shape[:3]
    assert nheads % ngroups == 0
    if z is not None:
        dim = z.shape[-1]
        assert dim % nheads == 0
        headdim = dim // nheads
    else:
        if D.dim() == 1:
            assert headdim is not None
        else:
            headdim = D.shape[1]
        dim = nheads * headdim
    xBC = rearrange(causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation),
                    "b d s -> b s d")
    dstate = (xBC.shape[-1] - dim) // ngroups // 2
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(), z=z, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    return rearrange(out, "b s h p -> b s (h p)")


class MambaSplitConv1dScanCombinedFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu",
                rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None,
                ngroups=1, norm_before_gate=True, use_fused5_ssd=False):
        assert activation in [None, "silu", "swish"]
        if D.dim() == 1:
            assert headdim is not None
            nheads, = D.shape
        else:
            nheads, headdim = D.shape
        batch, seqlen, _ = zxbcdt.shape
        dim = nheads * headdim
        assert nheads % ngroups == 0
        dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
        d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
        assert d_nonssm >= 0
        assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv = rearrange(
            causal_conv1d_fwd_function(rearrange_and_update_stride(xBC, "b s d -> b d s"),
                                                 conv1d_weight, conv1d_bias, seq_idx, None, None, activation in ["silu", "swish"]),
            "b d s -> b s d"
        )
        x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
        x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
        C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
        z = rearrange(z, "b l (h p) -> b l h p", h=nheads) if z is not None else None
        if rmsnorm_weight is None:
            out, out_x, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, use_fused5_ssd=use_fused5_ssd)
            out = rearrange(out, "b s h p -> b s (h p)")
            rstd = None
            if d_nonssm > 0:
                out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)
        else:
            out_x, _, dt_out, dA_cumsum, states, final_states = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=None, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, dt_softplus=True, dt_limit=dt_limit, use_fused5_ssd=use_fused5_ssd)
            # reshape input data into 2D tensor
            x_rms = rearrange(out_x, "b s h p -> (b s) (h p)")
            z_rms = rearrange(z, "b s h p -> (b s) (h p)")
            rmsnorm_weight = rmsnorm_weight.contiguous()
            if d_nonssm == 0:
                out = None
            else:
                out01 = torch.empty((batch, seqlen, d_nonssm + dim), dtype=x_rms.dtype, device=x_rms.device)
                out = rearrange(out01[..., d_nonssm:], "b s d -> (b s) d")
                _swiglu_fwd(zx0, out=out01[..., :d_nonssm])
            out, _, rstd = _layer_norm_fwd(x_rms, rmsnorm_weight, None, rmsnorm_eps, z_rms, out=out,
                                           group_size=dim // ngroups,
                                           norm_before_gate=norm_before_gate, is_rms_norm=True)
            if d_nonssm == 0:
                out = rearrange(out, "(b s) d -> b s d", b=batch)
            else:
                out = out01
        ctx.outproj_weight_dtype = outproj_weight.dtype if outproj_weight is not None else None
        if outproj_weight is not None:
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
            out = F.linear(out, outproj_weight, outproj_bias)
        else:
            assert outproj_bias is None
        ctx.save_for_backward(zxbcdt, conv1d_weight, conv1d_bias,
                              out_x, A, D, dt_bias, initial_states, seq_idx, rmsnorm_weight, rstd, outproj_weight, outproj_bias)
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.activation = activation
        ctx.rmsnorm_eps = rmsnorm_eps
        ctx.norm_before_gate = norm_before_gate
        ctx.chunk_size = chunk_size
        ctx.headdim = headdim
        ctx.ngroups = ngroups
        return out if not return_final_states else (out, final_states)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout, *args):
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


def mamba_split_conv1d_scan_combined(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states=None, seq_idx=None, dt_limit=(0.0, float("inf")), return_final_states=False, activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True, use_fused5_ssd=False):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen), int32
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    return MambaSplitConv1dScanCombinedFn.apply(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, initial_states, seq_idx, dt_limit, return_final_states, activation, rmsnorm_weight, rmsnorm_eps, outproj_weight, outproj_bias, headdim, ngroups, norm_before_gate, use_fused5_ssd=use_fused5_ssd)


def mamba_split_conv1d_scan_ref(zxbcdt, conv1d_weight, conv1d_bias, dt_bias, A, D, chunk_size, dt_limit=(0.0, float("inf")), activation="silu", rmsnorm_weight=None, rmsnorm_eps=1e-6, outproj_weight=None, outproj_bias=None, headdim=None, ngroups=1, norm_before_gate=True):
    """
    Argument:
        zxbcdt: (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads) where dim == nheads * headdim
        conv1d_weight: (dim + 2 * ngroups * dstate, width)
        conv1d_bias: (dim + 2 * ngroups * dstate,)
        dt_bias: (nheads,)
        A: (nheads)
        D: (nheads, headdim) or (nheads,)
        rmsnorm_weight: (dim,)
        outproj_weight: (out_dim, dim)
        outproj_bias: (out_dim,)
        headdim: if D is 1D, headdim must be passed in
        norm_before_gate: if True, we do RMSNorm(x) * F.silu(z). If False, we do RMSNorm(x * F.silu(z))
    Return:
        out: (batch, seqlen, dim)
    """
    if D.dim() == 1:
        assert headdim is not None
        nheads, = D.shape
    else:
        nheads, headdim = D.shape
    assert nheads % ngroups == 0
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    dstate = (zxbcdt.shape[-1] - 2 * dim - nheads) // ngroups // 2
    assert zxbcdt.shape == (batch, seqlen, 2 * dim + 2 * ngroups * dstate + nheads)
    assert dt_bias.shape == (nheads,)
    assert A.shape == (nheads,)
    if rmsnorm_weight is not None:
        assert rmsnorm_weight.shape == (dim,)
    z, xBC, dt = torch.split(zxbcdt, [dim, dim + 2 * ngroups * dstate, nheads], dim=-1)
    xBC = rearrange(causal_conv1d_fn(rearrange(xBC, "b s d -> b d s"), conv1d_weight, conv1d_bias, activation=activation),
                    "b d s -> b s d")
    x, B, C = torch.split(xBC, [dim, ngroups * dstate, ngroups * dstate], dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p", h=nheads)
    out = ssd_selective_scan(x, dt.to(x.dtype), A, B, C, D=D.float(),
                             z=z if rmsnorm_weight is None else None, dt_bias=dt_bias, dt_softplus=True, dt_limit=dt_limit)
    out = rearrange(out, "b s h p -> b s (h p)")
    if rmsnorm_weight is not None:
        out = rmsnorm_fn(out, rmsnorm_weight, None, z=rearrange(z, "b l h p -> b l (h p)"), eps=rmsnorm_eps,
                         norm_before_gate=norm_before_gate)
    if outproj_weight is not None:
        out = F.linear(out, outproj_weight, outproj_bias)
    return out

