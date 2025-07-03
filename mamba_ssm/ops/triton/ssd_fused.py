# The fused-5 ssd kernel

import math
import torch
import triton
import triton.language as tl

from packaging import version

from mamba_ssm.ops.triton.softplus import softplus
TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_HD': 64, 'BLOCK_SIZE_DS': 64, 'BLOCK_SIZE_CS': 64,
            'SP_BLOCK_SIZE_DS': 64,
            'CS_BLOCK_SIZE_CS': 64, 'CS_BLOCK_SIZE_DS': 32,
            'BMM_BLOCK_SIZE_M': 64, 'BMM_BLOCK_SIZE_N': 64, 'BMM_BLOCK_SIZE_K': 64, 'BMM_STAGES': 2,
            'CCS_BLOCK_SIZE_H': 16,
            }, num_stages=1, num_warps=4, maxnreg=128),
    ],
    key=['hdim', 'dstate', 'chunk_size', 'IS_CAUSAL'],
)
# NOTE: this kernel assumes that a warp resident in an SM (already executed some instructions) is not permanently starved if other warps are spamming atomic instructions
@triton.jit
def _fused5_ssd_kernel(
    ccs_wait_ptr, ccs_wait_stride_batch, ccs_wait_stride_chunk,
    bmm_wait_ptr, bmm_wait_stride_batch, bmm_wait_stride_chunk,
    grid_atomic, USE_ATOMIC_PID: tl.constexpr,
    sync_atomic, stride_sync_batch, stride_sync_head, stride_sync_hdim, stride_sync_dstate,

    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio, nheads, nchunks, ngroups,
    ########################################
    # Originally Chunk State
    ########################################
    # Pointers to matrices
    x_ptr, b_ptr, states_L_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_states_L_batch, stride_states_L_chunk, stride_states_L_head, stride_states_L_hdim, stride_states_L_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    
    ########################################
    # Originally State Passing
    ########################################
    # Pointers to matrices
    states_G_ptr, final_states_ptr, dA_ccs_ptr, initstates_ptr, # ccs = chunk_cumsum
    # Strides
    stride_states_G_batch, stride_states_G_chunk, stride_states_G_head, stride_states_G_hdim, stride_states_G_dstate,
    stride_final_states_batch, stride_final_states_head, stride_final_states_hdim, stride_final_states_dstate,
    stride_dA_ccs_batch, stride_dA_ccs_chunk, stride_dA_ccs_head,
    stride_initstates_batch, stride_initstates_head, stride_initstates_hdim, stride_initstates_dstate,

    ########################################
    # Originally Chunk Scan
    ########################################
    # Pointers to matrices
    cb_ptr, z_ptr, out_ptr, out_x_ptr, C_ptr, D_ptr,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_D_head,


    ########################################
    # Originally Chunk Cumsum
    ########################################
    # Pointers to matrices
    A_ptr, dt_bias_ptr, dt_orig_ptr,
    # Matrix dimension
    dt_min, dt_max,
    # Strides
    stride_dt_orig_batch, stride_dt_orig_seqlen, stride_dt_orig_head,
    stride_A_head,
    stride_dt_bias_head,


    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_HD: tl.constexpr, BLOCK_SIZE_DS: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr, CS_BLOCK_SIZE_CS: tl.constexpr,
    SP_BLOCK_SIZE_DS: tl.constexpr, 
    CS_BLOCK_SIZE_DS: tl.constexpr,
    BMM_BLOCK_SIZE_M: tl.constexpr,
    BMM_BLOCK_SIZE_N: tl.constexpr,
    BMM_BLOCK_SIZE_K: tl.constexpr,
    BMM_STAGES: tl.constexpr,
    CCS_BLOCK_SIZE_H: tl.constexpr, 
    # NOTE: not an autotune thing
    BLOCK_SIZE_DSTATE: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # some strides must be 64-bit to prevent int32 overflow
    # for now, only do the batch size for the largest things
    stride_x_batch = stride_x_batch.to(tl.int64)
    stride_b_batch = stride_b_batch.to(tl.int64)
    stride_z_batch = stride_z_batch.to(tl.int64)
    stride_out_batch = stride_out_batch.to(tl.int64)
    stride_C_batch = stride_C_batch.to(tl.int64)
    stride_states_L_batch = stride_states_L_batch.to(tl.int64)
    stride_states_G_batch = stride_states_G_batch.to(tl.int64)
    # the following are roughly ordered from largest to smallest
    # things with batch, seqlen, and ~32*128x more
    # stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    # stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    # stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    # stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    # stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    # stride_states_L_batch, stride_states_L_chunk, stride_states_L_head, stride_states_L_hdim, stride_states_L_dstate,
    # stride_states_G_batch, stride_states_G_chunk, stride_states_G_head, stride_states_G_hdim, stride_states_G_dstate,
    # # things with batch, seqlen, and ~32x more
    # stride_dt_orig_batch, stride_dt_orig_seqlen, stride_dt_orig_head,
    # stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    # stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    # # things with batch, seqlen
    # stride_seq_idx_batch, stride_seq_idx_seqlen,
    # stride_dA_ccs_batch, stride_dA_ccs_chunk, stride_dA_ccs_head,
    # # things with batch, ~32*128^2x more
    # stride_final_states_batch, stride_final_states_head, stride_final_states_hdim, stride_final_states_dstate,
    # stride_initstates_batch, stride_initstates_head, stride_initstates_hdim, stride_initstates_dstate,
    # # constant size things
    # stride_D_head,
    # stride_A_head,
    # stride_dt_bias_head,


    if USE_ATOMIC_PID:
        pid_og = tl.atomic_add(grid_atomic, 1)
    else:
        pid_og = tl.program_id(0)

    ccs_num_pid_h = tl.cdiv(nheads, CCS_BLOCK_SIZE_H)
    num_pids_css = batch * nchunks * ccs_num_pid_h
    num_pid_n = tl.cdiv(chunk_size, BMM_BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(chunk_size, BMM_BLOCK_SIZE_M)
    num_pids_bmm = num_pid_n * num_pid_m * batch * nchunks * ngroups

    if pid_og < num_pids_css:
        ########################################
        # Chunk Cumsum
        ########################################
        pid_ccs = pid_og
        pid_b = pid_ccs % batch
        pid_c = (pid_ccs // batch) % nchunks
        pid_h = (pid_ccs // (batch * nchunks)) % ccs_num_pid_h


        css_dt_ptr = dt_orig_ptr + pid_b * stride_dt_orig_batch + pid_c * chunk_size * stride_dt_orig_seqlen
        css_dt_out_ptr = dt_ptr + pid_b * stride_dt_batch + pid_c * stride_dt_chunk
        css_dA_cumsum_ptr = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

        offs_h = pid_h * CCS_BLOCK_SIZE_H + tl.arange(0, CCS_BLOCK_SIZE_H)
        offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
        dt_ptrs = css_dt_ptr + (offs_h[:, None] * stride_dt_orig_head + offs_c[None, :] * stride_dt_orig_seqlen)
        A_ptrs = A_ptr + offs_h * stride_A_head
        dt_out_ptrs = css_dt_out_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_csize)
        dA_cs_ptrs = css_dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

        dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
            dt += dt_bias[:, None]
        if DT_SOFTPLUS:
            dt = tl.where(dt <= 20.0, softplus(dt), dt)
        # As of Triton 2.2.0, tl.clamp is not available yet
        # TODO: clamp exists, check if reasonable to assume recent triton version
        # dt = tl.clamp(dt, dt_min, dt_max)
        dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
        dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
        tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
        A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dA = dt * A[:, None]
        dA_cs = tl.cumsum(dA, axis=1)
        tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))

        # mark progress
        tl.atomic_add(ccs_wait_ptr + pid_b * ccs_wait_stride_batch + pid_c * ccs_wait_stride_chunk, CCS_BLOCK_SIZE_H, sem='release')

    elif pid_og < num_pids_css + num_pids_bmm:
        ########################################
        # BMM (CB)
        ########################################
        pid_bmm = pid_og - num_pids_css
        pid_n = pid_bmm % num_pid_n
        pid_m = (pid_bmm // num_pid_n) % num_pid_m
        pid_b = (pid_bmm // (num_pid_n * num_pid_m)) % batch
        pid_c = (pid_bmm // (num_pid_n * num_pid_m * batch)) % nchunks
        pid_h = (pid_bmm // (num_pid_n * num_pid_m * batch * nchunks)) % ngroups
        if not IS_CAUSAL or pid_n * BMM_BLOCK_SIZE_N < (pid_m + 1) * BMM_BLOCK_SIZE_M:
            a_ptr = C_ptr + pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + pid_h * stride_C_head
            b_ptr_bmm = b_ptr + pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head
            if HAS_SEQ_IDX:
                bmm_seq_idx_ptr = seq_idx_ptr + pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

            offs_m = pid_m * BMM_BLOCK_SIZE_M + tl.arange(0, BMM_BLOCK_SIZE_M)
            offs_n = pid_n * BMM_BLOCK_SIZE_N + tl.arange(0, BMM_BLOCK_SIZE_N)
            offs_k = tl.arange(0, BMM_BLOCK_SIZE_K)
            a_ptrs = a_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k[None, :] * stride_C_dstate)
            b_ptrs = b_ptr_bmm + (offs_k[:, None] * stride_b_dstate + offs_n[None, :] * stride_b_seqlen)
            chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

            acc = tl.zeros((BMM_BLOCK_SIZE_M, BMM_BLOCK_SIZE_N), dtype=tl.float32)
            # for k in range(0, tl.cdiv(dstate, BMM_BLOCK_SIZE_K)):
            for k in tl.range(0, tl.cdiv(dstate, BMM_BLOCK_SIZE_K), num_stages=BMM_STAGES):
                a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate - k * BMM_BLOCK_SIZE_K), other=0.0).to(C_ptr.dtype.element_ty)
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < dstate - k * BMM_BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit), other=0.0).to(C_ptr.dtype.element_ty)
                acc += tl.dot(a, b)
                a_ptrs += BMM_BLOCK_SIZE_K * stride_C_dstate
                b_ptrs += BMM_BLOCK_SIZE_K * stride_b_dstate

            offs_m = pid_m * BMM_BLOCK_SIZE_M + tl.arange(0, BMM_BLOCK_SIZE_M)
            offs_n = pid_n * BMM_BLOCK_SIZE_N + tl.arange(0, BMM_BLOCK_SIZE_N)
            if HAS_SEQ_IDX:
                chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
                seq_idx_m = tl.load(bmm_seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
                seq_idx_n = tl.load(bmm_seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
                acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
            out = acc.to(cb_ptr.dtype.element_ty)

            out_ptr_cb = cb_ptr + pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_h * stride_cb_head
            out_ptrs_cb = out_ptr_cb + (stride_cb_csize_m * offs_m[:, None] + offs_n[None, :] * stride_cb_csize_k)
            tl.store(out_ptrs_cb, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
        # mark progress
        tl.atomic_add(bmm_wait_ptr + pid_b * bmm_wait_stride_batch + pid_c * bmm_wait_stride_chunk, BMM_BLOCK_SIZE_M * BMM_BLOCK_SIZE_N, sem='release')

    # is this threadblock for the last 3 fused kernels?
    if pid_og < (num_pids_css + num_pids_bmm):
        return
    pid_fused3 = pid_og - (num_pids_css + num_pids_bmm)

    # pids same for all 3 parts
    # all pids represent domain parallelism except pid_c for state passing
    num_pid_ds = tl.cdiv(dstate, BLOCK_SIZE_DS)
    num_pid_hd = tl.cdiv(hdim, BLOCK_SIZE_HD)
    pid_h = pid_fused3 % nheads
    pid_c = (pid_fused3 // (nheads)) % nchunks
    pid_b = (pid_fused3 // (nheads * nchunks)) % batch
    pid_hd = (pid_fused3 // (nheads * nchunks * batch)) % num_pid_hd


    # wait for this (batch, chunk)
    bmm_wait_ptr += pid_b * bmm_wait_stride_batch + pid_c * bmm_wait_stride_chunk
    bmm_wait_val = tl.atomic_add(bmm_wait_ptr, 0, sem='acquire')
    while bmm_wait_val < chunk_size * chunk_size * ngroups: # TODO: does not consider of block sizes do not divide chunk size
        bmm_wait_val = tl.atomic_add(bmm_wait_ptr, 0, sem='acquire')

    # wait for this (batch, chunk)
    ccs_wait_ptr += pid_b * ccs_wait_stride_batch + pid_c * ccs_wait_stride_chunk
    ccs_wait_val = tl.atomic_add(ccs_wait_ptr, 0, sem='acquire')
    while ccs_wait_val < nheads:
        ccs_wait_val = tl.atomic_add(ccs_wait_ptr, 0, sem='acquire')


    # advance ptrs up front to simplify and slightly reduce register pressure
    # does actually provide a small benefit vs the original separate ptrs per step
    states_G_ptr += pid_b * stride_states_G_batch + pid_h * stride_states_G_head + pid_c * stride_states_G_chunk
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    states_L_ptr += pid_b * stride_states_L_batch + pid_c * stride_states_L_chunk + pid_h * stride_states_L_head
    dA_ccs_ptr += pid_b * stride_dA_ccs_batch + pid_h * stride_dA_ccs_head + stride_dA_ccs_chunk * pid_c
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    sync_atomic += pid_b * stride_sync_batch + pid_h * stride_sync_head + pid_hd * stride_sync_hdim# + pid_ds * stride_sync_dstate

    ########################################
    # Chunk State
    ########################################

    # chunk state ptrs
    if HAS_SEQ_IDX:
        seq_idx_ptr_cs = seq_idx_ptr + pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    sp_num_pid_ds = tl.cdiv(dstate, SP_BLOCK_SIZE_DS)

    for pid_ds in range(0, num_pid_ds, 1):
        # chunk state offsets
        # NOTE: m ->hdim, n -> dstate, k -> chunk_size
        offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
        offs_ds = pid_ds * BLOCK_SIZE_DS + tl.arange(0, BLOCK_SIZE_DS)
        offs_cs = tl.arange(0, BLOCK_SIZE_CS)

        # chunk state ptr blocks
        x_ptrs_cs = x_ptr + (offs_hd[:, None] * stride_x_hdim + offs_cs[None, :] * stride_x_seqlen)
        b_ptrs_cs = b_ptr + (offs_ds[None, :] * stride_b_dstate + offs_cs[:, None] * stride_b_seqlen)
        dt_ptrs_cs = dt_ptr + offs_cs * stride_dt_csize
        dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
        dA_cumsum_ptrs_cs = dA_cumsum_ptr + offs_cs * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs_cs = seq_idx_ptr_cs + offs_cs * stride_seq_idx_seqlen

        # chunk state other setup
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        if HAS_SEQ_IDX:
            seq_idx_last = tl.load(seq_idx_ptr_cs + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

        # chunk state chunk_size loop
        acc = tl.zeros((BLOCK_SIZE_HD, BLOCK_SIZE_DS), dtype=tl.float32)
        for k in range(0, chunk_size_limit, BLOCK_SIZE_CS):
            x = tl.load(x_ptrs_cs, mask=(offs_hd[:, None] < hdim) & (offs_cs[None, :] < chunk_size_limit - k), other=0.0, eviction_policy='evict_first')
            b = tl.load(b_ptrs_cs, mask=(offs_cs[:, None] < chunk_size_limit - k) & (offs_ds[None, :] < dstate), other=0.0, eviction_policy='evict_first').to(tl.float32)
            dA_cs_k = tl.load(dA_cumsum_ptrs_cs, mask=offs_cs < chunk_size_limit - k, other=0.0).to(tl.float32)
            if HAS_SEQ_IDX:
                seq_idx_k = tl.load(seq_idx_ptrs_cs, mask=offs_cs < chunk_size_limit - k, other=-1)
            dt_k = tl.load(dt_ptrs_cs, mask=offs_cs < chunk_size_limit - k, other=0.0).to(tl.float32)
            if not HAS_SEQ_IDX:
                scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
            else:
                scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
            b *= scale[:, None]
            b = b.to(x_ptr.dtype.element_ty)
            acc += tl.dot(x, b)
            x_ptrs_cs += BLOCK_SIZE_CS * stride_x_seqlen
            b_ptrs_cs += BLOCK_SIZE_CS * stride_b_seqlen
            dt_ptrs_cs += BLOCK_SIZE_CS * stride_dt_csize
            dA_cumsum_ptrs_cs += BLOCK_SIZE_CS * stride_dA_cs_csize
            if HAS_SEQ_IDX:
                seq_idx_ptrs_cs += BLOCK_SIZE_CS * stride_seq_idx_seqlen
        states = acc.to(states_L_ptr.dtype.element_ty)

        # chunk state final store
        states_ptrs_cs = states_L_ptr + (offs_hd[:, None] * stride_states_L_hdim + offs_ds[None, :] * stride_states_L_dstate)
        c_mask = (offs_hd[:, None] < hdim) & (offs_ds[None, :] < dstate)
        tl.store(states_ptrs_cs, states, mask=c_mask, eviction_policy='evict_last')



    ########################################
    # State Passing
    ########################################

    # Instead of looping over chunks, we have pid_c
    # first sync (wait for previous), then all must load

    # sync
    # the atomic represents which pid_c is ready
    # therefore, wait for it to reach our pid_c
    sync_val = tl.atomic_add(sync_atomic, 0, sem='acquire')
    while sync_val < pid_c:
        sync_val = tl.atomic_add(sync_atomic, 0, sem='acquire')

    # sp_num_pid_hd = tl.cdiv(hdim, BLOCK_SIZE_HD)
    for pid_ds in range(0, sp_num_pid_ds, 1):
        offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
        offs_ds = pid_ds * SP_BLOCK_SIZE_DS + tl.arange(0, SP_BLOCK_SIZE_DS)
        states_ptrs = states_L_ptr + offs_hd[:, None] * stride_states_L_hdim + offs_ds[None, :] * stride_states_L_dstate
        state_G_ptrs = states_G_ptr + offs_hd[:, None] * stride_states_G_hdim + offs_ds[None, :] * stride_states_G_dstate
        final_states_ptrs = final_states_ptr + offs_hd[:, None] * stride_final_states_hdim + offs_ds[None, :] * stride_final_states_dstate

        main_mask = (offs_hd < hdim)[:, None] & (offs_ds < dstate)[None, :]

        # special case 0
        if pid_c == 0:
            if not HAS_INITSTATES:
                states_prev = tl.zeros((BLOCK_SIZE_HD, SP_BLOCK_SIZE_DS), dtype=tl.float32)
            else:
                initstates_ptrs = initstates_ptr + offs_hd[:, None] * stride_initstates_hdim + offs_ds[None, :] * stride_initstates_dstate
                states_prev = tl.load(initstates_ptrs, mask=main_mask, other=0.0).to(tl.float32)
            tl.store(state_G_ptrs, states_prev, mask=main_mask)
        else:
            # need to load states from previous one (but since already offset by 1, just pid_c)
            states_prev = tl.load(state_G_ptrs, mask=main_mask, other=0.0).to(tl.float32)

        # ptrs
        state_G_ptrs += stride_states_G_chunk # offset since 0 gets initial states

        new_states = tl.load(states_ptrs, mask=main_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)

        dA_cs = tl.load(dA_ccs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs)
        if HAS_SEQ_IDX:
            # TODO: need atomics here?
            seq_idx = tl.load(seq_idx_ptr + (min(pid_c * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            seq_idx_new = tl.load(seq_idx_ptr + (min((pid_c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
            scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
        states_mod = scale * states_prev + new_states
        if pid_c < nchunks - 1:
            tl.store(state_G_ptrs, states_mod, mask=main_mask)
        else:
            tl.store(final_states_ptrs, states_mod, mask=main_mask)

    # let the next one go
    tl.atomic_add(sync_atomic, 1, sem='release')



    ########################################
    # Chunk Scan
    ########################################
    # pids same for all 3 parts
    # all pids represent domain parallelism except pid_c for state passing
    # cs_num_pid_hd = tl.cdiv(hdim, BLOCK_SIZE_HD)
    cs_num_pid_cs = tl.cdiv(chunk_size, CS_BLOCK_SIZE_CS)

    seq_idx_ptr_og = seq_idx_ptr
    out_x_ptr_og = out_x_ptr
    z_ptr_og = z_ptr

    for pid_cs in range(0, cs_num_pid_cs, 1): # pid_other, num_pid_cs, num_pid_ds*CS_BLOCK_HD_MULT):
        if HAS_SEQ_IDX:
            seq_idx_ptr = seq_idx_ptr_og + pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

        offs_cs = pid_cs * CS_BLOCK_SIZE_CS + tl.arange(0, CS_BLOCK_SIZE_CS)
        offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
        dA_cs_m = tl.load(dA_cumsum_ptr + offs_cs * stride_dA_cs_csize, mask=offs_cs < chunk_size, other=0.0).to(tl.float32)

        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        if HAS_SEQ_IDX:
            seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
            seq_idx_m = tl.load(seq_idx_ptr + offs_cs * stride_seq_idx_seqlen, mask=offs_cs < chunk_size_limit, other=-1)
        acc = tl.zeros((CS_BLOCK_SIZE_CS, BLOCK_SIZE_HD), dtype=tl.float32)

        # Without the if (pid_c > -1), with Triton 2.1.0, I get
        # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
        # With Triton 2.2.0, this works
        if IS_TRITON_22 or pid_c > -1:
            # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
            offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else CS_BLOCK_SIZE_DS)
            C_ptrs = C_ptr + (offs_cs[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
            prev_states_ptrs = states_G_ptr + (offs_hd[None, :] * stride_states_G_hdim + offs_k_dstate[:, None] * stride_states_G_dstate)
            if not HAS_SEQ_IDX:
                scale_m = tl.exp(dA_cs_m)
            else:
                scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
            if BLOCK_SIZE_DSTATE <= 128:
                C = tl.load(C_ptrs, mask=(offs_cs[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_hd[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc = tl.dot(C, prev_states) * scale_m[:, None]
            else:
                for k in range(0, dstate, CS_BLOCK_SIZE_DS):
                    C = tl.load(C_ptrs, mask=(offs_cs[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
                    # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                    prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_hd[None, :] < hdim), other=0.0)
                    prev_states = prev_states.to(C_ptr.dtype.element_ty)
                    acc += tl.dot(C, prev_states)
                    C_ptrs += CS_BLOCK_SIZE_DS
                    prev_states_ptrs += CS_BLOCK_SIZE_DS
                acc *= scale_m[:, None]

        offs_k = tl.arange(0, CS_BLOCK_SIZE_DS)
        cb_ptrs = cb_ptr + (offs_cs[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
        x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_hd[None, :] * stride_x_hdim)
        dt_ptrs = dt_ptr + offs_k * stride_dt_csize
        dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
        K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_cs + 1) * CS_BLOCK_SIZE_CS, chunk_size_limit)
        for k in range(0, K_MAX, CS_BLOCK_SIZE_DS):
            cb = tl.load(cb_ptrs, mask=(offs_cs[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0, eviction_policy='evict_last').to(tl.float32)
            dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
            # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
            # So we don't need masking wrt seq_idx here.
            cb *= tl.exp((dA_cs_m[:, None] - dA_cs_k[None, :]))
            dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
            cb *= dt_k
            if IS_CAUSAL:
                mask = offs_cs[:, None] >= k + offs_k[None, :]
                cb = tl.where(mask, cb, 0.0)
            cb = cb.to(x_ptr.dtype.element_ty)
            x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_hd[None, :] < hdim), other=0.0, eviction_policy='evict_last')
            acc += tl.dot(cb, x)
            cb_ptrs += CS_BLOCK_SIZE_DS * stride_cb_csize_k
            x_ptrs += CS_BLOCK_SIZE_DS * stride_x_seqlen
            dt_ptrs += CS_BLOCK_SIZE_DS * stride_dt_csize
            dA_cumsum_ptrs += CS_BLOCK_SIZE_DS * stride_dA_cs_csize

        offs_out_m = pid_cs * CS_BLOCK_SIZE_CS + tl.arange(0, CS_BLOCK_SIZE_CS)
        offs_out_n = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)

        if HAS_D:
            if D_HAS_HDIM:
                D = tl.load(D_ptr + pid_h * stride_D_head + offs_hd, mask=offs_hd < hdim, other=0.0).to(tl.float32)
            else:
                D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
            x_residual = tl.load(x_ptr + (offs_cs[:, None] * stride_x_seqlen + offs_hd[None, :] * stride_x_hdim),
                                mask=(offs_cs[:, None] < chunk_size_limit) & (offs_hd[None, :] < hdim), other=0.0).to(tl.float32)
            acc += x_residual * D

        if HAS_Z:
            out_x_ptr = out_x_ptr_og + pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
            out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
            tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

            z_ptr = z_ptr_og + pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
            z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
            z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
            acc *= z * tl.sigmoid(z)

        out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
        tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), eviction_policy='evict_first')


def _fused5_ssd(
    C, B, x, out_dtype, D,
    dt, A, chunk_size,
    initial_states=None, seq_idx=None, states_in_fp32=True, z=None, use_atomic_pid=True, # if True, don't count on 1d grid launching in order
    dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))
):
    batch, seqlen, nheads, hdim = x.shape
    # setup from chunk cumsum
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    # setup from chunk state
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    states_dtype = torch.float32 if states_in_fp32 else B.dtype
    states_L = torch.empty((batch, nchunks, nheads, hdim, dstate), device=x.device, dtype=states_dtype)
    # setup from bmm
    CB = torch.empty((batch, nchunks, ngroups, chunk_size, chunk_size), device=C.device, dtype=torch.float32)
    # setup from state passing
    dA_chunk_cumsum = dA_cumsum[:, :, :, -1]
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, hdim, dstate)
    if seq_idx is not None:
        assert chunk_size is not None
        assert seq_idx.shape == (batch, seqlen)
    states_G_dtype = states_L.dtype if out_dtype is None else out_dtype
    states_G = torch.empty((batch, nchunks, nheads, hdim, dstate), device=states_L.device, dtype=states_G_dtype)
    final_states = torch.empty((batch, nheads, hdim, dstate), device=states_L.device, dtype=torch.float32)
    # setup from chunk scan
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, hdim) or D.shape == (nheads,)
    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, hdim, device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, hdim, device=x.device, dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                  if z is not None else (0, 0, 0, 0))

    grid = lambda META: (
        # Chunk Cumsum grid
        batch * nchunks * triton.cdiv(nheads, META['CCS_BLOCK_SIZE_H']) +
        # BMM grid
        triton.cdiv(chunk_size, META['BMM_BLOCK_SIZE_N']) * triton.cdiv(chunk_size, META['BMM_BLOCK_SIZE_M']) * batch * nchunks * ngroups +
        # fused3 grid
        batch * nchunks * nheads * triton.cdiv(hdim, META['BLOCK_SIZE_HD'])
    ,)

    # 32 is for cache lines, dstate is not used here
    states_ready_size = batch * nheads * hdim * 32
    grid_atomic_size = 1 * 32
    bmm_ready_size = batch * nchunks * 32
    cs_ready_size = batch * nchunks * 32
    sync_atomic = torch.zeros((states_ready_size + grid_atomic_size + bmm_ready_size + cs_ready_size,), dtype=torch.int32, device=x.device)

    nheads_ngroups_ratio = nheads // ngroups
    _fused5_ssd_kernel[grid](
        # ccs_wait_ptr, ccs_wait_stride_batch, ccs_wait_stride_chunk,
        sync_atomic[states_ready_size + grid_atomic_size + bmm_ready_size: states_ready_size + grid_atomic_size + bmm_ready_size+1],
        nchunks * 32, 32,
        # bmm_wait_ptr, bmm_wait_stride_batch, bmm_wait_stride_chunk,
        sync_atomic[states_ready_size + grid_atomic_size: states_ready_size + grid_atomic_size+1],
        nchunks * 32, 32,
        # grid_atomic, use_atomic_pid
        # sync_atomic, sync_atomic.stride(0), sync_atomic.stride(1), sync_atomic.stride(2), sync_atomic.stride(3),
        sync_atomic[states_ready_size : states_ready_size + 1], use_atomic_pid,
        sync_atomic, nheads * hdim * 32, hdim * 32, 32, 1,

        # Matrix dimensions
        hdim, dstate, chunk_size,
        batch, seqlen, nheads_ngroups_ratio, nheads, nchunks, ngroups,
        ########################################
        # Originally Chunk State
        ########################################
        # Pointers to matrices
        x, B, states_L, dt_out, dA_cumsum, seq_idx,# x_ptr, b_ptr, states_L_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
        # Strides
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), # stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
        B.stride(0), B.stride(1), B.stride(2), B.stride(-1), # stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
        states_L.stride(0), states_L.stride(1), states_L.stride(2), states_L.stride(3), states_L.stride(4), # stride_states_L_batch, stride_states_L_chunk, stride_states_L_head, stride_states_L_hdim, stride_states_L_dstate,
        dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3), # stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3), # stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)), # stride_seq_idx_batch, stride_seq_idx_seqlen,
        
        ########################################
        # Originally State Passing
        ########################################
        # Pointers to matrices
        states_G, final_states, dA_chunk_cumsum, initial_states,
        # Strides
        states_G.stride(0), states_G.stride(1), states_G.stride(2), states_G.stride(3), states_G.stride(4),
        final_states.stride(0), final_states.stride(1), final_states.stride(2), final_states.stride(3),
        dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
        *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2), initial_states.stride(3)) if initial_states is not None else (0, 0, 0, 0)),

        ########################################
        # Originally Chunk Scan
        ########################################
        # Pointers to matrices
        CB, z, out, out_x, C, D,
        # Strides
        CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        D.stride(0) if D is not None else 0,

        ########################################
        # Originally Chunk Cumsum
        ########################################
        # ptrs
        A, dt_bias, dt,
        dt_limit[0], dt_limit[1],
        # strides
        dt.stride(0), dt.stride(1), dt.stride(2),
        A.stride(0),
        dt_bias.stride(0) if dt_bias is not None else 0,


        # Meta-parameters
        IS_CAUSAL=True,
        HAS_D=D is not None,
        D_HAS_HDIM=D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
        HAS_INITSTATES=initial_states is not None,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        DT_SOFTPLUS=dt_softplus,
        HAS_DT_BIAS=dt_bias is not None,
    )

    return out, out_x, states_G, final_states, dA_cumsum, dt_out
