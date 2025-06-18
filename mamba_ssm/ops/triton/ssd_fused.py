# The fused ssd kernel

from einops import rearrange
import torch
import triton
import triton.language as tl

from packaging import version

from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_fwd
from mamba_ssm.ops.triton.ssd_state_passing import _state_passing_fwd
TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


# A100 Autotuning Results (useful for making fused_x3 autotune configs):

# Triton autotuning for function _chunk_state_fwd_kernel finished after 1.26s; best config selected: 
# BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 128, BLOCK_SIZE_K: 32, num_warps: 4, num_ctas: 1, num_stages: 4, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
# Triton autotuning for function _state_passing_fwd_kernel finished after 0.78s; best config selected: 
# BLOCK_SIZE: 512, num_warps: 4, num_ctas: 1, num_stages: 3, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
# Triton autotuning for function _chunk_scan_fwd_kernel finished after 1.83s; best config selected: 
# BLOCK_SIZE_M: 32, BLOCK_SIZE_N: 64, BLOCK_SIZE_K: 32, num_warps: 2, num_ctas: 1, num_stages: 5, num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
# for fused x2, manually found triton.Config({'BLOCK_SIZE_M': 64//2,  'BLOCK_SIZE_N': 128//2, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_SP': 1024, 'SKIP_THREADBLOCK_COUNT': 128}, num_stages=1, num_warps=8),

# Combined:
# NOTE: block sizes might mean different things!
# ~3% drop in ssd throughput if we lock state passing at 2048 block size
# ~4% drop in ssd throughput if we lock chunk state at 32 x 64 block size MxN

# for chunk state:
# M goes along x's hdim
# K goes along x and b's seqlen
# N goes along b's dstate

# for state passing:
# block is 2d for flattened "dim" = hdim * dstate

# for chunk scan:
# acc (MxN) = C @ prev_states
# M goes along C's chunk_size_limit
# K goes along C and state's dstate
# N goes along state's hdim

# This means that overall we have:
# BLOCK_SIZE_H along hdim
# BLOCK_SIZE_D along dstate
# BLOCK_SIZE_S along seqlen (seqlen is within a chunk)



# pids

# chunk state
# pid_c = pid_bc // batch
# pid_b = pid_bc - pid_c * batch
# pid_h = tl.program_id(axis=2)
# num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
# pid_m = tl.program_id(axis=0) // num_pid_n
# pid_n = tl.program_id(axis=0) % num_pid_n
# loop:
# for k in range(0, chunk_size_limit, BLOCK_SIZE_K):

# state passing
# b, h match, m can be split into m, n, for c can be pid
# can fully match
# pid_b = tl.program_id(axis=1)
# pid_h = tl.program_id(axis=2)
# pid_m = tl.program_id(axis=0)
# loop:
# for c in range(nchunks):

# chunk scan
# pids match already
# pid_c = pid_bc // batch
# pid_b = pid_bc - pid_c * batch
# pid_h = tl.program_id(axis=2)
# num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
# pid_m = tl.program_id(axis=0) // num_pid_n
# pid_n = tl.program_id(axis=0) % num_pid_n
# loop:
# for k in range(0, dstate, BLOCK_SIZE_K):
# for k in range(0, chunk_size_limit, BLOCK_SIZE_K):


# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
#     ],
#     key=['hdim', 'dstate', 'chunk_size'],
# )
# @triton.jit
# def _chunk_state_fwd_kernel(

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE': 64}),
#         triton.Config({'BLOCK_SIZE': 128}),
#         triton.Config({'BLOCK_SIZE': 256}),
#         triton.Config({'BLOCK_SIZE': 512}),
#         triton.Config({'BLOCK_SIZE': 1024}),
#         triton.Config({'BLOCK_SIZE': 2048}),
#     ],
#     key=['dim'],
# )
# @triton.jit
# def _state_passing_fwd_kernel(

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
#     ],
#     key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
# )
# @triton.jit
# def _chunk_scan_fwd_kernel(

# TODO: probably should have chunks as outermost pid so that even batch can be done before sync (less total sync)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_DS': 64, 'BLOCK_SIZE_CS': 32}, num_stages=5, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size', 'IS_CAUSAL'],
)
@triton.jit
def _fused3_ssd_kernel(
    sync_atomic, stride_sync_batch, stride_sync_head, stride_sync_hdim, stride_sync_dstate,

    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio, nheads, nchunks,
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

    # # Pointers to matrices
    # # Strides
    # states_G.stride(0), states_G.stride(1), states_G.stride(2), states_G.stride(3), states_G.stride(4),
    # final_states.stride(0), final_states.stride(1), final_states.stride(2), final_states.stride(3),
    # dA_chunk_cumsum.stride(0), dA_chunk_cumsum.stride(2), dA_chunk_cumsum.stride(1),
    # *((initial_states.stride(0), initial_states.stride(1), initial_states.stride(2), initial_states.stride(3)) if initial_states is not None else (0, 0, 0, 0)),



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



    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_HD: tl.constexpr, BLOCK_SIZE_DS: tl.constexpr, BLOCK_SIZE_CS: tl.constexpr,
):
    # 5 pids, same for all 3 parts
    # all pids represent domain parallelism except pid_c for state passing
    pid = tl.program_id(0)
    num_pid_ds = tl.cdiv(dstate, BLOCK_SIZE_DS)
    num_pid_hd = tl.cdiv(hdim, BLOCK_SIZE_HD)
    pid_ds = pid % num_pid_ds
    pid_hd = (pid // num_pid_ds) % num_pid_hd
    pid_h = (pid // (num_pid_ds * num_pid_hd)) % nheads
    pid_c = (pid // (num_pid_ds * num_pid_hd * nheads)) % nchunks
    pid_b = (pid // (num_pid_ds * num_pid_hd * nheads * nchunks)) % batch


    ########################################
    # Chunk State
    ########################################

    # chunk state ptrs
    b_ptr_cs = b_ptr + pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr_cs = x_ptr + pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr_cs = dt_ptr + pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr_cs = dA_cumsum_ptr + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr_cs = seq_idx_ptr + pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    # chunk state offsets
    # NOTE: m ->hdim, n -> dstate, k -> chunk_size
    offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
    offs_ds = pid_ds * BLOCK_SIZE_DS + tl.arange(0, BLOCK_SIZE_DS)
    offs_cs = tl.arange(0, BLOCK_SIZE_CS)

    # chunk state ptr blocks
    x_ptrs_cs = x_ptr_cs + (offs_hd[:, None] * stride_x_hdim + offs_cs[None, :] * stride_x_seqlen)
    b_ptrs_cs = b_ptr_cs + (offs_ds[None, :] * stride_b_dstate + offs_cs[:, None] * stride_b_seqlen)
    dt_ptrs_cs = dt_ptr_cs + offs_cs * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr_cs + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs_cs = dA_cumsum_ptr_cs + offs_cs * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs_cs = seq_idx_ptr_cs + offs_cs * stride_seq_idx_seqlen

    # chunk state other setup
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr_cs + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    # chunk state chunk_size loop
    acc = tl.zeros((BLOCK_SIZE_HD, BLOCK_SIZE_DS), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_CS):
        x = tl.load(x_ptrs_cs, mask=(offs_hd[:, None] < hdim) & (offs_cs[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs_cs, mask=(offs_cs[:, None] < chunk_size_limit - k) & (offs_ds[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs_cs, mask=offs_cs < chunk_size_limit - k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs_cs, mask=offs_cs < chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs_cs, mask=offs_cs < chunk_size_limit - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
        else:
            scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr_cs.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs_cs += BLOCK_SIZE_CS * stride_x_seqlen
        b_ptrs_cs += BLOCK_SIZE_CS * stride_b_seqlen
        dt_ptrs_cs += BLOCK_SIZE_CS * stride_dt_csize
        dA_cumsum_ptrs_cs += BLOCK_SIZE_CS * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs_cs += BLOCK_SIZE_CS * stride_seq_idx_seqlen
    states = acc.to(states_L_ptr.dtype.element_ty)

    # chunk state final store
    states_L_ptr_cs = states_L_ptr + pid_b * stride_states_L_batch + pid_c * stride_states_L_chunk + pid_h * stride_states_L_head
    # TODO: seems like a duplicate `offs` computation, but maybe it reduces register pressure
    offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
    offs_ds = pid_ds * BLOCK_SIZE_DS + tl.arange(0, BLOCK_SIZE_DS)
    states_ptrs_cs = states_L_ptr_cs + (offs_hd[:, None] * stride_states_L_hdim + offs_ds[None, :] * stride_states_L_dstate)
    c_mask = (offs_hd[:, None] < hdim) & (offs_ds[None, :] < dstate)
    tl.store(states_ptrs_cs, states, mask=c_mask)



    ########################################
    # State Passing
    ########################################

    sync_atomic += pid_b * stride_sync_batch + pid_h * stride_sync_head + pid_hd * stride_sync_hdim + pid_ds * stride_sync_dstate

    states_L_ptr += pid_b * stride_states_L_batch + pid_h * stride_states_L_head
    dA_ccs_ptr += pid_b * stride_dA_ccs_batch + pid_h * stride_dA_ccs_head
    states_G_ptr += pid_b * stride_states_G_batch + pid_h * stride_states_G_head
    final_states_ptr += pid_b * stride_final_states_batch + pid_h * stride_final_states_head
    if HAS_INITSTATES:
        initstates_ptr += pid_b * stride_initstates_batch + pid_h * stride_initstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch

    offs_hd = pid_hd * BLOCK_SIZE_HD + tl.arange(0, BLOCK_SIZE_HD)
    offs_ds = pid_ds * BLOCK_SIZE_DS + tl.arange(0, BLOCK_SIZE_DS)
    states_ptrs = states_L_ptr + offs_hd[:, None] * stride_states_L_hdim + offs_ds[None, :] * stride_states_L_dstate
    state_G_ptrs = states_G_ptr + offs_hd[:, None] * stride_states_G_hdim + offs_ds[None, :] * stride_states_G_dstate
    final_states_ptrs = final_states_ptr + offs_hd[:, None] * stride_final_states_hdim + offs_ds[None, :] * stride_final_states_dstate

    main_mask = (offs_hd < hdim)[:, None] & (offs_ds < dstate)[None, :]
    
    # Instead of looping over chunks, we have pid_c
    # first sync (wait for previous), then all must load

    # sync
    # the atomic represents which pid_c is ready
    # therefore, wait for it to reach our pid_c
    sync_val = tl.atomic_add(sync_atomic, 0)
    while sync_val < pid_c:
        sync_val = tl.atomic_add(sync_atomic, 0)

    # special case 0
    if pid_c == 0:
        if not HAS_INITSTATES:
            states_prev = tl.zeros((BLOCK_SIZE_HD, BLOCK_SIZE_DS), dtype=tl.float32)
        else:
            initstates_ptrs = initstates_ptr + offs_hd[:, None] * stride_initstates_hdim + offs_ds[None, :] * stride_initstates_dstate
            states_prev = tl.load(initstates_ptrs, mask=main_mask, other=0.0).to(tl.float32)
        tl.store(state_G_ptrs, states_prev, mask=main_mask)
    else:
        # need to load states from previous one (but since already offset by 1, just pid_c)
        states_prev = tl.load(state_G_ptrs + stride_states_G_chunk * pid_c, mask=main_mask, other=0.0).to(tl.float32)

    # ptrs
    seq_idx = 0
    states_ptrs += stride_states_L_chunk * pid_c
    dA_ccs_ptr += stride_dA_ccs_chunk * pid_c
    state_G_ptrs += stride_states_G_chunk * (pid_c + 1) # offset since 0 gets initial states

    # TODO: load states from last one, must read most recent, so might need to be atomic
    new_states = tl.load(states_ptrs, mask=main_mask, other=0.0).to(tl.float32)

    dA_cs = tl.load(dA_ccs_ptr).to(tl.float32)
    scale = tl.exp(dA_cs)
    if HAS_SEQ_IDX:
        seq_idx_new = tl.load(seq_idx_ptr + (min((c + 1) * chunk_size, seqlen) - 1) * stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_new == seq_idx, scale, 0.0)
        seq_idx = seq_idx_new
    states_mod = scale * states_prev + new_states
    if pid_c < nchunks - 1:
        tl.store(state_G_ptrs, states_mod, mask=main_mask)
    else:
        tl.store(final_states_ptrs, states_mod, mask=main_mask)

    # let the next one go
    tl.atomic_add(sync_atomic, 1)



    ########################################
    # Chunk Scan
    ########################################


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_HD': 32, 'BLOCK_SIZE_DS': 64, 'BLOCK_SIZE_CS': 32, 'CS_BLOCK_SIZE_HD': 64, 'CS_BLOCK_SIZE_DS': 32}, num_stages=5, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size', 'IS_CAUSAL'],
)
@triton.jit
def _chunk_scan_fwd_kernel_new(
    # Pointers to matrices
    cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, prev_states_ptr, D_ptr,
    # Matrix dimensions
    nheads, nchunks, chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_CS: tl.constexpr, BLOCK_SIZE_HD: tl.constexpr, BLOCK_SIZE_DS: tl.constexpr,
    CS_BLOCK_SIZE_HD: tl.constexpr, CS_BLOCK_SIZE_DS: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    CS_BLOCK_HD_MULT = CS_BLOCK_SIZE_HD // BLOCK_SIZE_HD
    # 5 pids, same for all 3 parts
    # all pids represent domain parallelism except pid_c for state passing
    pid = tl.program_id(0)
    num_pid_ds = tl.cdiv(dstate, BLOCK_SIZE_DS) # TODO: see below
    num_pid_cs = tl.cdiv(chunk_size, BLOCK_SIZE_CS)
    num_pid_hd = tl.cdiv(hdim, BLOCK_SIZE_HD) # TODO: fix, need separate block size somehow, maybe a mult, maybe in autotune
    pid_ds = pid % num_pid_ds
    pid_hd = (pid // num_pid_ds) % num_pid_hd
    pid_h = (pid // (num_pid_ds * num_pid_hd)) % nheads
    pid_c = (pid // (num_pid_ds * num_pid_hd * nheads)) % nchunks
    pid_b = (pid // (num_pid_ds * num_pid_hd * nheads * nchunks)) % batch

    # only 1 pid_ds, half the pid_hd
    # to handle this, we put the extra along the pid_cs which doesn't exist
    # this way, we use the exact same grid without any early returns
    # in return for possbily pad tail SM utilization for effective pid_cs > num_pid_cs
    pid_other = (pid_hd % CS_BLOCK_HD_MULT) * num_pid_ds + pid_ds
    pid_hd = pid_hd // CS_BLOCK_HD_MULT

    cb_ptr_og = cb_ptr
    x_ptr_og = x_ptr
    dt_ptr_og = dt_ptr
    dA_cumsum_ptr_og = dA_cumsum_ptr
    C_ptr_og = C_ptr
    prev_states_ptr_og = prev_states_ptr
    seq_idx_ptr_og = seq_idx_ptr
    out_x_ptr_og = out_x_ptr
    z_ptr_og = z_ptr
    out_ptr_og = out_ptr

    for pid_cs in range(pid_other, num_pid_cs, num_pid_ds*CS_BLOCK_HD_MULT):
        cb_ptr = cb_ptr_og + pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
        x_ptr = x_ptr_og + pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
        dt_ptr = dt_ptr_og + pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
        dA_cumsum_ptr = dA_cumsum_ptr_og + pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
        C_ptr = C_ptr_og + pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
        prev_states_ptr = prev_states_ptr_og + pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
        if HAS_SEQ_IDX:
            seq_idx_ptr = seq_idx_ptr_og + pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

        offs_cs = pid_cs * BLOCK_SIZE_CS + tl.arange(0, BLOCK_SIZE_CS)
        offs_hd = pid_hd * CS_BLOCK_SIZE_HD + tl.arange(0, CS_BLOCK_SIZE_HD)
        dA_cs_m = tl.load(dA_cumsum_ptr + offs_cs * stride_dA_cs_csize, mask=offs_cs < chunk_size, other=0.0).to(tl.float32)

        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        if HAS_SEQ_IDX:
            seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
            seq_idx_m = tl.load(seq_idx_ptr + offs_cs * stride_seq_idx_seqlen, mask=offs_cs < chunk_size_limit, other=-1)
        acc = tl.zeros((BLOCK_SIZE_CS, CS_BLOCK_SIZE_HD), dtype=tl.float32)

        # Without the if (pid_c > -1), with Triton 2.1.0, I get
        # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
        # With Triton 2.2.0, this works
        if IS_TRITON_22 or pid_c > -1:
            # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
            offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else CS_BLOCK_SIZE_DS)
            C_ptrs = C_ptr + (offs_cs[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
            prev_states_ptrs = prev_states_ptr + (offs_hd[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
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
        K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_cs + 1) * BLOCK_SIZE_CS, chunk_size_limit)
        for k in range(0, K_MAX, CS_BLOCK_SIZE_DS):
            cb = tl.load(cb_ptrs, mask=(offs_cs[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
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
            x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_hd[None, :] < hdim), other=0.0)
            acc += tl.dot(cb, x)
            cb_ptrs += CS_BLOCK_SIZE_DS * stride_cb_csize_k
            x_ptrs += CS_BLOCK_SIZE_DS * stride_x_seqlen
            dt_ptrs += CS_BLOCK_SIZE_DS * stride_dt_csize
            dA_cumsum_ptrs += CS_BLOCK_SIZE_DS * stride_dA_cs_csize

        offs_out_m = pid_cs * BLOCK_SIZE_CS + tl.arange(0, BLOCK_SIZE_CS)
        offs_out_n = pid_hd * CS_BLOCK_SIZE_HD + tl.arange(0, CS_BLOCK_SIZE_HD)

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

        out_ptr = out_ptr_og + pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
        tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))


def _fused3_ssd(
    C, B, CB, x, dt, dA_cumsum, out_dtype, D,
    initial_states=None, seq_idx=None, states_in_fp32=True, z=None
):
    # setup from chunk state
    batch, seqlen, nheads, hdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # TODO: check why chunk state had a `states` parameter
    states_dtype = torch.float32 if states_in_fp32 else B.dtype
    states_L = torch.empty((batch, nchunks, nheads, hdim, dstate), device=x.device, dtype=states_dtype)

    # setup from state passing
    dA_chunk_cumsum = dA_cumsum[:, :, :, -1]
    assert dA_chunk_cumsum.shape == (batch, nheads, nchunks)
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, hdim, dstate)
    if seq_idx is not None:
        assert chunk_size is not None
        seqlen = seq_idx.shape[-1]
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

    grid = lambda META: (batch * nchunks * nheads * triton.cdiv(hdim, META['BLOCK_SIZE_HD']) * triton.cdiv(dstate, META['BLOCK_SIZE_DS']),)

    sync_atomic = torch.zeros((batch, nheads, hdim, dstate), dtype=torch.int32, device=x.device)

    nheads_ngroups_ratio = nheads // ngroups
    BAD_STRIDE=1000000000
    BAD_PTR_FP16 = torch.zeros((1,), dtype=torch.float16, device=x.device)
    BAD_PTR_FP32 = torch.zeros((1,), dtype=torch.float32, device=x.device)
    _fused3_ssd_kernel[grid](
        sync_atomic, sync_atomic.stride(0), sync_atomic.stride(1), sync_atomic.stride(2), sync_atomic.stride(3),

        # Matrix dimensions
        hdim, dstate, chunk_size,
        batch, seqlen, nheads_ngroups_ratio, nheads, nchunks,
        ########################################
        # Originally Chunk State
        ########################################
        # Pointers to matrices
        x, B, states_L, dt, dA_cumsum, seq_idx,# x_ptr, b_ptr, states_L_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
        # Strides
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), # stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
        B.stride(0), B.stride(1), B.stride(2), B.stride(-1), # stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
        states_L.stride(0), states_L.stride(1), states_L.stride(2), states_L.stride(3), states_L.stride(4), # stride_states_L_batch, stride_states_L_chunk, stride_states_L_head, stride_states_L_hdim, stride_states_L_dstate,
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3), # stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
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
        BAD_PTR_FP32, BAD_PTR_FP16, BAD_PTR_FP16, BAD_PTR_FP16, BAD_PTR_FP16, BAD_PTR_FP32, # cb_ptr, z_ptr, out_ptr, out_x_ptr, C_ptr, D_ptr,
        # Strides
        BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, # stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
        BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, # stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
        BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, # stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
        BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, BAD_STRIDE, # stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
        BAD_STRIDE, # stride_D_head,



        # Meta-parameters
        IS_CAUSAL=True,
        HAS_D=True,
        D_HAS_HDIM=False,
        HAS_Z=False,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
        HAS_INITSTATES=initial_states is not None,
    )

    _chunk_scan_fwd_kernel_new[grid](
        CB, x, z, out, out_x, dt, dA_cumsum, seq_idx, C, states_G, D,
        nheads, nchunks, chunk_size, hdim, dstate,
        batch, seqlen, nheads // ngroups,
        CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        states_G.stride(0), states_G.stride(1), states_G.stride(2), states_G.stride(3), states_G.stride(4),
        D.stride(0) if D is not None else 0,
        True,
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
    )

    # return torch.zeros_like(x), None, None, None
    return out, out_x, states_G, final_states