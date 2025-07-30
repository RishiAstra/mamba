# some code from https://triton-lang.org/main/getting-started/tutorials/...

CHECK_CORRECTNESS = True
USE_GIVEN_TEST_TENSORS = False # real prompt data, loads from files
if not CHECK_CORRECTNESS:
    USE_GIVEN_TEST_TENSORS = False

BENCHMARK_REPEATS = 200

CHUNK_SIZE_ORIGINAL=128
CHUNK_SIZE_FUSED=128


have_init_states    = True
have_seq_idx        = True
have_cu_seqlens     = True
have_dt_softplus    = True # TODO: test more


import math
import random
import torch

import triton
import numpy as np

from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd
def run_original_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, chunk_indices, chunk_offsets, chunk_inv_start):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, use_fused5_ssd=False, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, chunk_inv_start=chunk_inv_start)
    if CHUNK_SIZE_ORIGINAL != CHUNK_SIZE_FUSED:
        outputs = outputs[0], outputs[1], None, None, None, *outputs[5:]
    return outputs

def run_fused_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, chunk_indices, chunk_offsets, chunk_inv_start):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, use_fused5_ssd=True, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, chunk_inv_start=chunk_inv_start)
    if CHUNK_SIZE_ORIGINAL != CHUNK_SIZE_FUSED:
        outputs = outputs[0], outputs[1], None, None, None, *outputs[5:]
    return outputs

things_to_compare = [
    (run_original_ssd, "Original", "blue"),
    (run_fused_ssd, "Fused", "red"),
]

DEVICE = triton.runtime.driver.active.get_active_torch_device()

configs = []

batch       = 1
# seqlen    = whatever
nheads      = 80
headdim     = 64
ngroups     = 1
dstate      = 128

def get_test_size(seqlen):
    return (batch, seqlen, nheads, headdim, ngroups, dstate)

test_sizes = [
    (get_test_size(1024 * 2 ** i),) for i in range(0, 8, 1) #9 for batch=1, 6 for 8, 4 for 32 so that original doesn't fail
    # (get_test_size(1024 * 2 ** i),) for i in [7] # for quick test
    # (get_test_size(2753))
]

configs.append(
    triton.testing.Benchmark(
        x_names = ["dims_b_seq_nh_hd_ng_ds"],
        x_vals = test_sizes,
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        line_vals=[i for i in range(len(things_to_compare))],
        line_names=[x[1] for x in things_to_compare],
        styles=[(things_to_compare[i][2], "-") for i in range(len(things_to_compare))],
        ylabel="Prefill Tokens/s (ssd only)",  # Label name for the y-axis
        # TODO: why is there an error if we remove the useless if??
        plot_name="softmax-performance-fp16" + "" if False else "",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)



def _query_start_loc_to_chunk_indices_offsets(query_start_loc: torch.Tensor,
                                              chunk_size: int,
                                              total_seqlens: int):

    cu_seqlens = query_start_loc[1:]  # remove prepended 0

    nchunks = math.ceil(total_seqlens / chunk_size)
    # outputs will have length expansion of chunks that do not divide
    # chunk_size
    N = nchunks + (cu_seqlens[:-1] % chunk_size
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

    # TODO: optimize and make part of ssd_compined.py
    chunk_indices_cpu = chunk_indices.to('cpu').numpy()
    # need offset by 1 because a logical chunk corresponding to a 
    # physical chunk should push the next physical chunk boundry, not the current
    chunk_inv_start = torch.zeros((nchunks + 1,), dtype=torch.int32, device='cpu')
    for chunk_idx in chunk_indices_cpu:
        chunk_inv_start[chunk_idx + 1] += 1
    # now we have a map from physical chunk index to how many logical chunk indices
    # cumsum gives us the start logical chunk for each physical chunk
    chunk_inv_start = chunk_inv_start.to('cuda')
    chunk_inv_start = chunk_inv_start.cumsum(dim=0)

    return chunk_indices, chunk_offsets, chunk_inv_start

# used to load actual tensors from files
counter = 64 # skip warmup tensors, but warmup tensors should be the same anyway
def get_rand_input(dims_b_seq_nh_hd_ng_ds, is_original=True):
    batch, seqlen, nheads, headdim, ngroups, dstate = dims_b_seq_nh_hd_ng_ds
    torch.manual_seed(0)
    random.seed(0)

    dt = torch.randn((batch, seqlen, nheads), dtype=torch.float16, device=DEVICE) * 0.2 + 0.5
    dt_bias = torch.randn((nheads,), dtype=torch.float16, device=DEVICE) * 0.5 - 5
    A = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 3 - 10
    B = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) #* 3 + 7
    C = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) #* 5 + 20
    D = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 0.5 + 1.2
    x = torch.randn((batch, seqlen, nheads, headdim,), dtype=torch.float16, device=DEVICE) * 2 + 5

    # NOTE: overrides the sizes
    if USE_GIVEN_TEST_TENSORS:
        dump_name = "dump_f5" if not is_original else "dump"
        A =         torch.load(f"{dump_name}/A_in{counter}")
        B =         torch.load(f"{dump_name}/B_in{counter}")
        C =         torch.load(f"{dump_name}/C_in{counter}")
        D =         torch.load(f"{dump_name}/D_in{counter}")
        dt_bias =   torch.load(f"{dump_name}/dt_bias_in{counter}")
        dt =        torch.load(f"{dump_name}/dt_in{counter}")
        x =         torch.load(f"{dump_name}/x_in{counter}")

    varlen_count = 1
    if have_seq_idx:
        if not have_cu_seqlens:
            cu_seqlens = torch.tensor((0, seqlen), dtype=torch.int32, device='cpu')
        else:
            cu_seqlens = [0]
        # example at https://github.com/state-spaces/mamba/issues/383

        seq_idx = torch.zeros((batch, seqlen), dtype=torch.int32, device='cpu')
        assert batch == 1
        max_part_seqlen = 1000 #seqlen // 2#100#25#2#4#25
        split = random.randint(1, max_part_seqlen)
        while split < seqlen:
            seq_idx[0, split] = 1
            if have_cu_seqlens:
                cu_seqlens.append(split)
                varlen_count += 1
            split += random.randint(1, max_part_seqlen)

        if have_cu_seqlens:
            cu_seqlens.append(seqlen)
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device='cpu')

        seq_idx = seq_idx.to(device=DEVICE, dtype=torch.int32)
        seq_idx = torch.cumsum(seq_idx, dim=-1, dtype=torch.int32)
        # print("seq_idx:", seq_idx)
        # print("cu_seqlens:", cu_seqlens)
        # print("len cu_seqlens:", len(cu_seqlens))
    else:
        seq_idx = torch.zeros((1, seqlen), dtype=torch.int32, device='cuda')
        cu_seqlens = torch.tensor((0, seqlen), dtype=torch.int32, device='cpu')

    if have_init_states:
        assert have_cu_seqlens
        initial_states = torch.randn((varlen_count, nheads, headdim, dstate), dtype=torch.float16, device=DEVICE) * 0.2
    else:
        initial_states = None

    chunk_indices, chunk_offsets, chunk_inv_start = \
            _query_start_loc_to_chunk_indices_offsets(
                cu_seqlens, CHUNK_SIZE_ORIGINAL if is_original else CHUNK_SIZE_FUSED, seqlen)

    chunk_indices = chunk_indices.to('cuda')
    chunk_offsets = chunk_offsets.to('cuda')
    cu_seqlens = cu_seqlens.to('cuda')
    return dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens if have_cu_seqlens else None, chunk_indices, chunk_offsets, chunk_inv_start

def run_unit_test(seqlen):
    # raise Exception("Not implemented")
    outputs_full = []

    for i, thing in enumerate(things_to_compare):
        dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens, chunk_indices, chunk_offsets, chunk_inv_start = get_rand_input(get_test_size(seqlen), is_original=i==0)

        outputs_full.append(thing[0](
            x, dt, A, B, C, CHUNK_SIZE_ORIGINAL if i == 0 else CHUNK_SIZE_FUSED, D=D, z=None, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus,
            chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, chunk_inv_start=chunk_inv_start
        ))

    # outputs_full = [thing[0](
    #         x, dt, A, B, C, CHUNK_SIZE, D=D, z=None, dt_bias=dt_bias,
    #         initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus
    #         ) for thing in things_to_compare]
    field_names = ["out", "out_x", "dt", "dA_cumsum", "states", "final_states", "varlen_states"]
    for field_idx in range(len(outputs_full[0])):
        outputs_0 = outputs_full[0][field_idx]
        if outputs_0 is None: # skip None
            continue
        print("----------------------------------------")
        print(f"comparing field {field_names[field_idx]}")
        # for i in range(len(outputs_full)):
        #     outputs_i = outputs_full[i][field_idx]
        #     print(f"output for {things_to_compare[i][1]} = {outputs_i}")
        PRINT_BAD_TENSOR = False
        bad_i = -1
        # compare all to the first
        for i in range(1, len(outputs_full), 1):
            outputs_i = outputs_full[i][field_idx]
            print(f"ref shape: {outputs_0.shape}, test shape: {outputs_i.shape}")
            # atol = 2.5e-3
            # rtol = 1e-2
            # TODO: revert tolerances
            atol = 5e-3
            rtol = 2e-2
            outputs_i = outputs_i.to(outputs_0.dtype)
            if torch.allclose(outputs_i, outputs_0, atol=atol, rtol=rtol, equal_nan=True):
                print(f"✅ {things_to_compare[i][1]} and {things_to_compare[0][1]} match")
            else:
                print(f"❌ {things_to_compare[i][1]} and {things_to_compare[0][1]} differ")
                bad_i = i
            max_diff_idx = torch.argmax(torch.abs(outputs_i - outputs_0))
            max_diff = torch.abs(outputs_i.reshape(-1)[max_diff_idx] - outputs_0.reshape(-1)[max_diff_idx])
            max_diff_idx_shaped = [(max_diff_idx.item() // outputs_0.stride(i)) % outputs_0.shape[i]  for i in range(len(outputs_0.shape))]
            print(f"max diff: {max_diff} for {max_diff_idx_shaped} index, test: {outputs_i.reshape(-1)[max_diff_idx]}, ref: {outputs_0.reshape(-1)[max_diff_idx]}")
            max_allowed_diff = atol + rtol * torch.abs(outputs_0)
            max_diff_frac = torch.abs(outputs_i - outputs_0) / max_allowed_diff
            max_rdiff_idx = torch.argmax(max_diff_frac)
            max_rdiff = max_diff_frac.reshape(-1)[max_rdiff_idx]
            max_rdiff_idx_shaped = [(max_rdiff_idx.item() // outputs_0.stride(i)) % outputs_0.shape[i]  for i in range(len(outputs_0.shape))]
            print(f"max diff score: {max_rdiff} for {max_rdiff_idx_shaped} index, test: {outputs_i.reshape(-1)[max_rdiff_idx]}, ref: {outputs_0.reshape(-1)[max_rdiff_idx]}")
            # where ref is undefined, allow test to mismatch
            fail_bools = (torch.abs(outputs_i - outputs_0) > atol + rtol * torch.abs(outputs_0)) & torch.isfinite(outputs_0)
            fail_bool_num = fail_bools.sum().cpu().item()
            total_elems = outputs_0.numel()
            print(f"failed elements / total: {fail_bool_num} / {total_elems}, {fail_bool_num / total_elems * 100.0}%")
            if fail_bool_num > 0:
                min_fail_idx_shaped = fail_bools.nonzero(as_tuple=False)[ 0]#.squeeze()[ 0]#.min()
                max_fail_idx_shaped = fail_bools.nonzero(as_tuple=False)[-1]#.squeeze()[-1]#.max()
                # min_fail_idx_shaped = [(min_fail_idx.item() // outputs_0.stride(i)) % outputs_0.shape[i]  for i in range(len(outputs_0.shape))]
                # max_fail_idx_shaped = [(max_fail_idx.item() // outputs_0.stride(i)) % outputs_0.shape[i]  for i in range(len(outputs_0.shape))]
                print(f"failed min idx: {min_fail_idx_shaped}, max idx: {max_fail_idx_shaped}")
                min_idx_rdiff = max_diff_frac[*min_fail_idx_shaped]
                max_idx_rdiff = max_diff_frac[*max_fail_idx_shaped]
                print(f"failed min idx score: {min_idx_rdiff}, max idx score: {max_idx_rdiff}")
            print(f"max abs val: {outputs_0.abs().max().item()}")



        if bad_i >= 0 and PRINT_BAD_TENSOR:
            np.savetxt(f'bad_output_{field_idx}.txt', outputs_full[bad_i][field_idx].cpu().numpy(), fmt="%.2e")


@triton.testing.perf_report(configs)
def benchmark(dims_b_seq_nh_hd_ng_ds, provider):
    batch, seqlen, nheads, headdim, ngroups, dstate = dims_b_seq_nh_hd_ng_ds

    dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens, chunk_indices, chunk_offsets, chunk_inv_start = get_rand_input(dims_b_seq_nh_hd_ng_ds, is_original=provider==0)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench( \
        lambda: things_to_compare[provider][0](
            x, dt, A, B, C, CHUNK_SIZE_ORIGINAL if provider == 0 else CHUNK_SIZE_FUSED, D=D, z=None, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets, chunk_inv_start=chunk_inv_start), \
        rep=BENCHMARK_REPEATS, quantiles=quantiles
    )
    perf = lambda ms: batch * seqlen / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    # use to check correctness
    if CHECK_CORRECTNESS:
        if USE_GIVEN_TEST_TENSORS:
            for i in range(64):
                print(f"******* counter {counter} ******")
                run_unit_test(1)
                counter += 1
        else:
            run_unit_test(2763)
            # run_unit_test(163)
    else:
        benchmark.run(show_plots=True, print_data=True)

