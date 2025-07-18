# some code from https://triton-lang.org/main/getting-started/tutorials/...

CHECK_CORRECTNESS = True
USE_GIVEN_TEST_TENSORS = False # real prompt data, loads from files
if not CHECK_CORRECTNESS:
    USE_GIVEN_TEST_TENSORS = False

BENCHMARK_REPEATS = 100

have_init_states    = False
have_seq_idx        = False
have_cu_seqlens     = False # TODO: test
have_dt_softplus    = True # TODO: test more


import random
import torch

import triton
import numpy as np

from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd
def run_original_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, use_fused5_ssd=False)
    return outputs

def run_fused_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, use_fused5_ssd=True)
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
    (get_test_size(1024 * 2 ** i),) for i in range(0, 9, 1) #9 for batch=1, 6 for 8, 4 for 32 so that original doesn't fail
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

# used to load actual tensors from files
counter = 64 # skip warmup tensors, but warmup tensors should be the same anyway
def get_rand_input(dims_b_seq_nh_hd_ng_ds, is_original=True):
    batch, seqlen, nheads, headdim, ngroups, dstate = dims_b_seq_nh_hd_ng_ds
    torch.manual_seed(0)

    dt = torch.randn((batch, seqlen, nheads), dtype=torch.float16, device=DEVICE) * 0.2 + 0.5
    dt_bias = torch.randn((nheads,), dtype=torch.float16, device=DEVICE) * 0.5 - 5
    A = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 3 - 10
    B = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 3 + 7
    C = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 5 + 20
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

    if have_init_states:
        initial_states = torch.randn((batch, nheads, headdim, dstate), dtype=torch.float32, device=DEVICE) * 0.2
    else:
        initial_states = None

    if have_seq_idx:
        # example at https://github.com/state-spaces/mamba/issues/383

        seq_idx = torch.zeros((batch, seqlen), dtype=torch.int32, device='cpu')
        # have at least 1 batch not have multiple sequences if b > 1
        no_seq_idx_b = -1
        if batch > 1:
            no_seq_idx_b = random.randint(0, batch)
        for b in range(batch):
            split = random.randint(0, seqlen)
            while split < seqlen and b != no_seq_idx_b:
                seq_idx[b, split] = 1
                split += random.randint(0, seqlen)

        seq_idx = seq_idx.to(device=DEVICE)
        seq_idx = torch.cumsum(seq_idx, dim=-1).to(torch.int32)

    else:
        seq_idx = None

    return dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, None #, cu_seqlens


def run_unit_test(seqlen):
    # raise Exception("Not implemented")
    
    CHUNK_SIZE=256

    outputs_full = []

    for i, thing in enumerate(things_to_compare):
        dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens = get_rand_input(get_test_size(seqlen), is_original=i==0)

        outputs_full.append(thing[0](
            x, dt, A, B, C, CHUNK_SIZE, D=D, z=None, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus
        ))

    # outputs_full = [thing[0](
    #         x, dt, A, B, C, CHUNK_SIZE, D=D, z=None, dt_bias=dt_bias,
    #         initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus
    #         ) for thing in things_to_compare]
    field_names = ["out", "out_x", "dt", "dA_cumsum", "states", "final_states"]
    for field_idx in range(len(outputs_full[0])):
        outputs_0 = outputs_full[0][field_idx]
        if outputs_0 is None: # skip None
            continue
        print(f"comparing field {field_names[field_idx]}")
        # for i in range(len(outputs_full)):
        #     outputs_i = outputs_full[i][field_idx]
        #     print(f"output for {things_to_compare[i][1]} = {outputs_i}")
        PRINT_BAD_TENSOR = False
        bad_i = -1
        # compare all to the first
        for i in range(1, len(outputs_full), 1):
            outputs_i = outputs_full[i][field_idx]
            atol = 2.5e-3
            rtol = 1e-2
            outputs_i = outputs_i.to(outputs_0.dtype)
            if torch.allclose(outputs_i, outputs_0, atol=atol, rtol=rtol, equal_nan=True):
                print(f"✅ {things_to_compare[i][1]} and {things_to_compare[0][1]} match")
            else:
                print(f"❌ {things_to_compare[i][1]} and {things_to_compare[0][1]} differ")
                bad_i = i
            max_diff_idx = torch.argmax(torch.abs(outputs_i - outputs_0))
            max_diff = torch.abs(outputs_i.reshape(-1)[max_diff_idx] - outputs_0.reshape(-1)[max_diff_idx])
            print(f"max diff: {max_diff} for {max_diff_idx} index, a: {outputs_i.reshape(-1)[max_diff_idx]}, b: {outputs_0.reshape(-1)[max_diff_idx]}")
            max_allowed_diff = atol + rtol * torch.abs(outputs_0)
            max_diff_frac = torch.abs(outputs_i - outputs_0) / max_allowed_diff
            max_rdiff_idx = torch.argmax(max_diff_frac)
            max_rdiff = max_diff_frac.reshape(-1)[max_rdiff_idx]
            print(f"max diff score: {max_rdiff} for {max_rdiff_idx} index, a: {outputs_i.reshape(-1)[max_rdiff_idx]}, b: {outputs_0.reshape(-1)[max_rdiff_idx]}")
            fail_bools = torch.abs(outputs_i - outputs_0) > atol + rtol * torch.abs(outputs_0)
            fail_bool_num = fail_bools.sum().cpu().item()
            total_elems = outputs_0.numel()
            print(f"failed elements / total: {fail_bool_num} / {total_elems}, {fail_bool_num / total_elems * 100.0}%")


        if bad_i >= 0 and PRINT_BAD_TENSOR:
            np.savetxt(f'bad_output_{field_idx}.txt', outputs_full[bad_i][field_idx].cpu().numpy(), fmt="%.2e")


@triton.testing.perf_report(configs)
def benchmark(dims_b_seq_nh_hd_ng_ds, provider):
    batch, seqlen, nheads, headdim, ngroups, dstate = dims_b_seq_nh_hd_ng_ds

    dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens = get_rand_input(dims_b_seq_nh_hd_ng_ds)
    CHUNK_SIZE=256

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench( \
        lambda: things_to_compare[provider][0](
            x, dt, A, B, C, CHUNK_SIZE, D=D, z=None, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus), \
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
    else:
        benchmark.run(show_plots=True, print_data=True)

