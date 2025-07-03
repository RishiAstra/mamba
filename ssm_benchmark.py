# some code from https://triton-lang.org/main/getting-started/tutorials/...

# TODO: check correctness for out, out_x, dt, dA_cumsum, states, final_states, not just out
BENCHMARK_REPEATS = 25



have_init_states    = False
have_seq_idx        = False
have_cu_seqlens     = False # TODO: test
have_dt_softplus    = False # TODO: test

import random
import torch

import triton
import triton.language as tl
import numpy as np

from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd
def run_original_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, fused=False)
    return outputs[0]
    # return out, out_x, dt, dA_cumsum, states, final_states

def run_fused_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, fused=True)
    return outputs[0]
    # return out, out_x, dt, dA_cumsum, states, final_states

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
    (get_test_size(1024 * 2 ** i),) for i in range(0, 9, 1) #9 for batch=1, 7 for 8, 6 for 32?
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

def get_rand_input(dims_b_seq_nh_hd_ng_ds):
    batch, seqlen, nheads, headdim, ngroups, dstate = dims_b_seq_nh_hd_ng_ds
    torch.manual_seed(0)

    dt = torch.randn((batch, seqlen, nheads), dtype=torch.float16, device=DEVICE) * 0.005 + 0.025
    dt_bias = torch.randn((nheads,), dtype=torch.float16, device=DEVICE) * 0.01
    A = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 0.005 + 0.02
    B = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 0.1 + 0.3
    C = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 0.1 + 0.3
    D = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 0.01 + 0.3
    x = torch.randn((batch, seqlen, nheads, headdim,), dtype=torch.float16, device=DEVICE) * 0.4

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
    dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens = get_rand_input(get_test_size(seqlen))
    CHUNK_SIZE=256

    outputs = [thing[0](
            x, dt, A, B, C, CHUNK_SIZE, D=D, z=None, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus
            ) for thing in things_to_compare]
    for i in range(len(outputs)):
        print(f"output for {things_to_compare[i][1]} = {outputs[i]}")
    PRINT_BAD_TENSOR = False
    bad_i = -1
    # compare all to the first
    for i in range(1, len(outputs), 1):
        if torch.allclose(outputs[i], outputs[0], atol=5e-2, rtol=1e-2, equal_nan=True):
            print(f"✅ {things_to_compare[i][1]} and {things_to_compare[0][1]} match")
        else:
            print(f"❌ {things_to_compare[i][1]} and {things_to_compare[0][1]} differ")
            max_diff_idx = torch.argmax(torch.abs(outputs[i] - outputs[0]))
            max_diff = torch.abs(outputs[i].reshape(-1)[max_diff_idx] - outputs[0].reshape(-1)[max_diff_idx])
            print(f"max diff: {max_diff} for {max_diff_idx} index, a: {outputs[i].reshape(-1)[max_diff_idx]}, b: {outputs[0].reshape(-1)[max_diff_idx]}")
            bad_i = i

    if bad_i >= 0 and PRINT_BAD_TENSOR:
        np.savetxt('bad_output.txt', outputs[bad_i].cpu().numpy(), fmt="%.2e")


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
    # run_unit_test(1024)
    # run_unit_test(4096)
    # run_unit_test(2763)
    # run_unit_test(1783)
    # use to benchmark
    benchmark.run(show_plots=True, print_data=True)

