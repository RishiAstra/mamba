# some code from https://triton-lang.org/main/getting-started/tutorials/...

# TODO: check correctness for out, out_x, dt, dA_cumsum, states, final_states, not just out
BENCHMARK_REPEATS = 25

import torch

import triton
import triton.language as tl
import numpy as np

from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd

def run_original_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, method=None)
    return outputs[0]
    # return out, out_x, dt, dA_cumsum, states, final_states

def run_fused_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, method='fused')
    return outputs[0]
    # return out, out_x, dt, dA_cumsum, states, final_states

def run_fullyfused_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D, z, dt_bias, method='fullyfused')
    return outputs[0]
    # return out, out_x, dt, dA_cumsum, states, final_states

things_to_compare = [
    (run_original_ssd, "Original", "blue"),
    # (run_fused_ssd, "Fused", "red"),
    (run_fullyfused_ssd, "Fully Fused", "green"),
]

DEVICE = triton.runtime.driver.active.get_active_torch_device()

configs = []

# dt: torch.Size([1, 6, 80]), dt_bias: torch.Size([80]), A: torch.Size([80]), B: torch.Size([1, 6, 1, 128]),
# C: torch.Size([1, 6, 1, 128]), D: torch.Size([80]), x: torch.Size([1, 6, 80, 64]), chunk_size: 256
# dtypes: dt: torch.float16, dt_bias: torch.float16, A: torch.float32, B: torch.float16,
# C: torch.float16, D: torch.float32, x: torch.float16
# batch, seqlen, nheads, headdim = x.shape
# _, _, ngroups, dstate = B.shape
# assert nheads % ngroups == 0
# assert B.shape == (batch, seqlen, ngroups, dstate)
# assert x.shape == (batch, seqlen, nheads, headdim)
# assert dt.shape == (batch, seqlen, nheads)
# assert A.shape == (nheads,)
# assert C.shape == B.shape

batch       = 1
# seqlen    = whatever
nheads      = 80
headdim     = 64
ngroups     = 1
dstate      = 128

def get_test_size(seqlen):
    return (batch, seqlen, nheads, headdim, ngroups, dstate)

test_sizes = [
    # (get_test_size(1024 * 2 ** i),) for i in range(0, 9, 1)
    (get_test_size(1024 * 2 ** i),) for i in [6] # for quick test
]

configs.append(
    triton.testing.Benchmark(
        x_names = ["dims"],
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

def get_rand_input(dims):
    batch, seqlen, nheads, headdim, ngroups, dstate = dims
    torch.manual_seed(0)

    dt = torch.randn((batch, seqlen, nheads), dtype=torch.float16, device=DEVICE) * 0.01 + 0.05
    dt_bias = torch.randn((nheads,), dtype=torch.float16, device=DEVICE) * 0.01
    A = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 0.01 + 0.03
    B = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 0.1 + 0.3
    C = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 0.1 + 0.3
    D = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 0.01 + 0.95
    x = torch.randn((batch, seqlen, nheads, headdim,), dtype=torch.float16, device=DEVICE) * 0.4

    return dt, dt_bias, A, B, C, D, x


def run_unit_test(seqlen):
    # raise Exception("Not implemented")
    dt, dt_bias, A, B, C, D, x = get_rand_input(get_test_size(seqlen))
    CHUNK_SIZE=256

    outputs = [thing[0](x, dt, A, B, C, CHUNK_SIZE, D=D, z=None, dt_bias=dt_bias) for thing in things_to_compare]
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
def benchmark(dims, provider):
    batch, seqlen, nheads, headdim, ngroups, dstate = dims

    dt, dt_bias, A, B, C, D, x = get_rand_input(dims)
    CHUNK_SIZE=256

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench( \
        lambda: things_to_compare[provider][0](x, dt, A, B, C, CHUNK_SIZE, D=D, z=None, dt_bias=dt_bias), \
        rep=BENCHMARK_REPEATS, quantiles=quantiles
    )
    perf = lambda ms: seqlen / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    # use to check correctness
    # run_unit_test(1024)
    # use to benchmark
    benchmark.run(show_plots=True, print_data=True)

