# some code from https://triton-lang.org/main/getting-started/tutorials/...

####################################################################################################
#               Benchmark Settings                                                                 #
####################################################################################################

# check correctness or benchmark
CHECK_CORRECTNESS = True

# more settings for test tensors, probably best to leave as is
have_init_states    = True
have_dt_softplus    = True # TODO: test more

# the chunk sizes to use for original and fused kernels
# note that the original kernel and fused kernel have different optimal chunk sizes
CHUNK_SIZE_ORIGINAL=128#256
CHUNK_SIZE_FUSED=128

# dimensions to test
# seqlen    = set per test
nheads      = 80
headdim     = 64
ngroups     = 1
dstate      = 128

def get_test_size(seqlen):
    return (seqlen, nheads, headdim, ngroups, dstate)

test_sizes = [
    (get_test_size(1024 * 2 ** i),) for i in range(0, 9, 1) #9 for batch=1, 6 for 8, 4 for 32 so that original doesn't fail
    # (get_test_size(1024 * 2 ** i),) for i in [7] # for quick test
]

# I think this is the time to run each benchmark (ms)
BENCHMARK_REPEATS = 200

####################################################################################################
####################################################################################################
####################################################################################################


# imports
import random
import torch
import triton
import numpy as np
from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_fwd

# test functions
def run_original_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, cu_chunk_seqlens, dt_softplus):
    out = torch.empty_like(x)
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, out, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, cu_chunk_seqlens=cu_chunk_seqlens, dt_softplus=dt_softplus, fused=False)
    if CHUNK_SIZE_ORIGINAL != CHUNK_SIZE_FUSED: # can't compare some outputs if chunk sizes differ
        outputs = outputs[0], outputs[1], None, None, None, None, outputs[5]
    return outputs

def run_fused_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, cu_chunk_seqlens, dt_softplus):
    out = torch.empty_like(x)
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, out, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, cu_chunk_seqlens=cu_chunk_seqlens, dt_softplus=dt_softplus, fused=True)#
    if CHUNK_SIZE_ORIGINAL != CHUNK_SIZE_FUSED:
        outputs = outputs[0], outputs[1], None, None, None, None, outputs[5]
    return outputs

things_to_compare = [
    (run_original_ssd, "Original", "blue"),
    (run_fused_ssd, "Fused", "red"),
]

DEVICE = triton.runtime.driver.active.get_active_torch_device()

configs = []
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

def get_rand_input(dims_b_seq_nh_hd_ng_ds, fused=False):
    seqlen, nheads, headdim, ngroups, dstate = dims_b_seq_nh_hd_ng_ds
    torch.manual_seed(0)
    random.seed(0)

    dt = torch.randn((seqlen, nheads), dtype=torch.float16, device=DEVICE) * 0.2 + 0.5
    dt_bias = torch.randn((nheads,), dtype=torch.float16, device=DEVICE) * 0.5 - 5
    A = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 3 - 10
    B = torch.randn((seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 3 + 7
    C = torch.randn((seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 5 + 20
    D = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 0.5 + 1.2
    x = torch.randn((seqlen, nheads, headdim,), dtype=torch.float16, device=DEVICE) * 2 + 5

    # example at https://github.com/state-spaces/mamba/issues/383

    # seq_idx = torch.zeros((seqlen), dtype=torch.int32, device='cpu')
    # max_part_seqlen = seqlen // 4
    # split = random.randint(1, max_part_seqlen)
    # while split < seqlen:
    #     seq_idx[split] = 1
    #     split += random.randint(1, max_part_seqlen)

    # seq_idx = seq_idx.to(device=DEVICE, dtype=torch.int32)
    # seq_idx = torch.cumsum(seq_idx, dim=-1, dtype=torch.int32)
    # print(seq_idx)

    chunk_size = CHUNK_SIZE_FUSED if fused else CHUNK_SIZE_ORIGINAL

    seq_idx_cpu = [] # holds per chunk seq idx (the request #)
    cu_seqlens_cpu = [0] # holds bounds for sequences (inclusive, exclusive)
    cu_chunk_seqlens_cpu = [0] # holds token# of chunk boundaries
    # make random sequence lengths
    max_part_seqlen = seqlen // 4#32 # each sequence at most 1/4 of total length for a basic test
    split = random.randint(1, max_part_seqlen)
    while True: #split < seqlen:
        # mark split
        cu_seqlens_cpu.append(split)
        # process chunks
        seq_start = cu_seqlens_cpu[-2]
        seq_end = split
        for chunk_i in range(seq_start, seq_end, chunk_size):
            cu_chunk_seqlens_cpu.append(min(chunk_i + chunk_size, seq_end))
            seq_idx_cpu.append(len(cu_seqlens_cpu) - 2)

        if split >= seqlen:
            break
        split += random.randint(1, max_part_seqlen)
        if split >= seqlen:
            split = seqlen

    # convert
    seq_idx = torch.tensor(seq_idx_cpu, dtype=torch.int32, device='cuda')
    cu_seqlens = torch.tensor(cu_seqlens_cpu, dtype=torch.int32, device='cuda')
    cu_chunk_seqlens = torch.tensor(cu_chunk_seqlens_cpu, dtype=torch.int32, device='cuda')

    
    if have_init_states:
        initial_states = torch.randn((len(cu_seqlens_cpu) - 1, nheads, headdim, dstate), dtype=torch.float16, device=DEVICE) * 0.2
    else:
        initial_states = None

    return dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens, cu_chunk_seqlens

def idx_to_pos(i, tensor):
    return tuple([
        # for each dimension, get the index by dividing by the stride and modding by the shape
        (i // tensor.stride(dim)) % tensor.size(dim) for dim in range(tensor.dim())
    ])

def run_unit_test(seqlen):
    outputs_full = []

    for i, thing in enumerate(things_to_compare):
        dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens, cu_chunk_seqlens = get_rand_input(get_test_size(seqlen), fused=(i == 1))

        outputs_full.append(thing[0](
            x, dt, A, B, C, CHUNK_SIZE_ORIGINAL if i == 0 else CHUNK_SIZE_FUSED, D=D, z=None, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, cu_chunk_seqlens=cu_chunk_seqlens, dt_softplus=have_dt_softplus
        ))

    _, _, _, _, _, _, _, _, _, _, cu_chunk_seqlens = get_rand_input(get_test_size(seqlen), fused=(i == 1))

    field_names = ["out", "out_x", "dt", "dA_cumsum", "CB", "states", "final_states"]
    for field_idx in range(len(outputs_full[0])):
        outputs_0 = outputs_full[0][field_idx]
        if outputs_0 is None: # skip None
            continue
        print(f"comparing field {field_names[field_idx]}")

        # CB could have partial chunks, need to zero out for comparison
        if field_names[field_idx] == "CB":
            for output in outputs_full:
                for i in range(len(cu_chunk_seqlens)-1):
                    chunk_len = cu_chunk_seqlens[i+1] - cu_chunk_seqlens[i]
                    if output[field_idx] is not None:
                        output[field_idx][i, :, chunk_len:, :] = 0
        # CB causal, zero out triangle
        if field_names[field_idx] == "CB":
            for output in outputs_full:
                output[field_idx].copy_(torch.tril(output[field_idx]))

        SAVE_BAD_TENSOR = False
        bad_tensor_idx_i = -1
        # compare all to the first
        for i in range(1, len(outputs_full), 1):
            outputs_i = outputs_full[i][field_idx]
            print(f"ref shape: {outputs_0.shape}, test shape: {outputs_i.shape}")
            # atol = 2.5e-3
            # rtol = 1e-2
            # from vLLM tests
            atol, rtol = 5e-3, 5e-3
            outputs_i = outputs_i.to(outputs_0.dtype)
            if torch.allclose(outputs_i, outputs_0, atol=atol, rtol=rtol, equal_nan=True):
                print(f"✅ {things_to_compare[i][1]} and {things_to_compare[0][1]} match")
            else:
                print(f"❌ {things_to_compare[i][1]} and {things_to_compare[0][1]} differ")
                bad_tensor_idx_i = i
            max_diff_idx = torch.argmax(torch.abs(outputs_i - outputs_0))
            max_diff_pos = idx_to_pos(max_diff_idx.item(), outputs_0)
            max_diff = torch.abs(outputs_i.reshape(-1)[max_diff_idx] - outputs_0.reshape(-1)[max_diff_idx])
            print(f"max diff: {max_diff} for {max_diff_pos} index, test: {outputs_i.reshape(-1)[max_diff_idx]}, expected: {outputs_0.reshape(-1)[max_diff_idx]}")
            max_allowed_diff = atol + rtol * torch.abs(outputs_0)
            max_diff_frac = torch.abs(outputs_i - outputs_0) / max_allowed_diff
            max_rdiff_idx = torch.argmax(max_diff_frac)
            max_rdiff_pos = idx_to_pos(max_rdiff_idx.item(), outputs_0)
            max_rdiff = max_diff_frac.reshape(-1)[max_rdiff_idx]
            print(f"max diff score: {max_rdiff} for {max_rdiff_pos} index, test: {outputs_i.reshape(-1)[max_rdiff_idx]}, expected: {outputs_0.reshape(-1)[max_rdiff_idx]}")
            fail_bools = torch.abs(outputs_i - outputs_0) > atol + rtol * torch.abs(outputs_0)
            fail_bool_num = fail_bools.sum().cpu().item()
            total_elems = outputs_0.numel()
            print(f"failed elements / total: {fail_bool_num} / {total_elems}, {fail_bool_num / total_elems * 100.0}%")


        if bad_tensor_idx_i >= 0 and SAVE_BAD_TENSOR:
            np.savetxt(f'bad_output_{field_idx}.txt', outputs_full[bad_tensor_idx_i][field_idx].reshape(-1, outputs_full[bad_tensor_idx_i][field_idx].shape[-1]).cpu().numpy(), fmt="%.2e")

# triton benchmark function
@triton.testing.perf_report(configs)
def benchmark(dims_b_seq_nh_hd_ng_ds, provider):
    seqlen, _, _, _, _ = dims_b_seq_nh_hd_ng_ds
    dt, dt_bias, A, B, C, D, x, initial_states, seq_idx, cu_seqlens, cu_chunk_seqlens = get_rand_input(dims_b_seq_nh_hd_ng_ds, fused=(provider == 1))

    ms = triton.testing.do_bench(
        lambda: things_to_compare[provider][0](
            x, dt, A, B, C, CHUNK_SIZE_ORIGINAL if provider == 0 else CHUNK_SIZE_FUSED, D=D, z=None, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, cu_chunk_seqlens=cu_chunk_seqlens, dt_softplus=have_dt_softplus), \
        rep=BENCHMARK_REPEATS, return_mode="median"
    )

    return seqlen / (ms * 1e-3)

if __name__ == "__main__":
    if CHECK_CORRECTNESS:
        run_unit_test(2763)
    else:
        benchmark.run(show_plots=True, print_data=True)

