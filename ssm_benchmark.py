# some code from https://triton-lang.org/main/getting-started/tutorials/...

####################################################################################################
#               Benchmark Settings                                                                 #
####################################################################################################

# check correctness or benchmark
CHECK_CORRECTNESS = True
atol = 0#1e-3
rtol = 0#1e-3


# more settings for test tensors, probably best to leave as is
states_in_fp32 = False
init_states_fp32 = False
fused_type = "high"
have_init_states    = True
have_dt_softplus    = True # TODO: test more
have_z = False
have_seq_idx        = False

# the chunk sizes to use for original and fused kernels
# note that fp32 does best with 256, fp16 with 128
CHUNK_SIZE_ORIGINAL =128
CHUNK_SIZE_FUSED    =128

# dimensions to test
batch       = 1
# seqlen    = set per test
nheads      = 80
headdim     = 64
ngroups     = 1
dstate      = 128

def get_test_size(seqlen):
    return (batch, seqlen, nheads, headdim, ngroups, dstate)

test_sizes = [
    (get_test_size(1024 * 2 ** i),) for i in range(0, 9, 1) #9 for batch=1, 6 for 8, 4 for 32 so that original doesn't fail
    # (get_test_size(1024 * 2 ** i),) for i in [7] # for quick test
]

# Time to run each benchmark (ms)
BENCHMARK_REPEATS = 1000

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
def run_original_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, states_in_fp32=states_in_fp32, mamba2_fusion_type="unfused")
    if CHUNK_SIZE_ORIGINAL != CHUNK_SIZE_FUSED: # can't compare some outputs if chunk sizes differ
        outputs = outputs[0], outputs[1], None, None, None, outputs[5]
    return outputs

def run_fused_ssd(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus):
    outputs = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, states_in_fp32=states_in_fp32, mamba2_fusion_type=fused_type)
    if CHUNK_SIZE_ORIGINAL != CHUNK_SIZE_FUSED:
        outputs = outputs[0], outputs[1], None, None, None, outputs[5]
    return outputs

things_to_compare = [
    (run_original_ssd, "Original", "blue"),
    (run_fused_ssd, "Fused", "red"),
]

DEVICE = 'cuda'

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

def get_rand_input(dims_b_seq_nh_hd_ng_ds, is_original=True):
    batch, seqlen, nheads, headdim, ngroups, dstate = dims_b_seq_nh_hd_ng_ds
    torch.manual_seed(0)
    random.seed(0)

    dt = torch.randn((batch, seqlen, nheads), dtype=torch.float16, device=DEVICE) * 0.2 + 0.5
    dt_bias = torch.randn((nheads,), dtype=torch.float16, device=DEVICE) * 0.5 - 5
    A = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 3 - 10
    B = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 3 + 7
    C = torch.randn((batch, seqlen, ngroups, dstate), dtype=torch.float16, device=DEVICE) * 5 + 20
    D = torch.randn((nheads,), dtype=torch.float32, device=DEVICE) * 0.5 + 1.2
    x = torch.randn((batch, seqlen, nheads, headdim,), dtype=torch.float16, device=DEVICE) * 2 + 5

    if have_z:
        z = torch.randn((batch, seqlen, nheads, headdim,), dtype=torch.float16, device=DEVICE) * 1
    else:
        z = None
    if have_init_states:
        initial_states = torch.randn((batch, nheads, headdim, dstate), dtype=torch.float16 if not init_states_fp32 else torch.float32, device=DEVICE) * 0.2
    else:
        initial_states = None

    if have_seq_idx:
        # example at https://github.com/state-spaces/mamba/issues/383

        seq_idx = torch.zeros((batch, seqlen), dtype=torch.int32, device='cpu')
        # have at least 1 batch not have multiple sequences if b > 1
        no_seq_idx_b = -1
        if batch > 1:
            no_seq_idx_b = random.randint(0, batch)
        max_part_seqlen = seqlen // 4
        for b in range(batch):
            split = random.randint(1, max_part_seqlen)
            while split < seqlen and b != no_seq_idx_b:
                seq_idx[b, split] = 1
                split += random.randint(1, max_part_seqlen)

        seq_idx = seq_idx.to(device=DEVICE, dtype=torch.int32)
        seq_idx = torch.cumsum(seq_idx, dim=-1, dtype=torch.int32)
        # print(seq_idx)

    else:
        seq_idx = None

    return dt, dt_bias, A, B, C, D, x, z, initial_states, seq_idx, None #, cu_seqlens

def idx_to_pos(i, tensor):
    return tuple([
        # for each dimension, get the index by dividing by the stride and modding by the shape
        (i // tensor.stride(dim)) % tensor.size(dim) for dim in range(tensor.dim())
    ])

def run_unit_test(seqlen):
    outputs_full = []

    for i, thing in enumerate(things_to_compare):
        dt, dt_bias, A, B, C, D, x, z, initial_states, seq_idx, cu_seqlens = get_rand_input(get_test_size(seqlen), is_original=i==0)

        outputs_full.append(thing[0](
            x, dt, A, B, C, CHUNK_SIZE_ORIGINAL if i == 0 else CHUNK_SIZE_FUSED, D=D, z=z, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus
        ))

    field_names = ["out", "out_x", "dt", "dA_cumsum", "states", "final_states"] # 5th is CB
    for field_idx in range(len(outputs_full[0])):
        if outputs_full[0][field_idx] is None:
            continue
        print(f"comparing field {field_names[field_idx]}")
        # # CB causal, zero out triangle
        # if field_names[field_idx] == "CB":
        #     for output in outputs_full:
        #         output[field_idx].copy_(torch.tril(output[field_idx]))

        SAVE_BAD_TENSOR = False
        bad_tensor_idx_i = -1
        # compare all to the first
        for i in range(1, len(outputs_full), 1):
            outputs_0 = outputs_full[0][field_idx]
            outputs_i = outputs_full[i][field_idx]
            print(f"ref shape: {outputs_0.shape}, test shape: {outputs_i.shape}")
            outputs_i = outputs_i.to(outputs_0.dtype)
            if torch.allclose(outputs_i, outputs_0, atol=atol, rtol=rtol, equal_nan=True):
                print(f"✅ {things_to_compare[i][1]} and {things_to_compare[0][1]} match")
            else:
                print(f"❌ {things_to_compare[i][1]} and {things_to_compare[0][1]} differ")
                bad_tensor_idx_i = i

            equal_nan = (torch.isnan(outputs_0) & torch.isnan(outputs_i))
            equal_nan_count = equal_nan.sum().cpu().item()
            if equal_nan_count > 0:
                print(f"Note: there are {equal_nan_count} NaN elements that match")
                outputs_0 = outputs_0.masked_fill(equal_nan, 0.0)
                outputs_i = outputs_i.masked_fill(equal_nan, 0.0)

            equal_inf = (torch.isinf(outputs_0) & torch.isinf(outputs_i) & (torch.sign(outputs_0) == torch.sign(outputs_i)))
            equal_inf_count = equal_inf.sum().cpu().item()
            if equal_inf_count > 0:
                print(f"Note: there are {equal_inf_count} Inf elements that match")
                outputs_0 = outputs_0.masked_fill(equal_inf, 0.0)
                outputs_i = outputs_i.masked_fill(equal_inf, 0.0)

            output_0_inf = torch.isinf(outputs_0)
            output_i_inf = torch.isinf(outputs_i)
            output_0_inf_count = output_0_inf.sum().cpu().item()
            output_i_inf_count = output_i_inf.sum().cpu().item()
            # swap inf for max value in order to get closeness
            if output_0_inf_count > 0:
                # 65504.0
                print(f"Warning: there are {output_0_inf_count} Inf elements in reference output")
                outputs_0 = outputs_0.masked_fill(output_0_inf & (torch.sign(outputs_0) > 0), 65504.0)
                outputs_0 = outputs_0.masked_fill(output_0_inf & (torch.sign(outputs_0) < 0), -65504.0)
            if output_i_inf_count > 0:
                print(f"Warning: there are {output_i_inf_count} Inf elements in test output")
                outputs_i = outputs_i.masked_fill(output_i_inf & (torch.sign(outputs_i) > 0), 65504.0)
                outputs_i = outputs_i.masked_fill(output_i_inf & (torch.sign(outputs_i) < 0), -65504.0)

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
    _, seqlen, _, _, _, _ = dims_b_seq_nh_hd_ng_ds
    dt, dt_bias, A, B, C, D, x, z, initial_states, seq_idx, cu_seqlens = get_rand_input(dims_b_seq_nh_hd_ng_ds)

    ms = triton.testing.do_bench(
        lambda: things_to_compare[provider][0](
            x, dt, A, B, C, CHUNK_SIZE_ORIGINAL if provider == 0 else CHUNK_SIZE_FUSED, D=D, z=z, dt_bias=dt_bias,
            initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=have_dt_softplus), \
        rep=BENCHMARK_REPEATS, return_mode="median"
    )

    return batch * seqlen / (ms * 1e-3)

if __name__ == "__main__":
    if CHECK_CORRECTNESS:
        run_unit_test(2763)
    else:
        benchmark.run(show_plots=True, print_data=True)

