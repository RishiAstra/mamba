// mamba2_fused_cuda.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <cuda_fp16.h>
#include <tuple>

#include "common.h"

// Compile-time tuning bag (we'll keep CHUNK_SIZE unused for now; runtime param is passed)
template <uint32_t CHUNK_SIZE_, uint32_t THREADS_, uint32_t TARGET_BLOCKS_, uint32_t CUMSUM_H_BLOCK_>
struct ssd_template_params {
    static constexpr uint32_t CHUNK_SIZE     = CHUNK_SIZE_;
    static constexpr uint32_t THREADS        = THREADS_;
    static constexpr uint32_t TARGET_BLOCKS  = TARGET_BLOCKS_;
    static constexpr uint32_t CUMSUM_H_BLOCK = CUMSUM_H_BLOCK_;
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of 32");
};

// --- helpers ---
__device__ __forceinline__ float softplusf(float x) {
    // match triton's guard: skip exp for large x
    return (x <= 20.f) ? log1pf(expf(x)) : x;
}

// Step 1 kernel: (dt[, +bias][, softplus]) -> dt_out; dA = dt * A -> cumsum along chunk
// One block handles a single (batch b, chunk c, head h). Thread 0 does the serial scan
// (correctness-first; we can make this a parallel scan later).
template <typename Params>
__global__ __launch_bounds__(Params::THREADS, Params::TARGET_BLOCKS)
void mamba2_ssd_step1_dt_transform_cumsum_kernel(
    const half* __restrict__ dt,     // [B, S, H] contiguous
    const float*  __restrict__ A,      // [H]
    const half* __restrict__ dt_bias,// [H]
    int B, int S, int H,
    bool dt_softplus,
    float* __restrict__ dt_out,        // [B, H, n_chunks, chunk_size] fp32
    float* __restrict__ dA_cumsum      // same shape
) {
    static_assert(Params::CHUNK_SIZE >= Params::CUMSUM_H_BLOCK);
    static_assert(32 >= Params::CUMSUM_H_BLOCK);

    const int tid  = threadIdx.x;

    // mask off if too many threads
    constexpr uint32_t N_THREADS = (Params::THREADS <= Params::CHUNK_SIZE) ? Params::THREADS : Params::CHUNK_SIZE;
    if (tid >= N_THREADS) return; // out of range

    const int b = blockIdx.y;
    const int chunk_id = blockIdx.x;
    const int h_base = blockIdx.z * Params::CUMSUM_H_BLOCK;

    if (b >= B || h_base >= H) return;

    const int s_start = chunk_id * Params::CHUNK_SIZE;
    if (s_start >= S) return;
    const int this_chunk_len = min(Params::CHUNK_SIZE, S - s_start);

    const int n_chunks = DIV_UP(S, Params::CHUNK_SIZE);


    float A_h[Params::CUMSUM_H_BLOCK];
    float bias_h[Params::CUMSUM_H_BLOCK];
    constexpr int WARPS = N_THREADS / 32;
    __shared__ float warp_sums[WARPS][Params::CUMSUM_H_BLOCK];
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    float v[Params::CUMSUM_H_BLOCK];
    half2 dt_block[DIV_UP(Params::CUMSUM_H_BLOCK, 2)][1]; // TODO: support elements per thread

    // load dt block
    const int s_off = tid;// + elem_idx * N_THREADS;
    // if (s_off >= Params::CHUNK_SIZE) break; // TODO: support extra threads (chunk size < block size)
    const int s = s_start + static_cast<int>(s_off);
    if (s >= S) {
        // zero pad tail TODO: check if correct, don't hardcode head block
        dt_block[0][0] = __float2half2_rn(0.f);
        dt_block[1][0] = __float2half2_rn(0.f); // TODO: more flexible
    } else {
        const int h0 = h_base;
        const half* p = dt + ((static_cast<int64_t>(b) * S + s) * H + h0);

        // Fast path: we can read 4 consecutive heads as two half2 loads
        if (h0 + 3 < H && ((reinterpret_cast<uintptr_t>(p) & 0x7) == 0)) { // align to 8B to match type
            uint2 v4;
            asm volatile(
                "ld.global.L1::evict_last.L2::256B.v2.s32 {%0, %1}, [%2];\n"
                : "=r"(v4.x), "=r"(v4.y)
                : "l"(p)
                : "memory"
            );
            dt_block[0][0] = *reinterpret_cast<const half2*>(&v4.x);
            dt_block[1][0] = *reinterpret_cast<const half2*>(&v4.y);
        } else {
            // Fallback: scalar (handles edges / misalignment)
            #pragma unroll
            for (int hh = 0; hh < 4; hh += 2) {
                const int h = h0 + hh;
                // TODO: check if correct
                bool b0 = (h < H);
                bool b1 = (h + 1 < H);
                half2 v = __halves2half2(b0 ? p[hh] : __float2half(0.f), b1 ? p[hh + 1] : __float2half(0.f));
                dt_block[hh / 2][0] = v;
            }
        }
    }


    #pragma unroll
    for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) {
        const int h = h_base + rep;
        A_h[rep] = (h < H) ? A[h] : 0.f;
        bias_h[rep] = (h < H && dt_bias != nullptr) ? __half2float(dt_bias[h]) : 0.f;
    }

    // TODO: support chunk size != block size
    #pragma unroll
    for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) {
        const int h = h_base + rep;
        // TODO: any masking or stuff, dt_block should already be zero-padded
        
        bool low_bits = rep % 2 == 0;
        float dt_val = (tid < this_chunk_len) ? (__half2float(low_bits ? __low2half(dt_block[rep / 2][0]) : __high2half(dt_block[rep / 2][0])) + bias_h[rep]) : 0.f; // TODO: support multiple elems per thread

        if (tid < this_chunk_len && dt_softplus) dt_val = softplusf(dt_val);
        // store dt_out[b,h,chunk,tid]
        const int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * n_chunks + chunk_id) * Params::CHUNK_SIZE + tid;
        // TODO: bounds check
        // // basic write
        // dt_out[out_idx] = dt_val;
        // evict first write
        float* p_out = dt_out + out_idx;
        __stcs(p_out, dt_val);

        v[rep] = dt_val * A_h[rep]; // keep any 0-padding for easy cumsum
    }

    // intra-warp inclusive scan
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        #pragma unroll
        for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) {
            float n = __shfl_up_sync(mask, v[rep], offset);
            if (lane >= offset) v[rep] += n;
        }
    }

    // write warp totals to shared memory
    // for now assume chunk size == block size
    // to avoid shuffle, only last lane (which has the final value) writes
    if (lane == 31) {
        #pragma unroll
        for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) {
            warp_sums[warp_id][rep] = v[rep];
        }
    }

    // wait for warp sums
    __syncthreads();

    // do prefix duplicated work to avoid sync
    // all threads load the warp prefix they need
    #pragma unroll
    for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) {
        const int h = h_base + rep;
        // add prefix of all previous warps
        #pragma unroll
        for (int w = 0; w < min(warp_id, WARPS); ++w) {
            if (h < H) v[rep] += warp_sums[w][rep];
        }

        // write dA_cumsum
        const int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * n_chunks + chunk_id) * Params::CHUNK_SIZE + tid;
        if (tid < Params::CHUNK_SIZE) {
            // // basic write
            // dA_cumsum[out_idx] = v[rep];
            // evict first write
            float* p_out = dA_cumsum + out_idx;
            __stcs(p_out, v[rep]);
        }
    }
}

template<typename T>
const T* get_optional_ptr(const c10::optional<at::Tensor>& t_opt) {
    return (t_opt && t_opt->defined()) ? t_opt->const_data_ptr<T>() : nullptr;
}

// ---------- public entry ----------
std::tuple<c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>> mamba2_cumsum_fwd_cuda(
    const Mamba2SSDArgs& args
) {
    // TODO: compile multiple versions, for different config
    // NOTE: we do not check for ptr overlap, since it would be crazy to overlap stuff like x and A
    // TODO: consider checking ptr overlap

    // --- launch kernel (correctness-first mapping: 1 head per block) ---
    using Params = ssd_template_params</*CHUNK_SIZE*/128, /*THREADS*/128, /*TARGET_BLOCKS*/4, /*CUMSUM_H_BLOCK*/4>; // can actually fit like 12 blocks instead of 4
    dim3 block(Params::THREADS);
    dim3 grid(args.n_chunks, args.Bsz, DIV_UP(args.H, Params::CUMSUM_H_BLOCK));
    auto stream = at::cuda::getCurrentCUDAStream();

    mamba2_ssd_step1_dt_transform_cumsum_kernel<Params><<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(args.dt.const_data_ptr<at::Half>()),
        args.A.const_data_ptr<float>(),
        reinterpret_cast<const half*>(get_optional_ptr<at::Half>(args.dt_bias)), // TODO: handle null
        args.Bsz, args.S, args.H,
        args.dt_softplus,
        args.dt_out.data_ptr<float>(),
        args.dA_cumsum.data_ptr<float>()
    );

#ifndef NDEBUG
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA launch failed (step1): ", cudaGetErrorString(err));
#endif

    // TODO: when we write the rest of the fused pipeline, return a tuple:
    // (out, out_x, dt_out, dA_cumsum, states, final_states)

    return std::make_tuple(c10::nullopt, c10::nullopt, args.dt_out, args.dA_cumsum, c10::nullopt, c10::nullopt); // placeholders for future outputs
}
