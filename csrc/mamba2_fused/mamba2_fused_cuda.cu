// mamba2_fused_cuda.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <cuda_fp16.h>
#include <tuple>

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

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
    const __half* __restrict__ dt,     // [B, S, H] contiguous
    const float*  __restrict__ A,      // [H]
    const __half* __restrict__ dt_bias,// [H]
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


    __shared__ float shared_A_h[Params::CUMSUM_H_BLOCK];
    __shared__ float shared_bias_h[Params::CUMSUM_H_BLOCK];
    float A_h[Params::CUMSUM_H_BLOCK];
    float bias_h[Params::CUMSUM_H_BLOCK];
    constexpr int WARPS = N_THREADS / 32;
    __shared__ float warp_sums[WARPS][Params::CUMSUM_H_BLOCK];
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    float v[Params::CUMSUM_H_BLOCK];
    __shared__ half dt_block[Params::CUMSUM_H_BLOCK][Params::CHUNK_SIZE];

    if constexpr(Params::CUMSUM_H_BLOCK == 1) {
        A_h[0] = A[h_base];
        bias_h[0] = __half2float(dt_bias[h_base]);
    } else {
        // load in parallel
        if (tid < Params::CUMSUM_H_BLOCK) {
            const int h = h_base + tid;
            if (h < H) {
                shared_A_h[tid] = A[h];
                shared_bias_h[tid] = __half2float(dt_bias[h]);
            }
        }
        // load dt block
        if constexpr (Params::CUMSUM_H_BLOCK == 4) {
            // Each thread copies multiple s positions for this (b, chunk, h_base..h_base+3)
            for (uint32_t s_off = threadIdx.x; s_off < Params::CHUNK_SIZE; s_off += N_THREADS) {
                const int s = s_start + static_cast<int>(s_off);
                if (s >= S) {
                    // zero pad tail
                    #pragma unroll
                    for (int hh = 0; hh < 4; ++hh) {
                        if (h_base + hh < H) dt_block[hh][s_off] = __float2half(0.f);
                    }
                    continue;
                }
                const int h0 = h_base;
                const __half* p = dt + ((static_cast<int64_t>(b) * S + s) * H + h0);

                // Fast path: we can read 4 consecutive heads as two __half2 loads
                if (h0 + 3 < H && ((reinterpret_cast<uintptr_t>(p) & 0x7) == 0)) { // align to 8B to match type
                    uint64_t v64 = __ldcg(reinterpret_cast<const uint64_t*>(p));

                    dt_block[0][s_off] = __ushort_as_half(static_cast<uint16_t>(v64 & 0xFFFF));
                    dt_block[1][s_off] = __ushort_as_half(static_cast<uint16_t>((v64 >> 16) & 0xFFFF));
                    dt_block[2][s_off] = __ushort_as_half(static_cast<uint16_t>((v64 >> 32) & 0xFFFF));
                    dt_block[3][s_off] = __ushort_as_half(static_cast<uint16_t>((v64 >> 48) & 0xFFFF));
                } else {
                    // Fallback: scalar (handles edges / misalignment)
                    #pragma unroll
                    for (int hh = 0; hh < 4; ++hh) {
                        const int h = h0 + hh;
                        dt_block[hh][s_off] = (h < H) ? p[hh] : __float2half(0.f);
                    }
                }
            }
        } else {
            for (uint32_t id = tid; id < Params::CHUNK_SIZE * Params::CUMSUM_H_BLOCK; id += N_THREADS) {
                const int s_off = id % Params::CHUNK_SIZE;
                const int s = s_start + s_off;
                const int h_off = id / Params::CHUNK_SIZE;
                const int h = h_base + h_off;
                if (h < H && s < S) {
                    const int64_t dt_idx = ((static_cast<int64_t>(b) * S + s) * H + h);
                    dt_block[h_off][s_off] = dt[dt_idx];
                }
            }
        }
        __syncthreads();

        // get local copy, won't be modified from now on
        #pragma unroll
        for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) {
            const int h = h_base + rep;
            if (h < H) {
                A_h[rep] = shared_A_h[rep];
                bias_h[rep] = shared_bias_h[rep];
            }
        }
    }

    // TODO: support chunk size != block size
    #pragma unroll
    for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) {
        const int h = h_base + rep;
        // compute dt_val (masked for tail) and write dt_out
        float dt_val = 0.f;
        if (tid < this_chunk_len) {
            dt_val = __half2float(dt_block[rep][tid]) + bias_h[rep]; // TODO: support less threads than chunk size
            if (dt_softplus) dt_val = softplusf(dt_val);
            // store dt_out[b,h,chunk,tid]
            const int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * n_chunks + chunk_id) * Params::CHUNK_SIZE + tid;
            dt_out[out_idx] = dt_val;
        } else {
            // ensure padded tail is deterministic zero
            const int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * n_chunks + chunk_id) * Params::CHUNK_SIZE + tid;
            if (tid < Params::CHUNK_SIZE) dt_out[out_idx] = 0.f;
        }
        v[rep] = (tid < this_chunk_len) ? (dt_val * A_h[rep]) : 0.f;
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

    // TODO: fix for chunk_size != block_size
    // compute warp total robustly (handle partial last warp)
    const int warp_base = warp_id * 32;
    int local_count = this_chunk_len - warp_base;
    if (local_count < 0) local_count = 0;
    if (local_count > 32) local_count = 32;
    float warp_total[Params::CUMSUM_H_BLOCK];
    #pragma unroll
    for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) warp_total[rep] = (local_count > 0) ? __shfl_sync(mask, v[rep], local_count - 1) : 0.f;
    if (lane < Params::CUMSUM_H_BLOCK) warp_sums[warp_id][lane] = warp_total[lane];

    // second stage gets across warps
    __syncthreads();
    // small serial prefix over warps (W <= 8 here)
    if (threadIdx.x < Params::CUMSUM_H_BLOCK) {
        float accw = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            float t = warp_sums[w][threadIdx.x];
            accw += t;
            warp_sums[w][threadIdx.x] = accw;  // inclusive
        }
    }
    __syncthreads();

    #pragma unroll
    for (int rep = 0; rep < Params::CUMSUM_H_BLOCK; rep++) {
        const int h = h_base + rep;
        // add prefix of previous warps
        float warp_prefix = (warp_id == 0) ? 0.f : warp_sums[warp_id - 1][rep];
        v[rep] += warp_prefix;

        // write dA_cumsum
        const int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * n_chunks + chunk_id) * Params::CHUNK_SIZE + tid;
        if (tid < this_chunk_len) {
            dA_cumsum[out_idx] = v[rep];
        } else if (tid < Params::CHUNK_SIZE) {
            dA_cumsum[out_idx] = 0.f;
        }
    }
}


// ---------- public entry ----------
std::tuple<at::Tensor, at::Tensor> mamba2_fused_ssd_combined_fwd(
    const at::Tensor& x,          // [B, S, H, D] (unused in step1, we use it to infer B,S,H)
    const at::Tensor& dt,         // [B, S, H]   half
    const at::Tensor& A,          // [H]         float
    const at::Tensor& B,          // (unused here)
    const at::Tensor& C,          // (unused here)
    int64_t            chunk_size,
    const at::Tensor& D,          // (unused here)
    c10::optional<at::Tensor> z,  // (unused here)
    const at::Tensor& dt_bias,    // [H]         half
    c10::optional<at::Tensor> initial_states, // (unused here)
    c10::optional<at::Tensor> seq_idx,        // (unused here)
    c10::optional<at::Tensor> cu_seqlens,     // (unused here)
    bool dt_softplus
) {
    // --- device / dtype sanity ---
    auto dev = x.device();
    TORCH_CHECK(x.is_cuda(),        "x must be CUDA");
    TORCH_CHECK(dt.is_cuda(),       "dt must be CUDA");
    TORCH_CHECK(A.is_cuda(),        "A must be CUDA");
    TORCH_CHECK(dt_bias.is_cuda(),  "dt_bias must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be float16 (Half)");
    TORCH_CHECK(dt.scalar_type() == at::kHalf, "dt must be float16 (Half)");
    TORCH_CHECK(A.scalar_type()  == at::kFloat, "A must be float32");
    TORCH_CHECK(dt_bias.scalar_type() == at::kHalf, "dt_bias must be float16 (Half)");

    // --- infer shapes ---
    TORCH_CHECK(x.dim() >= 3, "x must have shape [B, S, H, ...]");
    const int Bsz = static_cast<int>(x.size(0));
    const int S   = static_cast<int>(x.size(1));
    const int H   = static_cast<int>(x.size(2));
    TORCH_CHECK(dt.sizes() == at::IntArrayRef({Bsz, S, H}), "dt must be [B,S,H]");
    TORCH_CHECK(A.sizes()  == at::IntArrayRef({H}),         "A must be [H]");
    TORCH_CHECK(dt_bias.sizes() == at::IntArrayRef({H}),    "dt_bias must be [H]");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be > 0");

    // TODO: compile multiple versions, for different config
    assert(chunk_size == 128);
    int chunk_size_int = static_cast<int>(chunk_size);

    // NOTE: we do not check for ptr overlap, since it would be crazy to overlap stuff like x and A

    const int n_chunks = DIV_UP(S, chunk_size_int);

    // --- allocate outputs for step 1 (fp32) ---
    auto opts_f = at::TensorOptions().dtype(at::kFloat).device(dev);
    auto dt_out    = at::empty({Bsz, H, n_chunks, chunk_size_int}, opts_f);
    auto dA_cumsum = at::empty_like(dt_out);

    // --- launch kernel (correctness-first mapping: 1 head per block) ---
    using Params = ssd_template_params</*CHUNK_SIZE*/128, /*THREADS*/128, /*TARGET_BLOCKS*/4, /*CUMSUM_H_BLOCK*/4>;
    dim3 block(Params::THREADS);
    dim3 grid(n_chunks, Bsz, DIV_UP(H, Params::CUMSUM_H_BLOCK));
    auto stream = at::cuda::getCurrentCUDAStream();

    mamba2_ssd_step1_dt_transform_cumsum_kernel<Params><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(dt.data_ptr<at::Half>()),
        A.data_ptr<float>(),
        reinterpret_cast<const __half*>(dt_bias.data_ptr<at::Half>()),
        Bsz, S, H,
        dt_softplus,
        dt_out.data_ptr<float>(),
        dA_cumsum.data_ptr<float>()
    );

#ifndef NDEBUG
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA launch failed (step1): ", cudaGetErrorString(err));
#endif

    // TODO: when we wire the rest of the fused pipeline, return a tuple:
    // (out, out_x, dt_out, dA_cumsum, states, final_states)
    // For now, keep API unchanged and return x (contiguous) to not break callers.
    // return x.contiguous();
    // Return exactly (dA_cumsum, dt_out) to match your Python expectation
    return std::make_tuple(dA_cumsum, dt_out);
}
