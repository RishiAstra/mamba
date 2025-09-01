// mamba2_fused_cuda.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <cuda_fp16.h>
#include <tuple>

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

// Compile-time tuning bag (we'll keep CHUNK_SIZE unused for now; runtime param is passed)
template <uint32_t CHUNK_SIZE_, uint32_t THREADS_, uint32_t TARGET_BLOCKS_>
struct ssd_template_params {
    static constexpr uint32_t CHUNK_SIZE     = CHUNK_SIZE_;
    static constexpr uint32_t THREADS        = THREADS_;
    static constexpr uint32_t TARGET_BLOCKS  = TARGET_BLOCKS_;
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
    int chunk_size,
    bool dt_softplus,
    float* __restrict__ dt_out,        // [B, H, n_chunks, chunk_size] fp32
    float* __restrict__ dA_cumsum      // same shape
) {
    const int b = blockIdx.y;
    const int chunk_id = blockIdx.x;
    const int h = blockIdx.z;

    if (b >= B || h >= H) return;

    const int s_start = chunk_id * chunk_size;
    if (s_start >= S) return;
    const int this_chunk_len = min(chunk_size, S - s_start);

    const int n_chunks = DIV_UP(S, chunk_size);

    // gather scalars per head
    const float A_h = A[h];
    const float bias_h = __half2float(dt_bias[h]);

    // TODO: support chunk size != block size
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    constexpr int WARPS = Params::THREADS / 32;
    __shared__ float warp_sums[WARPS];

    // compute dt_val (masked for tail) and write dt_out
    float dt_val = 0.f;
    if (tid < this_chunk_len) {
        const int s = s_start + tid;
        const int64_t dt_idx = ((static_cast<int64_t>(b) * S + s) * H + h);
        dt_val = __half2float(dt[dt_idx]) + bias_h;
        if (dt_softplus) dt_val = softplusf(dt_val);
        // store dt_out[b,h,chunk,tid]
        const int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * n_chunks + chunk_id) * chunk_size + tid;
        dt_out[out_idx] = dt_val;
    } else {
        // ensure padded tail is deterministic zero
        const int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * n_chunks + chunk_id) * chunk_size + tid;
        if (tid < chunk_size) dt_out[out_idx] = 0.f;
    }

    // local value to scan
    float v = (tid < this_chunk_len) ? (dt_val * A_h) : 0.f;

    // intra-warp inclusive scan
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(mask, v, offset);
        if (lane >= offset) v += n;
    }

    // TODO: fix for chunk_size != block_size
    // compute warp total robustly (handle partial last warp)
    const int warp_base = warp_id * 32;
    int local_count = this_chunk_len - warp_base;
    if (local_count < 0) local_count = 0;
    if (local_count > 32) local_count = 32;
    float warp_total = (local_count > 0) ? __shfl_sync(mask, v, local_count - 1) : 0.f;

    // one thread per warp stores total
    if (lane == 0) warp_sums[warp_id] = warp_total;
    __syncthreads();

    // small serial prefix over warps (W <= 8 here)
    if (threadIdx.x == 0) {
        float accw = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            float t = warp_sums[w];
            accw += t;
            warp_sums[w] = accw;  // inclusive
        }
    }
    __syncthreads();

    // add prefix of previous warps
    float warp_prefix = (warp_id == 0) ? 0.f : warp_sums[warp_id - 1];
    v += warp_prefix;

    // write dA_cumsum
    const int64_t out_idx = ((static_cast<int64_t>(b) * H + h) * n_chunks + chunk_id) * chunk_size + tid;
    if (tid < this_chunk_len) {
        dA_cumsum[out_idx] = v;
    } else if (tid < chunk_size) {
        dA_cumsum[out_idx] = 0.f;
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

    const int n_chunks = DIV_UP(S, static_cast<int>(chunk_size));

    // --- allocate outputs for step 1 (fp32) ---
    auto opts_f = at::TensorOptions().dtype(at::kFloat).device(dev);
    auto dt_out    = at::empty({Bsz, H, n_chunks, static_cast<int>(chunk_size)}, opts_f);
    auto dA_cumsum = at::empty_like(dt_out);

    // --- launch kernel (correctness-first mapping: 1 head per block) ---
    using Params = ssd_template_params</*CHUNK_SIZE*/0, /*THREADS*/128, /*TARGET_BLOCKS*/4>;
    dim3 block(Params::THREADS);
    dim3 grid(n_chunks, Bsz, H);
    auto stream = at::cuda::getCurrentCUDAStream();

    mamba2_ssd_step1_dt_transform_cumsum_kernel<Params><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(dt.data_ptr<at::Half>()),
        A.data_ptr<float>(),
        reinterpret_cast<const __half*>(dt_bias.data_ptr<at::Half>()),
        Bsz, S, H,
        static_cast<int>(chunk_size),
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
