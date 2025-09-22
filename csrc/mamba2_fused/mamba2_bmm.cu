// mamba2_bmm.cu
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <c10/util/Optional.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <type_traits>
#include <tuple>

#include "common.h"  // DIV_UP, etc.

using namespace nvcuda;

// ---------------- template launch params (multiples of 16) ----------------
template <int BM_, int BN_, int BK_, int THREADS_, int TARGET_BLOCKS_>
struct bmm_wmma_params {
    static_assert(BM_ % 16 == 0 && BN_ % 16 == 0 && BK_ % 16 == 0, "BM, BN, BK must be multiples of 16");
    static constexpr int BM = BM_;
    static constexpr int BN = BN_;
    static constexpr int BK = BK_;
    static constexpr int THREADS = THREADS_;
    static constexpr int TARGET_BLOCKS = TARGET_BLOCKS_;
    static_assert(THREADS % 32 == 0, "THREADS must be a multiple of 32");
};

// ---------------- dtype helpers ----------------
template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// WMMA element/tag + K tile size
template <typename T> struct wmma_prec;
template <> struct wmma_prec<__half>         { using type = __half;                          static constexpr int WMMA_K = 16; };
template <> struct wmma_prec<__nv_bfloat16>  { using type = __nv_bfloat16;                   static constexpr int WMMA_K = 16; };
template <> struct wmma_prec<float>          { using type = nvcuda::wmma::precision::tf32;   static constexpr int WMMA_K = 8;  };

// ---------------- core kernel (WMMA tensor cores) ----------------
template <typename TA, typename TB, typename TO, typename P>
__global__ __launch_bounds__(P::THREADS, P::TARGET_BLOCKS)
void bmm_chunk_wmma_kernel(
    const TA* __restrict__ A,   // [B, S, (G,) K]
    const TB* __restrict__ B,   // [B, S, (G,) K]
    TO* __restrict__ Out,       // [B, n_chunks, (G,) chunk, chunk]
    const int32_t* __restrict__ seq_idx, // [B, S] or nullptr
    // dims
    int Bsz, int S, int chunk_size, int K, int ngroups,
    // strides for A
    int64_t saB, int64_t saS, int64_t saH, int64_t saK,
    // strides for B
    int64_t sbB, int64_t sbS, int64_t sbH, int64_t sbK,
    // strides for Out
    int64_t soB, int64_t soC, int64_t soH, int64_t soM, int64_t soN,
    // strides for seq_idx (may be zeros if null)
    int64_t ssB, int64_t ssS,
    // flags
    bool is_causal
) {
#if __CUDA_ARCH__ < 800
    return; // require SM80+
#endif
    constexpr int BM = P::BM;
    constexpr int BN = P::BN;
    constexpr int BK = P::BK;

    const int n_chunks = DIV_UP(S, chunk_size);

    // grid mapping
    const int batch = blockIdx.y;
    const int cg = blockIdx.z; // chunk-group merged
    const int chunk_id = (ngroups > 1) ? (cg / ngroups) : cg;
    const int group_id = (ngroups > 1) ? (cg % ngroups) : 0;

    if (batch >= Bsz || chunk_id >= n_chunks) return;

    const int m_tiles = DIV_UP(chunk_size, BM);
    const int n_tiles = DIV_UP(chunk_size, BN);

    const int tile_id = blockIdx.x;
    const int pid_m = tile_id / n_tiles;
    const int pid_n = tile_id - pid_m * n_tiles;
    if (pid_m >= m_tiles || pid_n >= n_tiles) return;

    // causal tile-skip
    if (is_causal) {
        if (pid_n * BN >= (pid_m + 1) * BM) return;
    }

    const int s_start = chunk_id * chunk_size;
    const int chunk_limit = min(chunk_size, S - s_start);

    // Base pointers
    const TA* A_base = A + batch * saB + (int64_t)s_start * saS + (int64_t)group_id * saH;
    const TB* B_base = B + batch * sbB + (int64_t)s_start * sbS + (int64_t)group_id * sbH;
    TO* O_base = Out + batch * soB + (int64_t)chunk_id * soC + (int64_t)group_id * soH;

    const int32_t* seq_base = nullptr;
    if (seq_idx) seq_base = seq_idx + batch * ssB + (int64_t)s_start * ssS;

    // This block computes a BMxBN tile at (m0.., n0..)
    const int m0 = pid_m * BM;
    const int n0 = pid_n * BN;

    // Shared memory: A [BM x BK], B [BK x BN], and per-warp C buffer [16x16]
    extern __shared__ __align__(16) unsigned char smem_raw[];
    TA* As = reinterpret_cast<TA*>(smem_raw);                          // [BM x BK] row-major
    TB* Bs = reinterpret_cast<TB*>(As + (size_t)BM * BK);               // [BK x BN] row-major buffer
    float* Cs = reinterpret_cast<float*>(Bs + (size_t)BK * BN);         // [warps_per_block x 16 x 16]

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warps_per_block = P::THREADS / 32;

    // 16x16 sub-tiles inside the BMxBN tile
    constexpr int MT = BM / 16;
    constexpr int NT = BN / 16;
    const int num_subtiles = MT * NT;

    using ATag = typename wmma_prec<TA>::type;
    using BTag = typename wmma_prec<TB>::type;
    static_assert(std::is_same_v<TA, TB>, "TA/TB must match for this WMMA path");
    constexpr int WMMA_K = wmma_prec<TA>::WMMA_K;

    // Each warp processes its set of subtiles and accumulates across all K-slabs
    for (int st = warp_id; st < num_subtiles; st += warps_per_block) {
        const int tile_m = st / NT;   // 0..MT-1
        const int tile_n = st % NT;   // 0..NT-1

        // Accumulator fragment for this subtile
        wmma::fragment<wmma::accumulator, 16, 16, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Walk K in slabs of BK
        for (int k0 = 0; k0 < K; k0 += BK) {
            const int thisBK = min(BK, K - k0);
            const int thisBK_round = ((thisBK + WMMA_K - 1) / WMMA_K) * WMMA_K; // pad to WMMA_K

            // ---- Stage A [BM x thisBK_round] into shared ----
            const int a_elems = BM * thisBK_round;
            for (int idx = threadIdx.x; idx < a_elems; idx += blockDim.x) {
                const int r = idx / thisBK_round;  // [0..BM)
                const int c = idx % thisBK_round;  // [0..thisBK_round)
                const int m = m0 + r;
                const int k = k0 + c;
                TA val = from_float<TA>(0.f);
                if (m < chunk_limit && k < K) {
                    val = A_base[(int64_t)m * saS + (int64_t)k * saK];
                }
                As[r * BK + c] = val; // row-major with stride BK
            }

            // ---- Stage B [thisBK_round x BN] into shared ----
            // const int b_elems = thisBK_round * BN;
            // for (int idx = threadIdx.x; idx < b_elems; idx += blockDim.x) {
            //     const int r = idx / BN;  // [0..thisBK_round)
            //     const int c = idx % BN;  // [0..BN)
            //     const int n = n0 + c;
            //     const int k = k0 + r;
            //     TB val = from_float<TB>(0.f);
            //     if (n < chunk_limit && k < K) {
            //         val = B_base[(int64_t)k * sbK + (int64_t)n * sbS];
            //     }
            //     Bs[r * BN + c] = val; // row-major buffer; load as col_major with ld=BN
            // }
            const int b_elems = thisBK_round * BN;
            for (int idx = threadIdx.x; idx < b_elems; idx += blockDim.x) {
                const int c = idx / thisBK_round;  // column within BN   (n-subtile column)
                const int r = idx % thisBK_round;  // row within K slab  (kk within K)
                const int n = n0 + c;
                const int k = k0 + r;
                TB val = from_float<TB>(0.f);
                if (n < chunk_limit && k < K) {
                    val = B_base[(int64_t)k * sbK + (int64_t)n * sbS];
                }
                // col-major store: element (r, c) -> c*ld + r, with ld = thisBK_round
                Bs[c * thisBK_round + r] = val; // TODO: should load static number of elements with static layout, but pad 0s
            }

            __syncthreads();

            // ---- MMA over this slab (step by WMMA_K) ----
            for (int kk = 0; kk < thisBK_round; kk += WMMA_K) {
                const TA* a_tile_ptr = &As[(tile_m * 16) * BK + kk];
                // const TB* b_tile_ptr = &Bs[kk * BN + (tile_n * 16)];
                const TB* b_tile_ptr = &Bs[(tile_n * 16) * thisBK_round + kk];

                wmma::fragment<wmma::matrix_a, 16, 16, WMMA_K, ATag, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, WMMA_K, BTag, wmma::col_major> b_frag;
                // wmma::fragment<wmma::matrix_b, 16, 16, WMMA_K, BTag, wmma::row_major> b_frag;

                wmma::load_matrix_sync(a_frag, a_tile_ptr, BK);
                // wmma::load_matrix_sync(b_frag, b_tile_ptr, BN);
                wmma::load_matrix_sync(b_frag, b_tile_ptr, thisBK_round);

                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            __syncthreads(); // allow next slab staging
        }

        // ---- Store accumulator to shared (per-warp), then masked write to global ----
        float* Cw = Cs + warp_id * 16 * 16;
        wmma::store_matrix_sync(Cw, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        const int m_base = m0 + tile_m * 16;
        const int n_base = n0 + tile_n * 16;

        for (int t = lane_id; t < 16 * 16; t += 32) {
            const int dm = t / 16;
            const int dn = t % 16;
            const int m = m_base + dm;
            const int n = n_base + dn;

            bool valid = (m < chunk_size) && (n < chunk_size) &&
                         (m < chunk_limit) && (n < chunk_limit);
            if (is_causal && n > m) valid = false;

            if (valid && seq_base) {
                const int32_t sm = seq_base[m * ssS];
                const int32_t sn = seq_base[n * ssS];
                if (sm != sn) valid = false;
            }

            const float v = valid ? Cw[dm * 16 + dn] : 0.0f;
            if (m < chunk_size && n < chunk_size) {
                const int64_t o_idx = (int64_t)m * soM + (int64_t)n * soN;
                O_base[o_idx] = from_float<TO>(v);
            }
        }
    }
}

template <typename TA, typename TB, typename TO, typename P>
static void launch_bmm_chunk_wmma_typed(
    const at::Tensor& a, const at::Tensor& b, at::Tensor& out,
    const c10::optional<at::Tensor>& seq_idx_opt,
    int Bsz, int S, int chunk_size, int K, int ngroups,
    // strides
    int64_t saB, int64_t saS, int64_t saH, int64_t saK,
    int64_t sbB, int64_t sbS, int64_t sbH, int64_t sbK,
    int64_t soB, int64_t soC, int64_t soH, int64_t soM, int64_t soN,
    int64_t ssB, int64_t ssS,
    bool is_causal
) {
    using Param = P;

    const int n_chunks = DIV_UP(S, chunk_size);
    const int m_tiles = DIV_UP(chunk_size, Param::BM);
    const int n_tiles = DIV_UP(chunk_size, Param::BN);

    dim3 block(Param::THREADS);
    dim3 grid(m_tiles * n_tiles, Bsz, n_chunks * ngroups);

    // Shared memory size: As + Bs + per-warp Cs
    const int warps_per_block = Param::THREADS / 32;
    size_t shmem = (size_t)Param::BM * Param::BK * sizeof(TA) +
                   (size_t)Param::BK * Param::BN * sizeof(TB) +
                   (size_t)warps_per_block * 16 * 16 * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    const int32_t* seq_ptr = nullptr;
    if (seq_idx_opt && seq_idx_opt->defined()) {
        TORCH_CHECK(seq_idx_opt->dtype() == at::kInt, "seq_idx must be int32");
        seq_ptr = seq_idx_opt->data_ptr<int32_t>();
    }

    bmm_chunk_wmma_kernel<TA, TB, TO, Param><<<grid, block, shmem, stream>>>(
        reinterpret_cast<const TA*>(a.const_data_ptr()),
        reinterpret_cast<const TB*>(b.const_data_ptr()),
        reinterpret_cast<TO*>(out.data_ptr()),
        seq_ptr,
        Bsz, S, chunk_size, K, ngroups,
        saB, saS, saH, saK,
        sbB, sbS, sbH, sbK,
        soB, soC, soH, soM, soN,
        ssB, ssS,
        is_causal
    );

#ifndef NDEBUG
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA launch failed (bmm_chunk_wmma): ", cudaGetErrorString(err));
#endif
}

// --------------- public entry (exported) ---------------
// Return (out, None, None, None, None, None)
std::tuple<c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>>
mamba2_bmm_chunk_fwd_cuda(const Mamba2SSDArgs& args) {
    const auto& a = args.C;      // [B, S, (G,) K]
    const auto& b = args.B;      // [B, S, (G,) K]
    auto out      = args.CB;    // [B, C, (G,) chunk, chunk], dtype => output type

    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda() && out.device().is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(a.dim() == b.dim(), "a and b must have same rank (3 or 4)");
    TORCH_CHECK(out.is_contiguous() == false || out.stride(out.dim()-1) == 1, "out must be writable with given strides");

    const int Bsz = args.Bsz;
    const int S   = args.S;
    const int K   = args.dstate;       // K = d_state
    const int chunk_size = args.chunk_size;
    const int ngroups = std::max<int>(1, static_cast<int>(args.ngroups));

    // strides
    const int64_t saB = a.stride(0);
    const int64_t saS = a.stride(1);
    const int64_t saH = (a.dim()==4) ? a.stride(2) : 0;
    const int64_t saK = a.stride(-1);

    const int64_t sbB = b.stride(0);
    const int64_t sbS = b.stride(1);
    const int64_t sbH = (b.dim()==4) ? b.stride(2) : 0;
    const int64_t sbK = b.stride(-1);

    const int64_t soB = out.stride(0);
    const int64_t soC = out.stride(1);
    const int64_t soH = (out.dim()==5) ? out.stride(2) : 0;
    const int64_t soM = out.stride(out.dim()-2);
    const int64_t soN = out.stride(out.dim()-1);

    int64_t ssB = 0, ssS = 0;
    if (args.seq_idx && args.seq_idx->defined()) {
        ssB = args.seq_idx->stride(0);
        ssS = args.seq_idx->stride(1);
    }

    // // dtype routing
    // const auto in_common = [&](){
    //     if (a.scalar_type() == at::kBFloat16 || b.scalar_type() == at::kBFloat16) return at::kBFloat16;
    //     if (a.scalar_type() == at::kHalf     || b.scalar_type() == at::kHalf)     return at::kHalf;
    //     return at::kFloat;
    // }();
    const auto out_type  = out.scalar_type();

    // Default WMMA config (good general start on SM80+). Keep multiples of 16.
    using P = bmm_wmma_params</*BM*/128, /*BN*/128, /*BK*/64, /*THREADS*/256, /*TARGET_BLOCKS*/2>;
    // using P = bmm_wmma_params</*BM*/16, /*BN*/16, /*BK*/16, /*THREADS*/32, /*TARGET_BLOCKS*/1>;

#define LAUNCH(TA, TB, TO) \
    launch_bmm_chunk_wmma_typed<TA, TB, TO, P>( \
        a, b, out, args.seq_idx, \
        Bsz, S, chunk_size, K, ngroups, \
        saB, saS, saH, saK, \
        sbB, sbS, sbH, sbK, \
        soB, soC, soH, soM, soN, \
        ssB, ssS, \
        false /* is_causal */ \
    );

    // TODO: don't hardcode
    LAUNCH(__half, __half, __half);

    // if (in_common == at::kBFloat16) {
    //     if (out_type == at::kFloat)          { LAUNCH(__nv_bfloat16, __nv_bfloat16, float); }
    //     else if (out_type == at::kHalf)      { LAUNCH(__nv_bfloat16, __nv_bfloat16, __half); }
    //     else if (out_type == at::kBFloat16)  { LAUNCH(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16); }
    //     else                                  { TORCH_CHECK(false, "Unsupported out dtype for bf16 in/out"); }
    // } else if (in_common == at::kHalf) {
    //     if (out_type == at::kFloat)          { LAUNCH(__half, __half, float); }
    //     else if (out_type == at::kHalf)      { LAUNCH(__half, __half, __half); }
    //     else if (out_type == at::kBFloat16)  { LAUNCH(__half, __half, __nv_bfloat16); }
    //     else                                  { TORCH_CHECK(false, "Unsupported out dtype for f16 in/out"); }
    // } else { // float inputs — WMMA TF32 path
    //     if (out_type == at::kFloat)          { LAUNCH(float, float, float); }
    //     else if (out_type == at::kHalf)      { LAUNCH(float, float, __half); }
    //     else if (out_type == at::kBFloat16)  { LAUNCH(float, float, __nv_bfloat16); }
    //     else                                  { TORCH_CHECK(false, "Unsupported out dtype for f32 in/out"); }
    // }
#undef LAUNCH

    return std::make_tuple(out, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
}
