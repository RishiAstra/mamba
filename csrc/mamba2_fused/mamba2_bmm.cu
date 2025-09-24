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

// Copy a BM/BN x BK tile (row-major line in smem) using 16B vectors where possible.
// Template params are compile-time tile geometry.
// Pass ROWS = Param::BM for A, ROWS = Param::BN for B.
template <typename T, int ROWS, int COLS>
__device__ __forceinline__ void stage_tile_vec16(
    const T* __restrict__ g_tile0,   // global ptr at (row0, k0) of this BLOCK tile
    int64_t g_stride_row,            // global stride (elements) to next row (saS/sbS)
    int64_t g_stride_k,              // global stride (elements) along K (saK/sbK; usually 1)
    T* __restrict__ s_tile0,         // shared ptr at (row0, k0) for this BLOCK tile
    int      s_ld,                   // shared leading dim (elements): A->BK, B(col-major)->thisBK_round
    int      rows_active,            // min(ROWS, chunk_limit - row0)
    int      K_rem,                  // min(COLS, K - k0)  (elements remaining from k0)
    int      warp_id,                // 0..(warps_per_block-1)
    int      warps_per_block         // blockDim.x / 32
){
    constexpr int VBYTES = 4;
    constexpr int VE     = VBYTES / sizeof(T);           // 1 for fp32, 2 for fp16/bf16
    constexpr int WARP_ELEMS = VE * 32;                  // elements per full warp vector copy
    constexpr int ROW_VECS   = DIV_UP(COLS, WARP_ELEMS); // how many 16B vectors per row (full warp)
    constexpr int TOTAL_VECS = ROWS * ROW_VECS; // TODO: don't idle threads if vector is 2x or more than BK

    const int lane_id = threadIdx.x & 31;

    #pragma unroll
    for (int v = warp_id; v < TOTAL_VECS; v += warps_per_block) {
        const int r      = v / ROW_VECS;                // 0..ROWS-1 (virtual row)
        const int c_vec  = v - r * ROW_VECS;            // 0..ROW_VECS-1
        const int k_off  = c_vec * WARP_ELEMS;          // element offset along K

        T*       s = s_tile0 + r * s_ld + k_off;        // row-major line in smem
        const T* g = g_tile0 + r * g_stride_row + k_off * g_stride_k;

        const bool row_ok  = (r < rows_active);
        const bool k_fits  = (k_off + (WARP_ELEMS - 1) < K_rem);
        const bool contigK  = (g_stride_k == 1);
        const bool aligned = (((uintptr_t)g & 0x3) == 0) && (((uintptr_t)s & 0x3) == 0); // 4B align (for int32/float)

        if (row_ok && k_fits && contigK && aligned) {
            // 4B per lane -> 128B per warp
            reinterpret_cast<int32_t*>(s)[lane_id] = reinterpret_cast<const int32_t*>(g)[lane_id];
        } else {
            // slow path: elementwise for this 16B chunk (handles row tail, K tail, misalign)
            #pragma unroll
            for (int i = lane_id; i < WARP_ELEMS; i += 32) {
                const int kk = k_off + i;
                if (kk < COLS) {
                    T val = (row_ok && kk < K_rem) ? g[i * g_stride_k] : from_float<T>(0.0f);
                    s[i] = val; // contiguous along K in smem
                }
            }
        }
    }
}


// TODO: make sure it works even with K not multiple of BK or WMMA_K; BK not multiple of WMMA_K
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
    // TODO: align 16B
    TA* As = reinterpret_cast<TA*>(smem_raw);                           // [BM x BK] row-major
    TB* Bs = reinterpret_cast<TB*>(As + (size_t)BM * BK);               // [BK x BN] col-major (similar to [BN x BK] row-major)
    float* Cs = reinterpret_cast<float*>(Bs + (size_t)BK * BN);         // [BM x BN] row-major

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

    // we need to iterate over K in chunks of BK (otherwise not enough shared memory)
    for (int k0 = 0; k0 < K; k0 += BK) {
        // for now, load the whole BK up front
        const int rowsA    = max(0, min(BM, chunk_limit - m0));
        const int colsB    = max(0, min(BN, chunk_limit - n0));
        const int K_rem    = max(0, min(BK, K - k0));

        const TA* g_tileA0 = A_base + (int64_t)m0 * saS + (int64_t)k0 * saK;
        const TB* g_tileB0 = B_base + (int64_t)n0 * sbS + (int64_t)k0 * sbK;

        TA* s_tileA0 = As;               // row-major, ld = BK
        TB* s_tileB0 = Bs;               // row-major buffer, ld = BK

        // Stage A [BM x BK] into shared as row-major (ldA = BK)
        stage_tile_vec16<TA, BM, BK>(
            g_tileA0, saS, saK,
            s_tileA0, BK,
            rowsA, K_rem,
            warp_id, warps_per_block
        );

        // Stage B [BK x BN] into shared as col-major (ldB = BK)
        stage_tile_vec16<TB, BN, BK>(
            g_tileB0, sbS, sbK,
            s_tileB0, BK,
            colsB, K_rem,
            warp_id, warps_per_block
        );
        __syncthreads();


        // in reading order, each warp processes a contiguous set of subtiles (encouraging reuse along N in C, meaning less smem loads of B)
        // for now, always load, and very simple virtual warp loop

        // compute
        for (int v_warp = warp_id; v_warp < num_subtiles; v_warp += warps_per_block) {
            const int tile_m = v_warp / NT;   // 0..MT-1
            const int tile_n = v_warp % NT;   // 0..NT-1

            // Accumulator fragment for this subtile
            float* c_smem = Cs + (tile_m * 16) * BN + (tile_n * 16); // row-major with ld = BN
            wmma::fragment<wmma::accumulator, 16, 16, WMMA_K, float> c_frag;
            if (k0 == 0) wmma::fill_fragment(c_frag, 0.0f);
            else wmma::load_matrix_sync(c_frag, c_smem, BN, wmma::mem_row_major);

            // Walk K in slabs of WMMA_K
            for (int kk = 0; kk < BK; kk += WMMA_K) {
                // Load A and B fragments from shared memory
                wmma::fragment<wmma::matrix_a, 16, 16, WMMA_K, ATag, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, WMMA_K, BTag, wmma::col_major> b_frag;

                const TA* tileA_ptr = As + (tile_m * 16) * BK + kk; // row-major with ld = BK
                const TB* tileB_ptr = Bs + (tile_n * 16) * BK + kk; // col-major with ld = BK

                wmma::load_matrix_sync(a_frag, tileA_ptr, BK);
                wmma::load_matrix_sync(b_frag, tileB_ptr, BK);

                // Perform the matrix multiplication
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            // accumulate in smem C using wmma
            wmma::store_matrix_sync(c_smem, c_frag, BN, wmma::mem_row_major);

        } // end subtile loop

        __syncthreads();
    }

    // flush smem C to global as single 2d memcpy
    // constexpr int VBYTES = 16;
    constexpr int VBYTES = 4;
    constexpr int VE = VBYTES / sizeof(TO); // 1 for fp32, 2 for fp16/bf16
    constexpr int WARP_ELEMS = VE * 32;        // elements per full warp vector copy
    constexpr int ROW_VECS = DIV_UP(BN, WARP_ELEMS); // how many 16B vectors per row (full warp)
    const int tile_rows = max(0, min(BM, chunk_limit - m0));
    const int tile_cols = max(0, min(BN, chunk_limit - n0));

    #pragma unroll
    for (int v_warp = warp_id; v_warp < BM * ROW_VECS; v_warp += warps_per_block) {
        const int r      = v_warp / ROW_VECS;                // 0..BM-1 (virtual row)
        const int c_vec  = v_warp - r * ROW_VECS;            // 0..ROW_VECS-1
        const int n_off  = c_vec * WARP_ELEMS;                            // element offset along N

        float*    s_out = Cs + r * BN + n_off;                    // row-major smem C
        TO*       g_out = O_base + (int64_t)(m0 + r) * soM + (int64_t)(n0 + n_off) * soN;

        const bool row_ok = (r < tile_rows);
        const bool n_fits = (n_off + (WARP_ELEMS - 1) < tile_cols);
        // TODO: check if correct alignment for half/bfloat16
        const bool can_vec  = (soN == 1) && ((((uintptr_t)s_out & 0x7) == 0) && (((uintptr_t)g_out & 0x3) == 0)); // 4B align (for int32/float)
        const int  m_abs    = m0 + r;
        const bool row_in_chunk = (m_abs < chunk_size);

        if (row_ok && row_in_chunk && n_fits && can_vec) {
            // 128B per warp store, type-aware
            if constexpr (std::is_same_v<TO, float>) {
                reinterpret_cast<float*>(g_out)[lane_id] = s_out[lane_id];
            } else if constexpr (std::is_same_v<TO, __half>) {
                float2 f2 = reinterpret_cast<float2*>(s_out)[lane_id];
                half2  h2 = make_half2(__float2half(f2.x), __float2half(f2.y));
                reinterpret_cast<half2*>(g_out)[lane_id] = h2;
            } else if constexpr (std::is_same_v<TO, __nv_bfloat16>) {
                float2 f2 = reinterpret_cast<float2*>(s_out)[lane_id];
                __nv_bfloat162 b2 = __floats2bfloat162_rn(f2.x, f2.y);
                reinterpret_cast<__nv_bfloat162*>(g_out)[lane_id] = b2;
            } else {
                // Fallback: do scalar path below for unknown TO
                #pragma unroll
                for (int i = lane_id; i < WARP_ELEMS; i += 32) {
                    const int nn = n_off + i;
                    if (nn >= tile_cols) break;
                    g_out[i * soN] = from_float<TO>(s_out[i]);
                }
            }
        } else {
            #pragma unroll
            for (int i = lane_id; i < WARP_ELEMS; i += 32) {
                const int nn    = n_off + i;
                const int n_abs = n0 + nn;
                if (!row_in_chunk || n_abs >= chunk_size) break;

                float v = 0.0f;
                if (r < tile_rows && nn < tile_cols) {
                    v = s_out[i];
                }
                g_out[i * soN] = from_float<TO>(v);
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

    // Shared memory size: As + Bs + Cs
    size_t shmem = (size_t)Param::BM * Param::BK * sizeof(TA)
                 + (size_t)Param::BK * Param::BN * sizeof(TB)
                 + (size_t)Param::BM * Param::BN * sizeof(float);


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

    // using P = bmm_wmma_params</*BM*/32, /*BN*/32, /*BK*/64, /*THREADS*/128, /*TARGET_BLOCKS*/8>;
    // using P = bmm_wmma_params</*BM*/32, /*BN*/32, /*BK*/64, /*THREADS*/512, /*TARGET_BLOCKS*/4>;
    using P = bmm_wmma_params</*BM*/64, /*BN*/64, /*BK*/64, /*THREADS*/512, /*TARGET_BLOCKS*/4>;
    // using P = bmm_wmma_params</*BM*/64, /*BN*/64, /*BK*/64, /*THREADS*/256, /*TARGET_BLOCKS*/4>;
    // using P = bmm_wmma_params</*BM*/64, /*BN*/64, /*BK*/32, /*THREADS*/256, /*TARGET_BLOCKS*/4>;
    // // Default WMMA config (good general start on SM80+). Keep multiples of 16.
    // // using P = bmm_wmma_params</*BM*/128, /*BN*/64, /*BK*/64, /*THREADS*/256, /*TARGET_BLOCKS*/4>;
    // using P = bmm_wmma_params</*BM*/128, /*BN*/128, /*BK*/32, /*THREADS*/256, /*TARGET_BLOCKS*/4>;
    // // using P = bmm_wmma_params</*BM*/64, /*BN*/64, /*BK*/32, /*THREADS*/64, /*TARGET_BLOCKS*/8>; // close to triton config
    // // using P = bmm_wmma_params</*BM*/128, /*BN*/128, /*BK*/64, /*THREADS*/256, /*TARGET_BLOCKS*/4>;
    // // using P = bmm_wmma_params</*BM*/16, /*BN*/16, /*BK*/16, /*THREADS*/32, /*TARGET_BLOCKS*/1>;

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
