// mamba2_args.h
#pragma once
#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))
#define ASYNC_GPU_MEM (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
// #define DOUBLE_BUFF_BMM // actually reduces performance due to higher shared memory usage causing smaller blocks

struct Mamba2SSDArgs {
    // Required tensors (match get_rand_input)
    const at::Tensor x;       // [B, S, H, headdim] (Half)
    const at::Tensor dt;      // [B, S, H]          (Half)
    const at::Tensor A;       // [H]                (Float)
    const at::Tensor B;       // [B, S, ngroups, dstate] (Half)
    const at::Tensor C;       // [B, S, ngroups, dstate] (Half)
    const at::Tensor D;       // [H]                (Float)

    // Scalars (must be specified)
    const bool    dt_softplus;
    const int64_t chunk_size;

    // control CB dtype (false = 16-bit; true = fp32)
    const bool    cb_force_fp32 = false;

    // Optional tensors
    const c10::optional<at::Tensor> z;              // [B, S, H, headdim] (Half)
    const c10::optional<at::Tensor> dt_bias;        // [H]                (Half)
    const c10::optional<at::Tensor> initial_states; // [B, H, headdim, dstate] (Float)
    const c10::optional<at::Tensor> seq_idx;        // [B] (int64)
    const c10::optional<at::Tensor> cu_seqlens;     // [B+1] (int64)

    // Derived dims
    int64_t Bsz = 0;       // batch
    int64_t S    = 0;      // seqlen
    int64_t H    = 0;      // heads
    int64_t headdim = 0;
    int64_t ngroups = 0;
    int64_t dstate  = 0;
    int64_t n_chunks = 0;

    // outputs
    at::Tensor out;
    at::Tensor out_x;
    at::Tensor dt_out;
    at::Tensor dA_cumsum;
    at::Tensor states;
    at::Tensor final_states;
    // intermediate
    at::Tensor CB;

    // Explicit constructor with default optionals
    Mamba2SSDArgs(
        at::Tensor x,
        at::Tensor dt,
        at::Tensor A,
        at::Tensor B,
        at::Tensor C,
        at::Tensor D,
        bool dt_softplus = false,
        int64_t chunk_size = 128,
        bool cb_force_fp32 = false,
        // optionals
        c10::optional<at::Tensor> z = c10::nullopt,
        c10::optional<at::Tensor> dt_bias = c10::nullopt,
        c10::optional<at::Tensor> initial_states = c10::nullopt,
        c10::optional<at::Tensor> seq_idx = c10::nullopt,
        c10::optional<at::Tensor> cu_seqlens = c10::nullopt,
        // optional preallocated outputs
        c10::optional<at::Tensor> out = c10::nullopt,
        c10::optional<at::Tensor> out_x = c10::nullopt,
        c10::optional<at::Tensor> dt_out = c10::nullopt,
        c10::optional<at::Tensor> dA_cumsum = c10::nullopt,
        c10::optional<at::Tensor> states = c10::nullopt,
        c10::optional<at::Tensor> final_states = c10::nullopt,
        // optional preallocated intermediate
        c10::optional<at::Tensor> CB = c10::nullopt
    )
    : x(std::move(x))
    , dt(std::move(dt))
    , A(std::move(A))
    , B(std::move(B))
    , C(std::move(C))
    , D(std::move(D))
    , dt_softplus(dt_softplus)
    , chunk_size(chunk_size)
    , cb_force_fp32(cb_force_fp32)
    , z(std::move(z))
    , dt_bias(std::move(dt_bias))
    , initial_states(std::move(initial_states))
    , seq_idx(std::move(seq_idx))
    , cu_seqlens(std::move(cu_seqlens))
    {
        verify_and_init(
            std::move(out),
            std::move(out_x),
            std::move(dt_out),
            std::move(dA_cumsum),
            std::move(states),
            std::move(final_states),
            std::move(CB)
        );
    }

    void verify_and_init(
        c10::optional<at::Tensor> out_ = c10::nullopt,
        c10::optional<at::Tensor> out_x_ = c10::nullopt,
        c10::optional<at::Tensor> dt_out_ = c10::nullopt,
        c10::optional<at::Tensor> dA_cumsum_ = c10::nullopt,
        c10::optional<at::Tensor> states_ = c10::nullopt,
        c10::optional<at::Tensor> final_states_ = c10::nullopt,
        c10::optional<at::Tensor> CB_ = c10::nullopt
    ) {
        // ---- defined ----
        TORCH_CHECK(x.defined() && dt.defined() && A.defined() &&
                    B.defined() && C.defined() && D.defined(),
                    "all required tensors (x, dt, A, B, C, D) must be defined");

        // optional tensors
        bool has_z = z && z->defined();
        bool has_dt_bias = dt_bias && dt_bias->defined();
        bool has_initial_states = initial_states && initial_states->defined();
        bool has_seq_idx = seq_idx && seq_idx->defined();
        bool has_cu_seqlens = cu_seqlens && cu_seqlens->defined();

        // ---- dtypes ----
        TORCH_CHECK(x.scalar_type()       == at::kHalf,  "x must be float16");
        TORCH_CHECK(dt.scalar_type()      == at::kHalf,  "dt must be float16");
        TORCH_CHECK(A.scalar_type()       == at::kFloat, "A must be float32");
        TORCH_CHECK(B.scalar_type()       == at::kHalf,  "B must be float16");
        TORCH_CHECK(C.scalar_type()       == at::kHalf,  "C must be float16");
        TORCH_CHECK(D.scalar_type()       == at::kFloat, "D must be float32");

        if (has_z) TORCH_CHECK(z->scalar_type() == at::kHalf, 
                                        "z must be float16 if provided");
        if (has_dt_bias) TORCH_CHECK(dt_bias->scalar_type() == at::kHalf,
                                        "dt_bias must be float16 if provided");
        if (has_initial_states) TORCH_CHECK(initial_states->scalar_type() == at::kFloat,
                                        "initial_states must be float32 if provided");
        if (has_seq_idx) TORCH_CHECK(seq_idx->scalar_type() == at::kLong,
                                        "seq_idx must be int64 if provided");
        if (has_cu_seqlens) TORCH_CHECK(cu_seqlens->scalar_type() == at::kLong,
                                        "cu_seqlens must be int64 if provided");


        // ---- num of dims ----
        TORCH_CHECK(x.dim()       == 4, "x must be 4D: [B, S, H, headdim]");
        TORCH_CHECK(dt.dim()      == 3, "dt must be 3D: [B, S, H]");
        TORCH_CHECK(A.dim()       == 1, "A must be 1D: [H]");
        TORCH_CHECK(D.dim()       == 1, "D must be 1D: [H]");
        TORCH_CHECK(B.dim()       == 4, "B must be 4D: [B, S, ngroups, dstate]");
        TORCH_CHECK(C.dim()       == 4, "C must be 4D: [B, S, ngroups, dstate]");

        if (has_z) TORCH_CHECK(z->dim() == 4,
                                        "z must be 4D: [B, S, H, headdim]");
        if (has_dt_bias) TORCH_CHECK(dt_bias->dim() == 1,
                                        "dt_bias must be 1D: [H]");
        if (has_initial_states) TORCH_CHECK(initial_states->dim() == 4,
                                        "initial_states must be 4D: [B, H, headdim, dstate]");
        if (has_seq_idx) TORCH_CHECK(seq_idx->dim() == 1,
                                        "seq_idx must be 1D: [B]");
        if (has_cu_seqlens) TORCH_CHECK(cu_seqlens->dim() == 1,
                                        "cu_seqlens must be 1D: [B+1]");


        // ---- derive dims ----
        TORCH_CHECK(chunk_size == 128, "only chunk_size=128 is supported for now");

        Bsz      = x.size(0);
        S        = x.size(1);
        H        = x.size(2);
        headdim  = x.size(3);
        ngroups  = B.size(2);
        dstate   = B.size(3);
        n_chunks = DIV_UP(S, chunk_size);

        TORCH_CHECK(H > 0, "H must be > 0");
        TORCH_CHECK(ngroups > 0 && ngroups <= H, "ngroups must be in [1, H]");
        TORCH_CHECK(H % ngroups == 0, "H must be divisible by ngroups");

        // ---- shape compat checks ----
        TORCH_CHECK(x.size(0) == Bsz && x.size(1) == S && x.size(2) == H && x.size(3) == headdim,
                    "x must be [B, S, H, headdim]");
        TORCH_CHECK(dt.size(0) == Bsz && dt.size(1) == S && dt.size(2) == H,
                    "dt must be [B, S, H]");
        TORCH_CHECK(A.size(0) == H && D.size(0) == H,
                    "A and D must be [H]");
        TORCH_CHECK(B.size(0) == Bsz && B.size(1) == S && B.size(2) == ngroups && B.size(3) == dstate,
                    "B must be [B, S, ngroups, dstate]");
        TORCH_CHECK(C.size(0) == Bsz && C.size(1) == S && C.size(2) == ngroups && C.size(3) == dstate,
                    "C must be [B, S, ngroups, dstate]");
        if (has_z) {
            TORCH_CHECK(z->size(0) == Bsz && z->size(1) == S && z->size(2) == H && z->size(3) == headdim,
                        "z must be [B, S, H, headdim] if provided");
        }
        if (has_dt_bias) {
            TORCH_CHECK(dt_bias->size(0) == H,
                        "dt_bias must be [H] if provided");
        }
        if (has_initial_states) {
            TORCH_CHECK(initial_states->size(0) == Bsz && initial_states->size(1) == H &&
                        initial_states->size(2) == headdim && initial_states->size(3) == dstate,
                        "initial_states must be [B, H, headdim, dstate] if provided");
        }
        if (has_seq_idx) {
            TORCH_CHECK(seq_idx->size(0) == Bsz,
                        "seq_idx must be [B] if provided");
        }
        if (has_cu_seqlens) {
            TORCH_CHECK(cu_seqlens->size(0) == Bsz + 1,
                        "cu_seqlens must be [B+1] if provided");
        }


        // ---- must be contiguous ----
        TORCH_CHECK(x.is_contiguous() , "x must be contiguous");
        TORCH_CHECK(dt.is_contiguous(), "dt must be contiguous");
        TORCH_CHECK(A.is_contiguous() , "A must be contiguous");
        TORCH_CHECK(B.is_contiguous() , "B must be contiguous");
        TORCH_CHECK(C.is_contiguous() , "C must be contiguous");
        TORCH_CHECK(D.is_contiguous() , "D must be contiguous");

        if (has_z) TORCH_CHECK(z->is_contiguous(),
                    "z must be contiguous if provided");
        if (has_dt_bias) TORCH_CHECK(dt_bias->is_contiguous(),
                    "dt_bias must be contiguous if provided");
        if (has_initial_states) TORCH_CHECK(initial_states->is_contiguous(),
                    "initial_states must be contiguous if provided");
        if (has_seq_idx) TORCH_CHECK(seq_idx->is_contiguous(),
                    "seq_idx must be contiguous if provided");
        if (has_cu_seqlens) TORCH_CHECK(cu_seqlens->is_contiguous(),
                    "cu_seqlens must be contiguous if provided");

        // ---- temp ----
        TORCH_CHECK(has_dt_bias,
                    "dt_bias is required for the current implementation");


        // ---- device ----
        auto dev  = x.device();
        TORCH_CHECK(dt.device() == dev, "dt must be on the same device as x");
        TORCH_CHECK(A.device()  == dev, "A must be on the same device as x");
        TORCH_CHECK(B.device()  == dev, "B must be on the same device as x");
        TORCH_CHECK(C.device()  == dev, "C must be on the same device as x");
        TORCH_CHECK(D.device()  == dev, "D must be on the same device as x");

        if (has_z) TORCH_CHECK(z->device() == dev,
                    "z must be on the same device as x if provided");
        if (has_dt_bias) TORCH_CHECK(dt_bias->device() == dev,
                    "dt_bias must be on the same device as x if provided");
        if (has_initial_states) TORCH_CHECK(initial_states->device() == dev,
                    "initial_states must be on the same device as x if provided");
        if (has_seq_idx) TORCH_CHECK(seq_idx->device() == dev,
                    "seq_idx must be on the same device as x if provided");
        if (has_cu_seqlens) TORCH_CHECK(cu_seqlens->device() == dev,
                    "cu_seqlens must be on the same device as x if provided");

        auto opts_f32 = at::TensorOptions().dtype(at::kFloat).device(dev);

        // ---- alloc outputs if not provided ----
        if (!out_ || !out_->defined()) {
            out = at::empty({Bsz, S, H, headdim}, opts_f32);
        } else {
            out = std::move(*out_);
            TORCH_CHECK(out.device() == dev, "out device mismatch");
            TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
            TORCH_CHECK(out.dim() == 4 && out.sizes().equals(at::IntArrayRef{Bsz, S, H, headdim}),
                        "out must be [B, S, H, headdim]");
            TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
        }

        if (!out_x_ || !out_x_->defined()) {
            out_x = at::empty({Bsz, S, H, headdim}, opts_f32);
        } else {
            out_x = std::move(*out_x_);
            TORCH_CHECK(out_x.device() == dev, "out_x device mismatch");
            TORCH_CHECK(out_x.scalar_type() == at::kFloat, "out_x must be float32");
            TORCH_CHECK(out_x.dim() == 4 && out_x.sizes().equals(at::IntArrayRef{Bsz, S, H, headdim}),
                        "out_x must be [B, S, H, headdim]");
            TORCH_CHECK(out_x.is_contiguous(), "out_x must be contiguous");
        }

        if (!dt_out_ || !dt_out_->defined()) {
            dt_out = at::empty({Bsz, H, n_chunks, chunk_size}, opts_f32);
        } else {
            dt_out = std::move(*dt_out_);
            TORCH_CHECK(dt_out.device() == dev, "dt_out device mismatch");
            TORCH_CHECK(dt_out.scalar_type() == at::kFloat, "dt_out must be float32");
            TORCH_CHECK(dt_out.dim() == 4 && dt_out.sizes().equals(at::IntArrayRef{Bsz, H, n_chunks, chunk_size}),
                        "dt_out must be [B, H, n_chunks, chunk_size]");
            TORCH_CHECK(dt_out.is_contiguous(), "dt_out must be contiguous");
        }

        if (!dA_cumsum_ || !dA_cumsum_->defined()) {
            dA_cumsum = at::empty({Bsz, H, n_chunks, chunk_size}, opts_f32);
        } else {
            dA_cumsum = std::move(*dA_cumsum_);
            TORCH_CHECK(dA_cumsum.device() == dev, "dA_cumsum device mismatch");
            TORCH_CHECK(dA_cumsum.scalar_type() == at::kFloat, "dA_cumsum must be float32");
            TORCH_CHECK(dA_cumsum.dim() == 4 && dA_cumsum.sizes().equals(at::IntArrayRef{Bsz, H, n_chunks, chunk_size}),
                        "dA_cumsum must be [B, H, n_chunks, chunk_size]");
            TORCH_CHECK(dA_cumsum.is_contiguous(), "dA_cumsum must be contiguous");
        }

        if (!states_ || !states_->defined()) {
            states = at::empty({Bsz, n_chunks, H, headdim, dstate}, opts_f32);
        } else {
            states = std::move(*states_);
            TORCH_CHECK(states.device() == dev, "states device mismatch");
            TORCH_CHECK(states.scalar_type() == at::kFloat, "states must be float32");
            TORCH_CHECK(states.dim() == 5 && states.sizes().equals(at::IntArrayRef{Bsz, n_chunks, H, headdim, dstate}),
                        "states must be [B, n_chunks, H, headdim, dstate]");
            TORCH_CHECK(states.is_contiguous(), "states must be contiguous");
        }

        if (!final_states_ || !final_states_->defined()) {
            final_states = at::empty({Bsz, H, headdim, dstate}, opts_f32);
        } else {
            final_states = std::move(*final_states_);
            TORCH_CHECK(final_states.device() == dev, "final_states device mismatch");
            TORCH_CHECK(final_states.scalar_type() == at::kFloat, "final_states must be float32");
            TORCH_CHECK(final_states.dim() == 4 && final_states.sizes().equals(at::IntArrayRef{Bsz, H, headdim, dstate}),
                        "final_states must be [B, H, headdim, dstate]");
            TORCH_CHECK(final_states.is_contiguous(), "final_states must be contiguous");
        }

        // --------- allocate / validate CB ---------
        // choose 16-bit dtype when cb_force_fp32 == false (bf16 preferred if any input is bf16)
        at::ScalarType cb16 = at::kHalf;
        // TODO: use once bf16 is supported
        // TODO: might need to support fp32 C and B too
        // (B.scalar_type() == at::kBFloat16 || C.scalar_type() == at::kBFloat16) ? at::kBFloat16 : at::kHalf;
        auto cb_opts = at::TensorOptions().dtype(cb_force_fp32 ? at::kFloat : cb16).device(dev);

        std::array<int64_t,5> cb_shape5 = {Bsz, n_chunks, ngroups, chunk_size, chunk_size};

        if (!CB_ || !CB_->defined()) {
            CB = at::empty(cb_shape5, cb_opts);
        } else {
            CB = std::move(*CB_);
            TORCH_CHECK(CB.device() == dev, "CB device mismatch");
            TORCH_CHECK(CB.scalar_type() == (cb_force_fp32 ? at::kFloat : cb16),
                        "CB dtype mismatch with cb_force_fp32 setting");
            TORCH_CHECK(CB.dim() == 5 &&
                        CB.size(0) == cb_shape5[0] &&
                        CB.size(1) == cb_shape5[1] &&
                        CB.size(2) == cb_shape5[2] &&
                        CB.size(3) == cb_shape5[3] &&
                        CB.size(4) == cb_shape5[4],
                        "CB must be [B, n_chunks, ngroups, chunk_size, chunk_size]");
            TORCH_CHECK(CB.is_contiguous(), "CB must be contiguous");
        }
    }
};
