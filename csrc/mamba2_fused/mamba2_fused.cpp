// mamba2_fused.cpp
#include <torch/extension.h>
#include <tuple>
#include <optional>

#include "common.h"
#include "cumsum.h"
#include "bmm.h"

// Forward decl implemented in the .cu file
std::tuple<c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>, c10::optional<at::Tensor>> mamba2_fused_ssd_combined_fwd(
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    const bool dt_softplus,
    const int64_t chunk_size,
    const bool cb_force_fp32,
    const c10::optional<at::Tensor>& z,
    const c10::optional<at::Tensor>& dt_bias,
    const c10::optional<at::Tensor>& initial_states,
    const c10::optional<at::Tensor>& seq_idx,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& out,
    const c10::optional<at::Tensor>& out_x,
    const c10::optional<at::Tensor>& dt_out,
    const c10::optional<at::Tensor>& dA_cumsum,
    const c10::optional<at::Tensor>& states,
    const c10::optional<at::Tensor>& final_states,
    const c10::optional<at::Tensor>& CB
) {
    Mamba2SSDArgs args{
        x, dt, A, B, C, D,
        dt_softplus, chunk_size,
        /*cb_force_fp32=*/cb_force_fp32,
        z, dt_bias, initial_states, seq_idx, cu_seqlens,
        out, out_x, dt_out, dA_cumsum, states, final_states,
        CB
    };

    // for now, we just run cumsum
    return mamba2_cumsum_fwd_cuda(args);
    // return mamba2_bmm_chunk_fwd_cuda(args);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ssd_combined_fwd",
        &mamba2_fused_ssd_combined_fwd,
        pybind11::arg("x"),
        pybind11::arg("dt"),
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("D"),
        pybind11::arg("dt_softplus"),
        pybind11::arg("chunk_size"),
        pybind11::arg("cb_force_fp32") = false,
        pybind11::arg("z") = pybind11::none(),
        pybind11::arg("dt_bias") = pybind11::none(),
        pybind11::arg("initial_states") = pybind11::none(),
        pybind11::arg("seq_idx") = pybind11::none(),
        pybind11::arg("cu_seqlens") = pybind11::none(),
        pybind11::arg("out") = pybind11::none(),
        pybind11::arg("out_x") = pybind11::none(),
        pybind11::arg("dt_out") = pybind11::none(),
        pybind11::arg("dA_cumsum") = pybind11::none(),
        pybind11::arg("states") = pybind11::none(),
        pybind11::arg("final_states") = pybind11::none(),
        pybind11::arg("CB") = pybind11::none(),
        R"doc(
Returns a 6-tuple: (out, out_x, dt_out, dA_cumsum, states, final_states).
Currently only (dt_out, dA_cumsum) are populated; the rest are None.
dt_out and dA_cumsum are float32 with shape [B, H, n_chunks, chunk_size].
)doc"
    );
}
