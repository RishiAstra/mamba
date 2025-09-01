// mamba2_fused.cpp
#include <torch/extension.h>
#include <tuple>
#include <optional>

// Forward decl implemented in the .cu file
std::tuple<at::Tensor, at::Tensor> mamba2_fused_ssd_combined_fwd(
    const at::Tensor& x,
    const at::Tensor& dt,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    int64_t            chunk_size,
    const at::Tensor& D,
    c10::optional<at::Tensor> z,
    const at::Tensor& dt_bias,
    c10::optional<at::Tensor> initial_states,
    c10::optional<at::Tensor> seq_idx,
    c10::optional<at::Tensor> cu_seqlens,
    bool dt_softplus
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ssd_combined_fwd",
        &mamba2_fused_ssd_combined_fwd,
        pybind11::arg("x"),
        pybind11::arg("dt"),
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("chunk_size"),
        pybind11::arg("D"),
        pybind11::arg("z") = pybind11::none(),
        pybind11::arg("dt_bias"),
        pybind11::arg("initial_states") = pybind11::none(),
        pybind11::arg("seq_idx") = pybind11::none(),
        pybind11::arg("cu_seqlens") = pybind11::none(),
        pybind11::arg("dt_softplus") = true,
        R"doc(
Returns (dA_cumsum, dt_out) for step 1 of the fused Mamba SSD forward.
Both tensors are float32 with shape [B, n_chunks, H, chunk_size].
)doc"
    );
}
