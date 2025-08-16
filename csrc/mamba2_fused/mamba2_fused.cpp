#include <torch/extension.h>

// forward decl from .cu
at::Tensor mamba2_fused_add(const at::Tensor& a, const at::Tensor& b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &mamba2_fused_add,
          "mamba2_fused add: out = a + b (float32, CUDA)");
}
