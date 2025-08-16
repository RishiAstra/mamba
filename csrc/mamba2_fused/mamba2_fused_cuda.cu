#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// simple float32 elementwise add kernel
__global__ void add_kernel_f32(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out,
                               int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// C++/CUDA entrypoint
at::Tensor mamba2_fused_add(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
                "only float32 supported for now");
    TORCH_CHECK(a.sizes() == b.sizes(), "inputs must have the same shape");

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto out = at::empty_like(a_c);

    int64_t n = a_c.numel();
    if (n == 0) return out;

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    auto stream = at::cuda::getCurrentCUDAStream();

    add_kernel_f32<<<blocks, threads, 0, stream>>>(
        a_c.data_ptr<float>(),
        b_c.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

#ifndef NDEBUG
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ",
                cudaGetErrorString(err));
#endif
    return out;
}
