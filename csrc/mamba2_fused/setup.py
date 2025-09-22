from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_args = [
    "-O3",
    "-lineinfo",
    # TODO: check if fast math is ok
    "--use_fast_math",
    # TODO: decide what architectures to support
    "-gencode=arch=compute_89,code=sm_89",
]

cxx_args = ["-O3"]

setup(
    name="mamba2_fused",
    # IMPORTANT: no packages=... here; we're only building a C++/CUDA extension
    packages=[],  # prevent setuptools from trying to discover top-level packages
    ext_modules=[
        CUDAExtension(
            "mamba2_fused",
            ["mamba2_fused.cpp", "mamba2_fused_cuda.cu", "mamba2_bmm.cu"],
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
