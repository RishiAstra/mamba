# csrc/mamba2_fused/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mamba2_fused",
    # IMPORTANT: no packages=... here; we're only building a C++/CUDA extension
    packages=[],  # prevent setuptools from trying to discover top-level packages
    ext_modules=[
        CUDAExtension(
            "mamba2_fused",
            ["mamba2_fused.cpp", "mamba2_fused_cuda.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
