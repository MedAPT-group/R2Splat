"""Build the r2splat CUDA extension against the installed PyTorch."""

import os
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
BUILD_NO_CUDA = os.environ.get("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.environ.get("WITH_SYMBOLS", "0") == "1"


def cuda_extension():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    os.environ.setdefault("MAX_JOBS", "10")
    source_dir = Path("r2splat") / "cuda" / "csrc"
    sources = sorted(
        path.as_posix()
        for pattern in ("*.cpp", "*.cu")
        for path in source_dir.glob(pattern)
    )
    extension = CUDAExtension(
        name="r2splat.csrc",
        sources=sources,
        include_dirs=[str(ROOT / "r2splat" / "cuda" / "include")],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-std=c++17",
                "--expt-relaxed-constexpr",
            ],
        },
        extra_link_args=[] if WITH_SYMBOLS else ["-s"],
    )
    # Ninja is the default; older PyTorch forwards explicit options to distutils.
    return [extension], {"build_ext": BuildExtension}


# This opt-out supports metadata, source distributions, and CPU-side API tests.
# It does not provide a CPU implementation of the CUDA operators.
ext_modules, cmdclass = ([], {}) if BUILD_NO_CUDA else cuda_extension()

setup(
    name="r2splat",
    version="0.1.0",
    author="Dy164",
    description="Differentiable CUDA operators for Gaussian and Beta CT splatting",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license_expression="Apache-2.0",
    license_files=["LICENSE", "THIRD_PARTY_NOTICES*", "third_party/gsplat/LICENSE"],
    packages=find_packages(include=["r2splat", "r2splat.*"]),
    include_package_data=True,
    package_data={
        "r2splat": [
            "cuda/csrc/*.cpp",
            "cuda/csrc/*.cu",
            "cuda/csrc/*.h",
            "cuda/include/*.h",
            "cuda/include/*.cuh",
        ],
    },
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
    extras_require={"dev": ["pytest>=7"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
    ],
)
