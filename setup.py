"""
Build script for gsplat2ct package

This setup.py follows gsplat's pattern for flexible CUDA extension compilation:
- Supports BUILD_NO_CUDA=1 to skip CUDA compilation (for JIT mode)
- Supports LINE_INFO=1 for CUDA debugging symbols
- Supports WITH_SYMBOLS=1 to preserve symbol table
- Supports MAX_JOBS for parallel compilation control
- Uses TORCH_CUDA_ARCH_LIST for GPU architecture configuration

Installation modes:
- Development: BUILD_NO_CUDA=1 pip install -e .[dev]
- Production:  pip install .

Author: Dy164
Date: 2025-12-16
"""

import glob
import os
import os.path as osp
import pathlib
import sys

from setuptools import find_packages, setup

# Environment variables for build control
BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")

# Set default MAX_JOBS if not specified
if MAX_JOBS is None:
    os.environ["MAX_JOBS"] = "10"


def get_ext():
    """Get the BuildExtension class with appropriate settings."""
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(use_ninja=True)


def get_extensions():
    """Build the CUDA extension configuration."""
    from torch.utils.cpp_extension import CUDAExtension

    current_dir = pathlib.Path(__file__).parent.resolve()
    
    # Collect source files
    sources = (
        list(glob.glob(osp.join(current_dir, "gsplat2ct", "cuda", "csrc", "*.cu")))
        + list(glob.glob(osp.join(current_dir, "gsplat2ct", "cuda", "csrc", "*.cpp")))
    )
    
    # Include directories
    include_dirs = [
        osp.join(current_dir, "gsplat2ct", "cuda", "include"),
    ]
    
    # Compiler flags
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
        ],
    }
    
    # Add debug info if requested
    if LINE_INFO:
        extra_compile_args["nvcc"].append("-lineinfo")
    
    # Link arguments (strip symbols by default for smaller binary)
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]
    
    extension = CUDAExtension(
        name="gsplat2ct.csrc",  # Standard module name pattern
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    
    return [extension]


# Read README for long description
def get_long_description():
    readme_path = pathlib.Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


setup(
    name="gsplat2ct",
    version="0.1.0",
    author="Dy164",
    description="3D Gaussian Splatting to CT with Second-Order Optimization",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "rich",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "matplotlib",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "pylint",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
