"""Build script for gsplat2ct.

Install for development with:

    pip install -e . --no-build-isolation

CUDA sources are compiled during installation. If CUDA/C++ sources change,
run the install command again.
"""

import glob
import os
import os.path as osp
import pathlib

from setuptools import find_packages, setup

MAX_JOBS = os.getenv("MAX_JOBS")
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"

if MAX_JOBS is None:
    os.environ["MAX_JOBS"] = "10"


def _ensure_cuda_arch_list():
    """Build for both the requested arch list and the visible local GPUs."""
    try:
        import torch
    except Exception:
        return

    archs = []
    raw = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    for item in raw.replace(",", ";").replace(" ", ";").split(";"):
        item = item.strip()
        if item and item not in archs:
            archs.append(item)

    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(idx)
            arch = f"{major}.{minor}"
            if arch not in archs:
                archs.append(arch)

    if archs:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(archs)


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(use_ninja=True)


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    _ensure_cuda_arch_list()

    current_dir = pathlib.Path(__file__).parent.resolve()
    csrc_dir = osp.join(current_dir, "gsplat2ct", "cuda", "csrc")
    sources = sorted(glob.glob(osp.join(csrc_dir, "*.cu"))) + sorted(
        glob.glob(osp.join(csrc_dir, "*.cpp"))
    )

    return [
        CUDAExtension(
            name="gsplat2ct.csrc",
            sources=sources,
            include_dirs=[osp.join(current_dir, "gsplat2ct", "cuda", "include")],
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
    ]


def get_long_description():
    readme_path = pathlib.Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""


setup(
    name="gsplat2ct",
    version="0.1.0",
    author="Dy164",
    description="Fast first-order 3D Gaussian Splatting for CT reconstruction",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "matplotlib",
        "tqdm",
    ],
    extras_require={"dev": ["pytest", "black", "isort", "pylint"]},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": get_ext()},
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
