# Contributing

Keep changes focused on CT operators, their interfaces, and numerical tests.

## Development setup

Install CUDA-enabled PyTorch, a matching local toolkit, and a supported host
compiler, then run from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools>=77" wheel ninja
python -m pip install -e ".[dev]" --no-build-isolation
python -m pytest -q
python examples/basic.py
```

There is no runtime JIT build. Reinstall after changing PyTorch, the toolkit,
or CUDA/C++ sources. For CPU-side work only:

```bash
BUILD_NO_CUDA=1 python -m pip install -e ".[dev]" --no-build-isolation
CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

This installation has no CPU operators. Passing CPU tests does not validate CUDA.
For operator changes, confirm the loaded extension, compare forward values and
first-order gradients against independent references or finite differences, and
report Python, PyTorch, toolkit, compiler, and GPU versions. Keep numerical
samples away from support and culling boundaries.

## Build controls and troubleshooting

| Variable | Purpose |
| --- | --- |
| `CUDA_HOME` | Local toolkit matching the PyTorch CUDA build |
| `CC` / `CXX` | Compilers supported by that toolkit |
| `MAX_JOBS` | Build parallelism, default `10`; lower for limited RAM |
| `TORCH_CUDA_ARCH_LIST` | Target architectures; set without a visible GPU (`8.6` for RTX 3090) |
| `CUDA_VISIBLE_DEVICES` | GPU selection on shared machines |
| `WITH_SYMBOLS=1` | Retain extension symbols |
| `BUILD_NO_CUDA=1` | Skip extension construction for CPU tests or packaging |

PyTorch's bundled CUDA runtime does not include `nvcc`. Check
`python -c "import torch; print(torch.__version__, torch.version.cuda)"` against
`nvcc --version`; use the [official PyTorch install commands](https://pytorch.org/get-started/previous-versions/).
For CUDA 11.8 with a newer system compiler, set `CUDA_HOME=/usr/local/cuda-11.8`,
`CC=gcc-11`, and `CXX=g++-11` before installation. These tools must exist locally.

For an unavailable extension, check GPU visibility and build errors, then
reinstall. With a regular installation, verify imports outside the checkout:

```bash
python -c "import r2splat, r2splat.csrc; print(r2splat.__file__); print(r2splat.csrc.__file__)"
```

## Validated CUDA combinations

Source-build and wheel-install checks were recorded on Ubuntu 24.04.4 x86-64,
two RTX 3090 GPUs (compute capability 8.6), and driver 580.178.04:

| Python | PyTorch | Toolkit (`nvcc`) | GCC / G++ | Tests |
| --- | --- | --- | --- | --- |
| 3.9.25 | 2.0.0+cu118 | 11.8.89 | 11.5.0 | 168, earlier suite |
| 3.9.25 | 2.1.2+cu118 | 11.8.89 | 11.5.0 | 274, current suite |
| 3.10.19 | 2.6.0+cu124 | 12.4.99 | 12.4.0 | 168, earlier suite |
| 3.11.15 | 2.11.0+cu130 | 13.0.48 | 13.3.0 | 168, earlier suite |

All listed test runs passed without skips. The current regression reused a
previously built extension with verified matching native sources. Earlier runs
imported installed wheels outside the checkout and passed the Gaussian/Beta examples.

Python/PyTorch lower bounds do not certify arbitrary combinations. Other GPU
architectures and operating systems remain unverified. GitHub Actions checks CPU
combinations and source packaging; it does not execute CUDA kernels.

## Source distribution

```bash
BUILD_NO_CUDA=1 python setup.py sdist
BUILD_NO_CUDA=1 python -m pip wheel dist/*.tar.gz --no-deps --no-build-isolation --wheel-dir /tmp/r2splat-wheels
```

The archive includes Python/CUDA code, headers, tests, examples, docs, and licenses.
The commands above check packaging and produce a wheel without CUDA operators.
For a CUDA release, build without `BUILD_NO_CUDA`, install the wheel in a clean
environment, and check imports, tests, and `examples/basic.py` outside the checkout.

Preserve source attributions and notices. Contributions use the repository's
Apache-2.0 license. See [docs/api.md](docs/api.md) for operator details.
