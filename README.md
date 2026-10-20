# R2Splat

CUDA operators for CT projection and density voxelization in PyTorch, with
Gaussian/Beta primitives, uncertainty propagation, and explicit first-order
forward/backward functions. Projection adds density line integrals without alpha compositing.

## Installation

Requires Linux, Python 3.9+, CUDA-enabled PyTorch 2.0+, an NVIDIA GPU, a matching
local CUDA toolkit, and a compatible C++ compiler. Install CUDA-enabled PyTorch
first; its bundled runtime does not include the required `nvcc` compiler.

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools>=77" wheel ninja
python -m pip install -e . --no-build-isolation
```

Run these commands from the repository root. For a regular installation, use
`python -m pip install . --no-build-isolation` and run outside the checkout.
`torch` is the only runtime dependency. Reinstall after changing PyTorch, the
CUDA toolkit, or native sources. See [CONTRIBUTING.md](CONTRIBUTING.md) for
validated environments, build controls, and tests.

## Quick start

Fit a synthetic projection with explicit gradients and Adam; no dataset is needed:

```python
import torch
from torch.nn import functional as F
from r2splat import rasterize_fwd, rasterize_bwd

with torch.no_grad():
    means = torch.nn.Parameter(torch.tensor([[0.05, 0.0, 2.0]], device="cuda"))
    quats = torch.nn.Parameter(torch.tensor([[2.0, 0.2, -0.1, 0.3]], device="cuda"))
    raw_scales = torch.nn.Parameter(torch.tensor([[0.15, 0.20, 0.25]], device="cuda").log())
    raw_density = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    viewmats = torch.eye(4, device="cuda")[None]
    Ks = torch.tensor([[[40., 0., 16.], [0., 40., 12.], [0., 0., 1.]]], device="cuda")
    image, _ = rasterize_fwd(means, quats, raw_scales.exp(), F.softplus(raw_density),
                             viewmats, Ks, (24, 32))
    target = 0.8 * image
    optimizer = torch.optim.Adam([means, quats, raw_scales, raw_density], lr=0.01)
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        scales, densities = raw_scales.exp(), F.softplus(raw_density)
        image, ctx = rasterize_fwd(means, quats, scales, densities, viewmats, Ks, (24, 32))
        g = rasterize_bwd(ctx, 2 * (image - target) / image.numel())
        means.grad, quats.grad = g.means, g.quats
        raw_scales.grad = g.scales * scales
        raw_density.grad = g.densities * raw_density.sigmoid()
        optimizer.step()
    print("MSE:", (image - target).square().mean().item())
```

Run [`python examples/basic.py`](examples/basic.py) for explicit projection/volume
VJPs with uncertainty, or [`python examples/optimize.py`](examples/optimize.py)
for 50-step Gaussian/Beta fitting with manually combined gradients.

## API

```python
rasterize_fwd(means, quats, scales, densities, viewmats, Ks, image_size,
              *, beta=None, uncertainty=None, eps=1e-8, radius_clip=None,
              detach_variance=False, backward_mode="primitive")
voxelize_fwd(means, quats, scales, densities, volume_shape,
             *, volume_extent, volume_center=(0.0, 0.0, 0.0), beta=None,
             uncertainty=None, eps=1e-8, radius_clip=None,
             detach_variance=False, backward_mode="primitive")
rasterize_bwd(ctx, v_image, v_variance=None)
voxelize_bwd(ctx, v_volume, v_variance=None)
```

Forward returns `(output, ctx)` without an autograd graph. Backward takes upstream
output gradients and returns `ParameterGradients` fields `means`, `quats`,
`scales`, `densities`, `beta`, and `uncertainty`; the last two are `None` when
unused. It returns physical-parameter gradients without writing `.grad`.
Use each context at most once and keep its inputs unchanged until backward.

Tensors must share a CUDA device and `torch.float32` or `torch.float64` dtype;
noncontiguous inputs are accepted. Parameters are physical values; activations
and their chain rules belong to the caller, as in the example above.

| Argument | Shape / convention |
| --- | --- |
| `means` | `[N, 3]`, world `(x, y, z)` |
| `quats` | `[N, 4]`, finite nonzero `(w, x, y, z)`, normalized in CUDA |
| `scales` | `[N, 3]`, positive principal-axis lengths in world units |
| `densities` | `[N]`, nonnegative peak densities |
| `viewmats` | `[C, 4, 4]`, world-to-camera rigid transforms; positive z faces forward |
| `Ks` | `[C, 3, 3]`, positive focal lengths, zero skew, last row `[0, 0, 1]` |
| `image_size` | `(height, width)`; output `[C, H, W]` |
| `volume_shape` | `(depth, height, width)`; output `[D, H, W]`, indexed `[z, y, x]` |
| `volume_extent` | Positive full lengths `(x, y, z)`, sequence or tensor `[3]` |
| `volume_center` | World center `(x, y, z)`, sequence or tensor `[3]` |
| `beta` | Optional `[N]` positive physical 3D exponents; omit for Gaussian |
| `uncertainty` | Optional `[N]` nonnegative relative variances |

Pixel/voxel centers use half-integer coordinates. Voxel x index `k` samples
`center_x + ((k + 0.5) / width - 0.5) * extent_x`.

Beta density is `density * max(1 - d², 0)**beta`; projection uses exponent
`beta + 0.5` and its line-integral factor. Gaussian density is `density * exp(-d²/2)`.
With uncertainty, `output` is `(mean, variance)` of matching shapes:
`variance = sum(q_i * contribution_i**2)`. Supply the mean gradient as the second
backward argument and the optional variance gradient as `v_variance`. `detach_variance=True`
restricts the variance branch's parameter gradients to `q`.

See [docs/api.md](docs/api.md) for loss gradients, activation chain rules,
and explicit covariance transforms.

## Numerical behavior and limits

- CUDA only, with first-order primitive gradients. Camera and volume geometry
  are constants; requesting their gradients is rejected. Higher-order derivatives
  and automatic mixed precision are unsupported.
- Projection uses a local perspective approximation to CT line integration.
  Parallel-beam geometry is unsupported; centers at camera depth `z <= 0.01` are culled.
- `eps=1e-8` regularizes all three projected covariance diagonals: two pixel axes
  and the ray-length axis in world units, affecting footprint and amplitude.
  Voxelization adds `eps` in voxel units squared. Use `eps=0` for well-conditioned
  unregularized reference comparisons.
- `radius_clip` defaults to `3` for Gaussian and `1` for Beta. Support selection
  has no gradient. Intersected tiles include their full extent, so tile size can
  change sampled Gaussian tails. Small-contribution thresholds also affect support.
- Input checks synchronize with the GPU; eager execution is required and CUDA
  Graph capture is unsupported. Float32 uses fast math; float64 Gaussian
  exponentials use double precision. Atomic summation can cause small differences.
- Empty primitive sets return zeros and zero parameter VJPs. Keep finite-difference
  samples away from support/culling boundaries.

## License and acknowledgements

[Apache-2.0](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for
gsplat attributions and dependency information.
