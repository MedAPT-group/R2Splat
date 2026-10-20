# API and mathematics

See [README.md](../README.md#api) for tensor shapes, coordinates, and public signatures.

## Explicit operator forward and backward

```python
from r2splat import rasterize_fwd, rasterize_bwd, voxelize_fwd, voxelize_bwd

image, ctx = rasterize_fwd(means, quats, scales, densities, viewmats, Ks, (24, 32))
grads = rasterize_bwd(ctx, 2 * (image - target) / image.numel())
# grads.means, .quats, .scales, .densities, .beta, .uncertainty
```

Forward returns `(output, ctx)` and builds no autograd graph. Backward returns
`ParameterGradients` for physical parameters without writing `.grad`.
`beta` and `uncertainty` fields are `None` when their inputs were omitted.
Upstream gradients must match output shape, device, and dtype. With uncertainty,
`output` is `(mean, variance)`; pass their gradients as the second and third
backward arguments, omitting the latter when unused.

Each context supports one backward call. Keep the original parameter and geometry
values unchanged until it finishes; apply optimizer updates after all contexts
sharing those inputs have been consumed. Recompute forward for another VJP.
Forward contexts used only to create fixed targets can be discarded.
Contexts require tensor version counters; inference-mode tensors are unsupported.

For a loss depending on projection and volume, call each backward separately
and sum gradients of their shared physical parameters. The examples use the
handwritten MSE output gradient `2 * (prediction - target) / prediction.numel()`.
For other losses, PyTorch may differentiate only the loss with respect to detached
output leaves to obtain these upstream gradients; the operator VJP still uses
`rasterize_bwd` or `voxelize_bwd` explicitly.

Positive parameterizations and their derivatives are caller-owned. Under
`torch.no_grad()`, assign the combined physical VJP to optimizer parameters:

```python
means.grad, quats.grad = grads.means, grads.quats
raw_scales.grad = grads.scales * scales             # scales = exp(raw_scales)
raw_densities.grad = grads.densities * raw_densities.sigmoid()  # softplus
raw_beta.grad = grads.beta * beta                    # beta = exp(raw_beta)
optimizer.step()
```

Use the Beta line only in Beta mode. `examples/optimize.py` contains a complete
Gaussian/Beta loop with two operators, finite-gradient checks, and a loss decrease check.

## Defaults and density

| Setting | Projection | Voxelization |
| --- | --- | --- |
| Regularization | `eps=1e-8` | `eps=1e-8` |
| Tile edge length | `16` pixels | `8` voxels |
| `backward_mode` | `"primitive"` | `"primitive"` |

`backward_mode` selects mean-only VJPs; mean/variance calls use fused
per-primitive VJPs. Tile selection includes each intersected tile's full extent,
so tile size can affect sampled Gaussian tails as well as performance.
`radius_clip` defaults to `3` for Gaussian and `1` for Beta; Beta density remains
zero outside its ellipsoid.

Both interfaces accept finite positive physical `beta [N]`; its VJP refers to
that exponent. Beta density is `density * max(1 - d², 0)**beta`, where `d²` is
squared Mahalanobis distance. Local perspective projection uses exponent
`beta + 0.5` and multiplies the Gaussian ray-length factor by
`B(1/2, beta + 1) / sqrt(2*pi)` for line-integral normalization.

With independent primitive contributions `c_i`, mean is `sum(c_i)` and variance
is `sum(q_i * c_i**2)`. `detach_variance=False` propagates the variance VJP
through contributions; setting it to `True` restricts this branch to `q_i`.

## Explicit covariance forward and VJP

```python
import torch
from r2splat.cuda import parameter_transform_fwd, parameter_transform_bwd

quats = torch.tensor([[2.0, 0.2, -0.1, 0.3]], device="cuda")
scales = torch.tensor([[0.15, 0.20, 0.25]], device="cuda")
covars, max_scales = parameter_transform_fwd(quats, scales)
v_covars = (2 * covars).contiguous()  # For L = sum(covars**2).
v_quats, v_scales = parameter_transform_bwd(quats, scales, v_covars)
```

CUDA normalizes wxyz quaternions and constructs `R @ diag(scales**2) @ R.T`.
Covariance is packed as `(00, 01, 02, 11, 12, 22)` with each off-diagonal once.
The VJP includes normalization and returns gradients of original quaternions and
physical scales, without an autograd graph or `.grad` writes. Reuse forward
parameter values and sum covariance gradients from all downstream uses.
`max_scales [N]` is nondifferentiable support metadata. Inputs must be contiguous
CUDA float32/float64 tensors on one device, with finite nonzero quaternions and
positive physical scales.

## Limits

Only first-order primitive VJPs are supported; camera and volume geometry are
constant. See [numerical limits](../README.md#numerical-behavior-and-limits).
Projection regularizes all three covariance axes, affecting footprint and
amplitude; volume regularization uses voxel units squared. Use zero only for
well-conditioned reference comparisons.
