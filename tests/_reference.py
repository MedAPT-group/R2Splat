"""Independent differentiable density formulas for small, well-conditioned scenes.

These references evaluate every sample and do not model discrete tile selection.
Tests must provide complete tile coverage and avoid support/culling boundaries.
Only ordinary PyTorch mathematics is used, with no R2Splat imports or CUDA VJPs.
"""

import math

import torch


def covariance_matrix(quats, scales):
    """Rotate basis vectors using quaternion cross products, then form R S² Rᵀ."""
    q = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    basis = torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], -1, -1)
    vector = q[:, None, 1:].expand_as(basis)
    cross = torch.linalg.cross(vector, basis, dim=-1)
    rotated = basis + 2 * (q[:, :1, None] * cross + torch.linalg.cross(vector, cross, dim=-1))
    rotation = rotated.transpose(-1, -2)
    return rotation @ torch.diag_embed(scales.square()) @ rotation.transpose(-1, -2)


def _profile(distance, exponent):
    if exponent is None:
        return torch.exp(-0.5 * distance)
    inside = (distance >= 0) & (distance < 1)
    # Keep inactive power branches finite without changing positive support.
    base = torch.where(inside, 1 - distance, torch.ones_like(distance))
    return torch.where(inside, base.pow(exponent), torch.zeros_like(distance))


def _accumulate(contribution, uncertainty, axis, detach_variance, threshold):
    contribution = torch.where(contribution.abs() >= threshold, contribution, 0)
    mean = contribution.sum(dim=axis)
    if uncertainty is None:
        return mean
    shape = [1] * contribution.ndim
    shape[axis] = -1
    branch = contribution.detach() if detach_variance else contribution
    variance = (uncertainty.reshape(shape) * branch.square()).sum(dim=axis)
    return mean, variance


def voxelize_reference(
    means, quats, scales, densities, volume_shape, *, volume_extent,
    volume_center=(0, 0, 0), beta=None, uncertainty=None, eps=0,
    detach_variance=False,
):
    """Evaluate physical density at voxel centers; eps has voxel-units squared."""
    depth, height, width = volume_shape
    extent = torch.as_tensor(volume_extent, device=means.device, dtype=means.dtype)
    center = torch.as_tensor(volume_center, device=means.device, dtype=means.dtype)
    dims = means.new_tensor([width, height, depth])
    covariance = covariance_matrix(quats, scales) + torch.diag(eps * (extent / dims).square())
    precision = torch.linalg.inv(covariance)
    axes = [
        center[axis] + ((torch.arange(size, device=means.device, dtype=means.dtype) + 0.5) / size - 0.5) * extent[axis]
        for axis, size in enumerate((width, height, depth))
    ]
    zz, yy, xx = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    delta = torch.stack((xx, yy, zz), dim=-1)[None] - means[:, None, None, None]
    distance = torch.einsum("ndhwi,nij,ndhwj->ndhw", delta, precision, delta)
    exponent = None if beta is None else beta[:, None, None, None]
    contribution = densities[:, None, None, None] * _profile(distance, exponent)
    threshold = 1e-6 if beta is None else 1e-5
    return _accumulate(contribution, uncertainty, 0, detach_variance, threshold)


def rasterize_reference(
    means, quats, scales, densities, viewmats, Ks, image_size, *, beta=None,
    uncertainty=None, eps=0, detach_variance=False,
):
    """Integrate the local camera/ray-coordinate density using matrix solves."""
    height, width = image_size
    rotation, translation = viewmats[:, :3, :3], viewmats[:, :3, 3]
    camera = torch.einsum("cij,nj->cni", rotation, means) + translation[:, None]
    x, y, z = camera.unbind(dim=-1)
    visible = z > 0.01
    z = torch.where(visible, z, torch.ones_like(z))
    fx, fy, cx, cy = (Ks[:, i, j, None] for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
    tx = z * torch.clamp(x / z, min=-(cx + 0.15 * width) / fx, max=(1.15 * width - cx) / fx)
    ty = z * torch.clamp(y / z, min=-(cy + 0.15 * height) / fy, max=(1.15 * height - cy) / fy)
    zero = torch.zeros_like(z)
    ray = torch.stack((tx, ty, z), dim=-1)
    jacobian = torch.stack((
        torch.stack((fx / z, zero, -fx * tx / z.square()), dim=-1),
        torch.stack((zero, fy / z, -fy * ty / z.square()), dim=-1),
        ray / torch.linalg.vector_norm(ray, dim=-1, keepdim=True),
    ), dim=-2)
    covariance = covariance_matrix(quats, scales)
    camera_covariance = rotation[:, None] @ covariance[None] @ rotation[:, None].transpose(-1, -2)
    projected = jacobian @ camera_covariance @ jacobian.transpose(-1, -2)
    projected = projected + eps * torch.eye(3, device=means.device, dtype=means.dtype)
    footprint = projected[..., :2, :2]
    cross = projected[..., :2, 2]
    ray_variance = projected[..., 2, 2] - (cross * torch.linalg.solve(footprint, cross[..., None]).squeeze(-1)).sum(-1)
    path = torch.sqrt(2 * math.pi * ray_variance)
    if beta is not None:
        # Moderate test exponents use the defining Beta/Gamma identity, separate
        # from the production implementation's stable asymptotic expansion.
        physical = beta.double()
        log_integral = math.lgamma(0.5) + torch.lgamma(physical + 1) - torch.lgamma(physical + 1.5)
        path = path * (log_integral.exp() / math.sqrt(2 * math.pi)).to(beta.dtype)[None]
    means2d = torch.stack((fx * x / z + cx, fy * y / z + cy), dim=-1)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=means.device, dtype=means.dtype) + 0.5,
        torch.arange(width, device=means.device, dtype=means.dtype) + 0.5,
        indexing="ij",
    )
    delta = torch.stack((xx, yy), dim=-1)[None, None] - means2d[:, :, None, None]
    distance = torch.einsum("cnhwi,cnij,cnhwj->cnhw", delta, torch.linalg.inv(footprint), delta)
    exponent = None if beta is None else (beta + 0.5)[None, :, None, None]
    contribution = (densities[None] * path)[:, :, None, None] * _profile(distance, exponent)
    contribution = torch.where(visible[:, :, None, None], contribution, 0)
    return _accumulate(contribution, uncertainty, 1, detach_variance, 1e-5)
