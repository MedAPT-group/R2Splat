"""Explicit CUDA forward/VJP calls for physical primitive parameters.

Forward returns ``(output, context)`` without recording a differentiation graph.
Backward consumes the context once and returns gradients without writing
``.grad``. Geometry is constant; parameter activations and loss derivatives
belong to the caller. Do not modify inputs between forward and backward.
Version tracking requires ordinary tensors; inference-mode tensors are unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch

from .cuda import _wrapper as cuda
from ._beta import beta_path_log_grad, beta_path_ratio
from ._validation import _constant, _options, _primitives, _shape, _tensor, _vector

Tensor = torch.Tensor
Output = Union[Tensor, tuple[Tensor, Tensor]]


@dataclass
class ParameterGradients:
    """VJP tensors matching the physical inputs; absent optional inputs give None."""

    means: Tensor
    quats: Tensor
    scales: Tensor
    densities: Tensor
    beta: Optional[Tensor] = None
    uncertainty: Optional[Tensor] = None


@dataclass
class _Context:
    kind: str
    shape: tuple
    primitives: tuple
    buffers: dict
    versions: tuple
    eps: float
    backward_mode: str
    detach_variance: bool
    used: bool = False


def _mode(value):
    value = str(value).lower()
    if value not in ("pixel", "primitive"):
        raise ValueError("backward_mode must be 'pixel' or 'primitive'")
    return value


def _context(kind, shape, primitives, buffers, originals, eps, mode, detach_variance):
    watched = []
    seen = set()
    values = list(originals) + list(zip(
        ("means", "quats", "scales", "densities", "beta", "uncertainty"), primitives
    )) + list(buffers.items())
    for name, value in values:
        if isinstance(value, Tensor) and id(value) not in seen:
            try:
                version = value._version
            except RuntimeError as error:
                raise ValueError("explicit contexts require tensors with version counters") from error
            watched.append((name, value, version))
            seen.add(id(value))
    return _Context(kind, shape, primitives, buffers, tuple(watched), eps, mode, detach_variance)


def _check_context(context, kind):
    if not isinstance(context, _Context) or context.kind != kind:
        raise TypeError(f"expected a context returned by {kind}_fwd")
    if context.used:
        raise RuntimeError("this context has already been consumed by backward")
    for name, value, version in context.versions:
        if value._version != version:
            raise RuntimeError(f"saved input {name} was modified in-place after forward")


def _adjoint(name, value, context):
    return None if value is None else _tensor(name, value, context.shape, context.primitives[0])


def _zero_gradients(context):
    return ParameterGradients(*(
        None if value is None else torch.zeros_like(value) for value in context.primitives
    ))


def _consume(context, gradients):
    context.used = True
    context.primitives = ()
    context.buffers.clear()
    context.versions = ()
    return gradients


@torch.no_grad()
def rasterize_fwd(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    densities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    image_size: Sequence[int],
    *,
    beta: Optional[Tensor] = None,
    uncertainty: Optional[Tensor] = None,
    eps: float = 1.0e-8,
    radius_clip: Optional[float] = None,
    detach_variance: bool = False,
    backward_mode: str = "primitive",
) -> tuple[Output, _Context]:
    """Project physical primitives, returning a detached image and VJP context.

    World-to-camera ``viewmats [C,4,4]`` and pinhole ``Ks [C,3,3]`` are
    constant. Output is ``[C,height,width]`` or ``(image, variance)`` with
    uncertainty. ``backward_mode`` selects the mean-only CUDA VJP;
    variance adjoints use a fused per-primitive VJP. No input needs requires_grad.
    """
    originals = list(zip(
        ("means", "quats", "scales", "densities", "viewmats", "Ks", "beta", "uncertainty"),
        (means, quats, scales, densities, viewmats, Ks, beta, uncertainty),
    ))
    height, width = _shape("image_size", image_size, 2)
    primitives = _primitives(means, quats, scales, densities, beta, uncertainty)
    means, quats, scales, densities, beta, uncertainty = primitives
    eps, radius_clip = _options(beta, eps, radius_clip, detach_variance)
    mode = _mode(backward_mode)
    # Check originals: a contiguous copy under no_grad drops requires_grad.
    for name, value in (("viewmats", viewmats), ("Ks", Ks)):
        if isinstance(value, Tensor):
            _constant(name, value)
    viewmats = _constant("viewmats", _tensor("viewmats", viewmats, (None, 4, 4), means))
    cameras = viewmats.shape[0]
    if cameras == 0:
        raise ValueError("viewmats must contain at least one camera")
    if cameras * max(height * width, means.shape[0]) > 2**31 - 1 or cameras > 65535:
        raise ValueError("camera batch exceeds CUDA indexing or grid limits")
    if max(height, width) > 16 * 65535:
        raise ValueError("image_size exceeds CUDA grid limits")
    Ks = _constant("Ks", _tensor("Ks", Ks, (cameras, 3, 3), means))
    if bool((Ks[:, (0, 1), (0, 1)] <= 0).any()):
        raise ValueError("Ks must have positive fx and fy")
    if bool((Ks[:, (0, 1, 2, 2), (1, 0, 0, 1)] != 0).any()) or bool((Ks[:, 2, 2] != 1).any()):
        raise ValueError("Ks must be pinhole intrinsics with zero skew and last row [0, 0, 1]")
    if not torch.equal(viewmats[:, 3], means.new_tensor([0, 0, 0, 1]).expand(cameras, -1)):
        raise ValueError("viewmats must have homogeneous last row [0, 0, 0, 1]")
    shape = (cameras, height, width)
    buffers = {"viewmats": viewmats, "Ks": Ks}
    if means.shape[0] == 0:
        image = means.new_zeros(shape)
        output = image if uncertainty is None else (image, image.clone())
    else:
        covars, _ = cuda.parameter_transform_fwd(quats, scales)
        mapped, cov2d, radii = cuda.world_to_image_projection_fwd(
            means, covars, viewmats, Ks, width, height, eps, radius_clip
        )
        conics, base_mus = cuda.conics_mu_fwd(cov2d, eps)
        opacity = densities.unsqueeze(0).expand(cameras, -1).contiguous()
        mus, beta2d, ratio = base_mus, None, None
        if beta is not None:
            ratio = beta_path_ratio(beta)
            mus = base_mus * ratio.unsqueeze(0)
            beta2d = (beta + 0.5).unsqueeze(0).expand(cameras, -1).contiguous()
        tiles_x, tiles_y = (width + 15) // 16, (height + 15) // 16
        _, isect_ids, flatten_ids = cuda.isect_tiles(mapped, radii, 16, tiles_x, tiles_y)
        offsets = cuda.isect_offset_encode(isect_ids, cameras, tiles_x, tiles_y)
        prefix = "beta_" if beta is not None else ""
        args = (mapped, conics, opacity, mus) + (() if beta is None else (beta2d,))
        if uncertainty is None:
            output = getattr(cuda, prefix + "rasterize_to_pixels_fwd")(
                *args, offsets, flatten_ids, width, height, 16
            )
        else:
            output = getattr(cuda, prefix + "rasterize_to_pixels_mean_var_fwd")(
                *args, uncertainty, offsets, flatten_ids, width, height, 16
            )
        buffers.update(covars=covars, mapped=mapped, cov2d=cov2d, conics=conics,
                       base_mus=base_mus, mus=mus, opacity=opacity, beta2d=beta2d,
                       ratio=ratio, offsets=offsets, isect_ids=isect_ids, flatten_ids=flatten_ids)
    context = _context("rasterize", shape, primitives, buffers, originals, eps, mode, detach_variance)
    return output, context


@torch.no_grad()
def rasterize_bwd(context: _Context, v_image: Optional[Tensor], v_variance: Optional[Tensor] = None) -> ParameterGradients:
    """Consume a projection context and return physical-parameter gradients.

    Adjoint tensors match the output's shape, device, and dtype. None denotes
    a zero adjoint, including variance-only losses. Inputs must remain unchanged.
    """
    _check_context(context, "rasterize")
    means, quats, scales, densities, beta, uncertainty = context.primitives
    if v_variance is not None and uncertainty is None:
        raise ValueError("v_variance requires uncertainty in the forward call")
    v_image = _adjoint("v_image", v_image, context)
    v_variance = _adjoint("v_variance", v_variance, context)
    if means.shape[0] == 0 or (v_image is None and v_variance is None):
        return _consume(context, _zero_gradients(context))
    if v_image is None:
        v_image = means.new_zeros(context.shape)
    b = context.buffers
    _, height, width = context.shape
    prefix = "beta_" if beta is not None else ""
    args = (b["mapped"], b["conics"], b["opacity"], b["mus"])
    args += () if beta is None else (b["beta2d"],)
    if v_variance is not None:
        grads = getattr(cuda, prefix + "rasterize_to_pixels_mean_var_bwd_per_gaussian")(
            *args, b["offsets"], b["isect_ids"], b["flatten_ids"], v_image, v_variance,
            uncertainty, context.detach_variance, width, height, 16, False
        )
        v_uncertainty = grads[-1]
    else:
        primitive = context.backward_mode == "primitive"
        ids = (b["offsets"],) + ((b["isect_ids"],) if primitive else ()) + (b["flatten_ids"],)
        function = prefix + "rasterize_to_pixels_bwd" + ("_per_gaussian" if primitive else "")
        grads = getattr(cuda, function)(*args, *ids, v_image, width, height, 16, False)
        v_uncertainty = None if uncertainty is None else torch.zeros_like(uncertainty)
    v_mapped, v_conics, v_densities, v_mus = grads[1:5]
    v_beta = None
    if beta is not None:
        # mu_beta = mu_gaussian * ratio(beta); exponent = beta + 1/2.
        v_beta = grads[5].sum(dim=0)
        v_beta += (v_mus * b["mus"]).sum(dim=0) * beta_path_log_grad(beta)
        v_mus = (v_mus * b["ratio"].unsqueeze(0)).contiguous()
    v_cov2d = cuda.conics_mu_bwd(
        b["cov2d"], context.eps, b["conics"], b["base_mus"],
        v_conics.contiguous(), v_mus.contiguous()
    )
    v_means, v_covars = cuda.world_to_image_projection_bwd(
        means, b["covars"], b["viewmats"], b["Ks"], width, height,
        v_mapped.contiguous(), v_cov2d.contiguous()
    )
    v_quats, v_scales = cuda.parameter_transform_bwd(quats, scales, v_covars.contiguous())
    gradients = ParameterGradients(v_means, v_quats, v_scales, v_densities.sum(dim=0), v_beta, v_uncertainty)
    return _consume(context, gradients)


@torch.no_grad()
def voxelize_fwd(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    densities: Tensor,
    volume_shape: Sequence[int],
    *,
    volume_extent: Union[Tensor, Sequence[float]],
    volume_center: Union[Tensor, Sequence[float]] = (0.0, 0.0, 0.0),
    beta: Optional[Tensor] = None,
    uncertainty: Optional[Tensor] = None,
    eps: float = 1.0e-8,
    radius_clip: Optional[float] = None,
    detach_variance: bool = False,
    backward_mode: str = "primitive",
) -> tuple[Output, _Context]:
    """Sample physical primitives, returning a detached volume and VJP context.

    ``volume_shape`` and output use ``[depth,height,width]``; physical extent
    and center use xyz order. With uncertainty, output is ``(volume, variance)``.
    """
    originals = list(zip(
        ("means", "quats", "scales", "densities", "volume_extent", "volume_center", "beta", "uncertainty"),
        (means, quats, scales, densities, volume_extent, volume_center, beta, uncertainty),
    ))
    depth, height, width = _shape("volume_shape", volume_shape, 3)
    if max(height, width) > 8 * 65535:
        raise ValueError("volume_shape exceeds CUDA grid limits")
    primitives = _primitives(means, quats, scales, densities, beta, uncertainty)
    means, quats, scales, densities, beta, uncertainty = primitives
    eps, radius_clip = _options(beta, eps, radius_clip, detach_variance)
    mode = _mode(backward_mode)
    for name, value in (("volume_extent", volume_extent), ("volume_center", volume_center)):
        if isinstance(value, Tensor):
            _constant(name, value)
    extent = _vector("volume_extent", volume_extent, means)
    center = _vector("volume_center", volume_center, means)
    if bool((extent <= 0).any()):
        raise ValueError("volume_extent must contain positive physical lengths")
    shape = (depth, height, width)
    buffers = {"extent": extent, "center": center}
    if means.shape[0] == 0:
        volume = means.new_zeros(shape)
        output = volume if uncertainty is None else (volume, volume.clone())
    else:
        covars, max_scales = cuda.parameter_transform_fwd(quats, scales)
        mapped, conics, radii = cuda.world_to_volume_mapping_fwd(
            means, covars, max_scales, extent, center, width, height, depth, eps, radius_clip
        )
        tiles_x, tiles_y, tiles_z = (width + 7) // 8, (height + 7) // 8, (depth + 7) // 8
        _, isect_ids, flatten_ids = cuda.isect_tiles_3d(mapped, radii, 8, tiles_x, tiles_y, tiles_z)
        offsets = cuda.isect_offset_encode_3d(isect_ids, tiles_x, tiles_y, tiles_z)
        prefix = "beta_" if beta is not None else ""
        args = (mapped, conics, densities) + (() if beta is None else (beta,))
        if uncertainty is None:
            output = getattr(cuda, prefix + "voxelize_to_voxels_fwd")(
                *args, offsets, flatten_ids, width, height, depth, 8
            )
        else:
            output = getattr(cuda, prefix + "voxelize_to_voxels_mean_var_fwd")(
                *args, uncertainty, offsets, flatten_ids, width, height, depth, 8
            )
        buffers.update(covars=covars, mapped=mapped, conics=conics,
                       offsets=offsets, isect_ids=isect_ids, flatten_ids=flatten_ids)
    context = _context("voxelize", shape, primitives, buffers, originals, eps, mode, detach_variance)
    return output, context


@torch.no_grad()
def voxelize_bwd(context: _Context, v_volume: Optional[Tensor], v_variance: Optional[Tensor] = None) -> ParameterGradients:
    """Consume a volume context and return gradients without updating inputs."""
    _check_context(context, "voxelize")
    means, quats, scales, densities, beta, uncertainty = context.primitives
    if v_variance is not None and uncertainty is None:
        raise ValueError("v_variance requires uncertainty in the forward call")
    v_volume = _adjoint("v_volume", v_volume, context)
    v_variance = _adjoint("v_variance", v_variance, context)
    if means.shape[0] == 0 or (v_volume is None and v_variance is None):
        return _consume(context, _zero_gradients(context))
    if v_volume is None:
        v_volume = means.new_zeros(context.shape)
    b = context.buffers
    depth, height, width = context.shape
    prefix = "beta_" if beta is not None else ""
    args = (b["mapped"], b["conics"], densities) + (() if beta is None else (beta,))
    if v_variance is not None:
        grads = getattr(cuda, prefix + "voxelize_to_voxels_mean_var_bwd_per_gaussian")(
            *args, b["offsets"], b["isect_ids"], b["flatten_ids"], v_volume, v_variance,
            uncertainty, context.detach_variance, width, height, depth, 8
        )
        v_uncertainty = grads[-1]
    else:
        primitive = context.backward_mode == "primitive"
        ids = (b["offsets"],) + ((b["isect_ids"],) if primitive else ()) + (b["flatten_ids"],)
        function = prefix + "voxelize_to_voxels_bwd" + ("_per_gaussian" if primitive else "")
        grads = getattr(cuda, function)(*args, *ids, v_volume, width, height, depth, 8)
        v_uncertainty = None if uncertainty is None else torch.zeros_like(uncertainty)
    v_mapped, v_conics, v_densities = grads[:3]
    v_beta = None if beta is None else grads[3]
    v_means, v_covars = cuda.world_to_volume_mapping_bwd(
        means, b["covars"], b["extent"], b["center"], width, height, depth,
        context.eps, v_mapped.contiguous(), v_conics.contiguous()
    )
    v_quats, v_scales = cuda.parameter_transform_bwd(quats, scales, v_covars.contiguous())
    gradients = ParameterGradients(v_means, v_quats, v_scales, v_densities, v_beta, v_uncertainty)
    return _consume(context, gradients)


__all__ = ["ParameterGradients", "rasterize_fwd", "rasterize_bwd", "voxelize_fwd", "voxelize_bwd"]
