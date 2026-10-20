"""Numerical checks for mapping world coordinates onto an anisotropic grid."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_volume_mapping_matches_reference_and_means_finite_difference():
    from r2splat.cuda._wrapper import (
        world_to_volume_mapping_bwd,
        world_to_volume_mapping_fwd,
    )

    dtype = torch.float64
    device = "cuda"
    # Dimensions and physical extents both differ along every axis. Equal
    # dimensions/extents hide axis swaps and erroneous inverse Jacobians.
    dim_xyz = (13, 9, 5)
    dims = torch.tensor(dim_xyz, dtype=dtype, device=device)
    ranges = torch.tensor([2.0, 3.0, 4.0], dtype=dtype, device=device)
    centers = torch.tensor([0.2, -0.3, 1.0], dtype=dtype, device=device)
    means = torch.tensor(
        [[0.1, -0.1, 0.7], [0.4, -0.5, 1.1]], dtype=dtype, device=device
    )
    covars = torch.tensor(
        [[0.04, 0.006, -0.002, 0.09, 0.003, 0.16],
         [0.08, -0.004, 0.001, 0.05, -0.002, 0.12]],
        dtype=dtype,
        device=device,
    )
    max_scales = torch.full((2,), 0.5, dtype=dtype, device=device)
    eps3d = 1.0e-3

    def forward(points):
        return world_to_volume_mapping_fwd(
            points, covars, max_scales, ranges, centers, *dim_xyz, eps3d, 3.0
        )

    mapped, conics, radii = forward(means)
    scale = dims / ranges
    torch.testing.assert_close(mapped, (means - centers) * scale + dims / 2)

    covariance = torch.stack(
        (covars[:, [0, 1, 2]], covars[:, [1, 3, 4]], covars[:, [2, 4, 5]]),
        dim=1,
    )
    voxel_covariance = (
        covariance * scale[:, None] * scale[None, :]
        + eps3d * torch.eye(3, dtype=dtype, device=device)
    )
    inverse = torch.linalg.inv(voxel_covariance)
    rows, cols = torch.triu_indices(3, 3, device=device)
    torch.testing.assert_close(conics, inverse[:, rows, cols])
    expected_radii = torch.ceil(3.0 * voxel_covariance.diagonal(dim1=-2, dim2=-1).sqrt())
    torch.testing.assert_close(radii, expected_radii.to(torch.int32))

    upstream = torch.tensor(
        [[0.4, -1.2, 2.3], [-0.7, 0.6, -0.2]], dtype=dtype, device=device
    )
    grad_means, grad_covars = world_to_volume_mapping_bwd(
        means, covars, ranges, centers, *dim_xyz, eps3d,
        upstream, torch.zeros_like(conics),
    )
    torch.testing.assert_close(grad_means, upstream * scale)
    torch.testing.assert_close(grad_covars, torch.zeros_like(covars))

    step = 1.0e-5
    finite_difference = torch.empty_like(means)
    for row in range(means.shape[0]):
        for axis in range(3):
            plus, minus = means.clone(), means.clone()
            plus[row, axis] += step
            minus[row, axis] -= step
            finite_difference[row, axis] = (
                (forward(plus)[0] * upstream).sum()
                - (forward(minus)[0] * upstream).sum()
            ) / (2.0 * step)
    torch.testing.assert_close(grad_means, finite_difference, atol=1.0e-8, rtol=1.0e-8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("covariance_scale", [1.0e-16, 1.0e-6, 1.0, 1.0e16])
def test_covariance_inverse_and_vjp_are_independent_of_absolute_scale(dtype, covariance_scale):
    from r2splat.cuda._wrapper import (
        world_to_volume_mapping_bwd,
        world_to_volume_mapping_fwd,
    )

    device = "cuda"
    means = torch.zeros((1, 3), dtype=dtype, device=device)
    # Nonzero off-diagonals exercise the symmetric packing and inverse VJP.
    covars = torch.tensor(
        [[2.0, 0.2, -0.3, 1.0, 0.15, 3.0]], dtype=dtype, device=device
    ) * covariance_scale
    dim_xyz = (3, 5, 7)
    ranges = means.new_tensor([2.0, 4.0, 6.0])
    centers = means.new_zeros(3)
    max_scales = means.new_full((1,), covariance_scale ** 0.5)
    _, conics, radii = world_to_volume_mapping_fwd(
        means, covars, max_scales, ranges, centers, *dim_xyz, 0.0, 3.0
    )

    # Reference uses a full matrix inverse in double precision, independent of
    # the CUDA implementation's normalized Cholesky and packed VJP.
    reference_covars = covars.double().detach().requires_grad_()
    covariance = torch.stack(
        (reference_covars[:, [0, 1, 2]], reference_covars[:, [1, 3, 4]],
         reference_covars[:, [2, 4, 5]]), dim=1,
    )
    scale = ranges.new_tensor(dim_xyz).double() / ranges.double()
    inverse = torch.linalg.inv(covariance * scale[:, None] * scale[None, :])
    rows, cols = torch.triu_indices(3, 3, device=device)
    expected = inverse[:, rows, cols]
    tolerance = 3.0e-6 if dtype == torch.float32 else 2.0e-12
    torch.testing.assert_close(
        conics.double() * covariance_scale, expected * covariance_scale,
        atol=tolerance, rtol=tolerance,
    )
    assert (radii > 0).all()

    upstream = means.new_tensor([[0.7, -0.2, 0.3, -0.8, 0.4, 1.1]]) * covariance_scale
    _, actual_grad = world_to_volume_mapping_bwd(
        means, covars, ranges, centers, *dim_xyz, 0.0,
        torch.zeros_like(means), upstream,
    )
    expected_grad, = torch.autograd.grad((expected * upstream.double()).sum(), reference_covars)
    torch.testing.assert_close(
        actual_grad.double() * covariance_scale, expected_grad * covariance_scale,
        atol=tolerance, rtol=tolerance,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_beta", [False, True])
def test_small_primitive_at_voxel_center_preserves_density_and_gradient(dtype, use_beta):
    from r2splat import voxelize_bwd, voxelize_fwd

    means = torch.zeros((1, 3), device="cuda", dtype=dtype)
    quats = means.new_tensor([[1.0, 0.0, 0.0, 0.0]])
    scales = means.new_full((1, 3), 0.001)
    density = means.new_tensor([2.0])
    beta = means.new_tensor([3.5]) if use_beta else None
    volume, context = voxelize_fwd(
        means, quats, scales, density, (1, 1, 1),
        volume_extent=(1.0, 1.0, 1.0), beta=beta, eps=0.0,
    )
    torch.testing.assert_close(volume, means.new_full((1, 1, 1), 2.0))
    result = voxelize_bwd(context, torch.ones_like(volume))
    gradients = (result.means, result.quats, result.scales, result.densities)
    for gradient in gradients[:-1]:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    torch.testing.assert_close(gradients[-1], torch.ones_like(density))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("covariance_scale", [1.0e-6, 1.0, 1.0e6])
def test_invalid_covariances_have_zero_conics_radii_and_covariance_gradients(dtype, covariance_scale):
    from r2splat.cuda._wrapper import (
        world_to_volume_mapping_bwd,
        world_to_volume_mapping_fwd,
    )

    # Zero, rank deficient, and indefinite (including a positive determinant).
    covars = torch.tensor(
        [[0, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0],
         [1, 1, 1, 1, 1, 1], [1, 2, 0, 1, 0, 1],
         [-1, 0, 0, -1, 0, 1]],
        dtype=dtype, device="cuda",
    ) * covariance_scale
    means = covars.new_zeros((len(covars), 3))
    ranges = covars.new_ones(3)
    centers = covars.new_zeros(3)
    _, conics, radii = world_to_volume_mapping_fwd(
        means, covars, covars.new_ones(len(covars)), ranges, centers,
        1, 1, 1, 0.0, 3.0,
    )
    torch.testing.assert_close(conics, torch.zeros_like(conics))
    torch.testing.assert_close(radii, torch.zeros_like(radii))
    _, gradient = world_to_volume_mapping_bwd(
        means, covars, ranges, centers, 1, 1, 1, 0.0,
        torch.zeros_like(means), torch.ones_like(conics),
    )
    torch.testing.assert_close(gradient, torch.zeros_like(gradient))
