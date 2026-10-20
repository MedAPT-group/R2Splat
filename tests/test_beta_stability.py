"""Beta normalization references computed with 160-digit Gamma/digamma arithmetic."""
import math

import pytest
import torch

from r2splat._beta import beta_path_log_grad, beta_path_ratio


# gamma, B(1/2, gamma+1)/sqrt(2*pi), d log B / d gamma.
REFERENCES = [
    (0.001, 0.7973953276921394, -0.6129962938258033),
    (3.5, 0.3427030844222071, -0.11724674207227156),
    (31.999, 0.12356049922492537, -0.015266752293913137),
    (32.0, 0.12355861290058569, -0.015266286187837997),
    (32.001, 0.12355672666263216, -0.01526582011022151),
    (1e5, 0.002236059592288547, -4.999962500249999e-6),
    (1e6, 0.0007071065160216427, -4.9999962500025e-7),
    (1e7, 0.0002236067893647245, -4.999999625000025e-8),
    (1e12, 7.071067811862824e-7, -4.99999999999625e-13),
    (1e30, 7.071067811865475e-16, -5e-31),
]


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_beta_ratio_and_gradient_across_large_exponents(dtype, device):
    data = torch.tensor(REFERENCES, device=device, dtype=dtype)
    gamma = data[:, 0].clone().requires_grad_()
    ratio = beta_path_ratio(gamma)
    automatic = torch.autograd.grad(ratio.log().sum(), gamma)[0]
    tolerance = 3e-6 if dtype == torch.float32 else 2e-12
    assert ratio.dtype == gamma.dtype
    torch.testing.assert_close(ratio, data[:, 1], atol=0, rtol=tolerance)
    torch.testing.assert_close(automatic, data[:, 2], atol=0, rtol=tolerance)
    torch.testing.assert_close(beta_path_log_grad(gamma), data[:, 2], atol=0, rtol=tolerance)


def test_huge_double_beta_keeps_inactive_branch_finite():
    gamma = torch.tensor([1e100], dtype=torch.float64, requires_grad=True)
    ratio = beta_path_ratio(gamma)
    gradient = torch.autograd.grad(ratio.log().sum(), gamma)[0]
    torch.testing.assert_close(ratio, gamma.new_tensor([7.071067811865475e-51]), atol=0, rtol=3e-14)
    torch.testing.assert_close(gradient, gamma.new_tensor([-5e-101]), atol=0, rtol=3e-14)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('row', [5, 6, 7])
def test_large_float32_beta_center_projection_and_gradient(row):
    from r2splat import rasterize_bwd, rasterize_fwd

    exponent, ratio, log_gradient = REFERENCES[row]
    means = torch.tensor([[0., 0., 1.]], device='cuda')
    quats = means.new_tensor([[1., 0., 0., 0.]])
    scales = means.new_full((1, 3), 0.2)
    density = means.new_tensor([1.])
    beta = means.new_tensor([exponent])
    view = torch.eye(4, device='cuda')[None]
    intrinsics = means.new_tensor([[[1., 0., .5], [0., 1., .5], [0., 0., 1.]]])
    image, context = rasterize_fwd(means, quats, scales, density, view, intrinsics, (1, 1), eps=0, beta=beta)
    gradients = rasterize_bwd(context, torch.ones_like(image))
    density_grad, beta_grad = gradients.densities, gradients.beta
    expected = 0.2 * math.sqrt(2 * math.pi) * ratio
    torch.testing.assert_close(image, means.new_full((1, 1, 1), expected), atol=0, rtol=4e-6)
    torch.testing.assert_close(density_grad, means.new_tensor([expected]), atol=0, rtol=4e-6)
    torch.testing.assert_close(beta_grad, means.new_tensor([expected * log_gradient]), atol=0, rtol=4e-6)
