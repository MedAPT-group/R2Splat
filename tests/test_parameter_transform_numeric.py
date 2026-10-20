"""Independent covariance references and CUDA parameter-VJP regressions."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_covars(quats, scales):
    """Rotate basis vectors with quaternion cross products, then form R S² Rᵀ."""
    q = quats / torch.linalg.vector_norm(quats, dim=-1, keepdim=True)
    basis = torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], -1, -1)
    vector = q[:, None, 1:].expand_as(basis)
    first_cross = torch.linalg.cross(vector, basis, dim=-1)
    rotated_basis = basis + 2 * (
        q[:, :1, None] * first_cross
        + torch.linalg.cross(vector, first_cross, dim=-1)
    )
    rotation = rotated_basis.transpose(-1, -2)
    matrix = rotation @ torch.diag_embed(scales.square()) @ rotation.transpose(-1, -2)
    rows, cols = torch.triu_indices(3, 3, device=q.device)
    return matrix[:, rows, cols]


def _random_inputs(dtype, count=7):
    generator = torch.Generator(device="cuda").manual_seed(381)
    quats = torch.randn(count, 4, device="cuda", dtype=dtype, generator=generator)
    quats *= torch.linspace(0.2, 4.0, count, device="cuda", dtype=dtype)[:, None]
    scales = 0.15 + torch.rand(count, 3, device="cuda", dtype=dtype, generator=generator)
    upstream = torch.randn(count, 6, device="cuda", dtype=dtype, generator=generator)
    return quats, scales, upstream


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("off_diagonal_only", [False, True])
def test_cuda_parameter_forward_and_vjp_match_independent_reference(dtype, off_diagonal_only):
    from r2splat.cuda._wrapper import parameter_transform_bwd, parameter_transform_fwd

    quats, scales, upstream = _random_inputs(dtype)
    if off_diagonal_only:
        # Packed covariances contain each off-diagonal once. This catches a
        # mistaken factor of two when unpacking the covariance adjoint.
        upstream[:, [0, 3, 5]] = 0
    expected = _reference_covars(quats.requires_grad_(), scales.requires_grad_())
    expected_grads = torch.autograd.grad((expected * upstream).sum(), (quats, scales))
    actual, max_scales = parameter_transform_fwd(quats, scales)
    actual_grads = parameter_transform_bwd(quats, scales, upstream)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-11
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(max_scales, scales.max(dim=-1).values)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype,tiny,huge", [
    (torch.float32, 1e-20, 1e20),
    (torch.float64, 1e-200, 1e200),
])
def test_quaternion_rescaling_preserves_covariance_and_transforms_vjp(dtype, tiny, huge):
    from r2splat.cuda._wrapper import parameter_transform_bwd, parameter_transform_fwd

    quats = torch.tensor([[0.7, -0.2, 0.4, 0.1]], device="cuda", dtype=dtype)
    scales = quats.new_tensor([[0.23, 0.57, 1.2]])
    upstream = quats.new_tensor([[0.2, -0.4, 0.8, 0.6, -0.7, 0.3]])
    expected = _reference_covars(quats.requires_grad_(), scales.requires_grad_())
    expected_q_grad, expected_s_grad = torch.autograd.grad(
        (expected * upstream).sum(), (quats, scales)
    )
    factors = quats.new_tensor([1.0, -1.0, 0.125, -4.0, tiny, huge])[:, None]
    scaled_quats = (quats.detach() * factors).contiguous()
    expanded_scales = scales.detach().expand(factors.shape[0], -1).contiguous()
    expanded_upstream = upstream.expand(factors.shape[0], -1).contiguous()
    actual, _ = parameter_transform_fwd(scaled_quats, expanded_scales)
    q_grad, s_grad = parameter_transform_bwd(scaled_quats, expanded_scales, expanded_upstream)
    assert torch.isfinite(actual).all() and torch.isfinite(q_grad).all() and torch.isfinite(s_grad).all()
    tolerance = 3e-5 if dtype == torch.float32 else 3e-11
    torch.testing.assert_close(actual, expected.expand_as(actual), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(q_grad * factors, expected_q_grad.expand_as(q_grad), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(s_grad, expected_s_grad.expand_as(s_grad), atol=tolerance, rtol=tolerance)
    # Scale invariance makes the quaternion adjoint tangent to its sphere.
    torch.testing.assert_close((q_grad * scaled_quats).sum(dim=-1), torch.zeros_like(factors[:, 0]), atol=tolerance, rtol=0)



def test_float32_large_quaternion_preserves_subnormal_nonzero_vjp():
    from r2splat.cuda._wrapper import parameter_transform_bwd, parameter_transform_fwd

    factor = 1e38
    # All four quaternion gradients for this case lie below float32.tiny.
    # Move them to CPU double before rescaling so the comparison cannot itself
    # flush subnormal CUDA values to zero.
    q_cpu = torch.tensor([[0.7, -0.2, 0.4, 0.1]], dtype=torch.float32) * factor
    s_cpu = torch.tensor([[0.23, 0.57, 1.2]], dtype=torch.float32)
    upstream_cpu = torch.tensor([[0.2, -0.4, 0.8, 0.6, -0.7, 0.3]], dtype=torch.float32)
    q_reference = (q_cpu.double() / factor).requires_grad_()
    s_reference = s_cpu.double().requires_grad_()
    expected = _reference_covars(q_reference, s_reference)
    expected_grad_q, expected_grad_s = torch.autograd.grad(
        (expected * upstream_cpu.double()).sum(), (q_reference, s_reference)
    )
    quats, scales, upstream = q_cpu.cuda(), s_cpu.cuda(), upstream_cpu.cuda()
    actual, _ = parameter_transform_fwd(quats, scales)
    grad_q, grad_s = parameter_transform_bwd(quats, scales, upstream)
    grad_q_cpu = grad_q.cpu().double()
    assert (grad_q_cpu.abs() > 0).all()
    assert (grad_q_cpu.abs() < torch.finfo(torch.float32).tiny).all()
    torch.testing.assert_close(actual.cpu().double(), expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(grad_q_cpu * factor, expected_grad_q, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(grad_s.cpu().double(), expected_grad_s, atol=3e-6, rtol=3e-5)


def test_float32_subnormal_quaternion_has_valid_covariance_and_zero_vjp():
    from r2splat.cuda._wrapper import parameter_transform_bwd, parameter_transform_fwd

    # Integer multiples of 2**-140 are exactly representable subnormals. Their
    # nonzero VJP can exceed float32 range, but an all-zero adjoint must stay 0.
    q_cpu = torch.tensor([[7, -2, 4, 1]], dtype=torch.float32) * (2.0 ** -140)
    s_cpu = torch.tensor([[0.23, 0.57, 1.2]], dtype=torch.float32)
    expected = _reference_covars(q_cpu.double(), s_cpu.double())
    quats, scales = q_cpu.cuda(), s_cpu.cuda()
    actual, _ = parameter_transform_fwd(quats, scales)
    grad_q, grad_s = parameter_transform_bwd(quats, scales, torch.zeros_like(actual))
    torch.testing.assert_close(actual.cpu().double(), expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(grad_q.cpu(), torch.zeros_like(q_cpu), atol=0, rtol=0)
    torch.testing.assert_close(grad_s.cpu(), torch.zeros_like(s_cpu), atol=0, rtol=0)


def test_native_parameter_vjp_matches_finite_difference_without_graph():
    from r2splat.cuda._wrapper import parameter_transform_bwd, parameter_transform_fwd

    quats, scales, upstream = _random_inputs(torch.float64, count=2)
    covariance, max_scales = parameter_transform_fwd(quats, scales)
    actual = parameter_transform_bwd(quats, scales, upstream)
    assert covariance.grad_fn is None and max_scales.grad_fn is None
    assert all(gradient.grad_fn is None for gradient in actual)
    step = 1e-6
    for input_index, parameter in enumerate((quats, scales)):
        numeric = torch.empty_like(parameter)
        for index in range(parameter.numel()):
            plus = [quats.clone(), scales.clone()]
            minus = [quats.clone(), scales.clone()]
            plus[input_index].reshape(-1)[index] += step
            minus[input_index].reshape(-1)[index] -= step
            plus_covars, _ = parameter_transform_fwd(*plus)
            minus_covars, _ = parameter_transform_fwd(*minus)
            numeric.reshape(-1)[index] = ((plus_covars - minus_covars) * upstream).sum() / (2 * step)
        torch.testing.assert_close(actual[input_index], numeric, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
def test_public_parameter_vjp_accepts_strided_inputs(operation):
    from r2splat import rasterize_bwd, rasterize_fwd, voxelize_bwd, voxelize_fwd

    quats, scales, _ = _random_inputs(torch.float64, count=3)
    q_storage = torch.stack((quats, torch.zeros_like(quats)), dim=-1).flatten(1)
    s_storage = torch.stack((scales, torch.zeros_like(scales)), dim=-1).flatten(1)
    q_view, s_view = q_storage[:, ::2], s_storage[:, ::2]
    assert not q_view.is_contiguous() and not s_view.is_contiguous()
    means = quats.new_tensor([[0.12, -0.13, 2.08], [-0.12, 0.05, 2.4], [0.08, 0.18, 2.2]])
    densities, uncertainty = quats.new_tensor([1.4, 0.8, 1.1]), quats.new_tensor([0.2, 0.1, 0.3])
    camera = torch.eye(4, device=quats.device, dtype=quats.dtype).unsqueeze(0)
    intrinsics = quats.new_tensor([[[12, 0, 4.5], [0, 10, 3.5], [0, 0, 1]]])

    def forward(q, physical_scales):
        kwargs = dict(uncertainty=uncertainty, eps=0, radius_clip=8)
        if operation == "rasterize":
            return rasterize_fwd(means, q, physical_scales, densities, camera, intrinsics, (7, 9), **kwargs)
        return voxelize_fwd(means, q, physical_scales, densities, (5, 7, 9),
                            volume_extent=(4, 3, 2), volume_center=(0, 0, 2), **kwargs)

    strided, strided_context = forward(q_view, s_view)
    dense, dense_context = forward(quats, scales)
    for actual, expected in zip(strided, dense):
        torch.testing.assert_close(actual, expected, atol=3e-11, rtol=3e-11)
    weights = torch.linspace(0.2, 1.3, dense[0].numel(), device=quats.device, dtype=quats.dtype).reshape_as(dense[0])
    backward = rasterize_bwd if operation == "rasterize" else voxelize_bwd
    actual = backward(strided_context, weights, 0.7 * weights)
    expected = backward(dense_context, weights, 0.7 * weights)
    for name in ("means", "quats", "scales", "densities", "uncertainty"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=3e-11, rtol=3e-11)
    assert actual.quats.shape == q_view.shape and actual.scales.shape == s_view.shape


def test_empty_parameter_transform_outputs_and_vjps():
    from r2splat.cuda._wrapper import parameter_transform_bwd, parameter_transform_fwd

    quats = torch.empty(0, 4, device="cuda", dtype=torch.float64)
    scales = torch.empty(0, 3, device="cuda", dtype=torch.float64)
    covariance, max_scales = parameter_transform_fwd(quats, scales)
    assert covariance.shape == (0, 6) and max_scales.shape == (0,)
    gradients = parameter_transform_bwd(quats, scales, torch.empty_like(covariance))
    assert gradients[0].shape == quats.shape and gradients[1].shape == scales.shape


@pytest.mark.parametrize("invalid", ["quaternion_shape", "scale_shape", "batch", "dtype", "upstream_shape", "upstream_dtype"])
def test_parameter_transform_rejects_incompatible_tensors(invalid):
    from r2splat.cuda._wrapper import parameter_transform_bwd, parameter_transform_fwd

    quats, scales, upstream = _random_inputs(torch.float64, count=2)
    if invalid == "quaternion_shape":
        quats = quats[:, :3].contiguous()
    elif invalid == "scale_shape":
        scales = scales[:, :2].contiguous()
    elif invalid == "batch":
        scales = scales[:1]
    elif invalid == "dtype":
        scales = scales.float()
    elif invalid == "upstream_shape":
        upstream = upstream[:, :5].contiguous()
    else:
        upstream = upstream.float()
    with pytest.raises(RuntimeError):
        parameter_transform_bwd(quats, scales, upstream)
    if not invalid.startswith("upstream"):
        with pytest.raises(RuntimeError):
            parameter_transform_fwd(quats, scales)


@pytest.mark.parametrize("operation", ["rasterize", "voxelize"])
@pytest.mark.parametrize("use_beta", [False, True])
def test_public_rotated_primitive_parameter_vjp_matches_finite_difference(operation, use_beta):
    from r2splat import rasterize_bwd, rasterize_fwd, voxelize_bwd, voxelize_fwd

    means = torch.tensor([[0.12, -0.13, 2.08]], device="cuda", dtype=torch.float64)
    # A nonunit quaternion and three unequal scales exercise the full chain.
    quats = means.new_tensor([[1.8, -0.35, 0.65, 0.2]])
    scales = means.new_tensor([[0.45, 0.55, 0.65]])
    densities = means.new_tensor([1.4])
    beta = means.new_tensor([3.5]) if use_beta else None
    uncertainty = means.new_tensor([0.2])
    camera = torch.eye(4, device=means.device, dtype=means.dtype).unsqueeze(0)
    intrinsics = means.new_tensor([[[12, 0, 4.5], [0, 10, 3.5], [0, 0, 1]]])

    def forward(q, s):
        kwargs = dict(beta=beta, uncertainty=uncertainty, eps=0, radius_clip=8)
        if operation == "rasterize":
            return rasterize_fwd(means, q, s, densities, camera, intrinsics, (7, 9), **kwargs)
        return voxelize_fwd(means, q, s, densities, (5, 7, 9),
                            volume_extent=(4, 3, 2), volume_center=(0, 0, 2), **kwargs)

    (mean, variance), context = forward(quats, scales)
    weights = torch.linspace(0.2, 1.3, mean.numel(), device=means.device, dtype=means.dtype).reshape_as(mean)
    backward = rasterize_bwd if operation == "rasterize" else voxelize_bwd
    gradients = backward(context, weights, 0.7 * weights)
    analytic = (gradients.quats, gradients.scales)

    def loss(q, s):
        (mean, variance), _ = forward(q, s)
        return (weights * (mean + 0.7 * variance)).sum()

    step = 1e-5
    for input_index, parameter in enumerate((quats, scales)):
        numeric = torch.empty_like(parameter)
        for axis in range(parameter.shape[1]):
            plus = [quats.clone(), scales.clone()]
            minus = [quats.clone(), scales.clone()]
            plus[input_index][0, axis] += step
            minus[input_index][0, axis] -= step
            numeric[0, axis] = (loss(*plus) - loss(*minus)) / (2 * step)
        torch.testing.assert_close(analytic[input_index], numeric, atol=4e-4, rtol=3e-4)
