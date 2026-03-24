"""
Test script for WorldToImageProjection module.

Validates the CUDA implementation against the PyTorch autograd
reference implementation by comparing:
1. Forward outputs (means2d, cov2d_3x3)
2. First-order backward outputs (v_means, v_covars)
3. Second-order backward outputs (Hessian blocks)
4. Numerical finite-difference validation

Usage:
    python tests/test_world_to_image_projection.py

Author: Dy164
Date: 2026-02-19
"""

import torch
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.data import Params, Data, Config
from gsplat2ct.ops.world_to_image_projection import (
    WorldToImageProjectionCUDA,
    WorldToImageProjectionAutograd,
    create_world_to_image_projection,
    _compute_world_to_image_projection,
)


# ============================================================================
# Utility: ParamWrapper
# ============================================================================

class ParamWrapper:
    """Simple wrapper to satisfy Params.add_param interface."""
    def __init__(self, data):
        self.data = data


# ============================================================================
# Data Generation Utilities
# ============================================================================

def generate_random_spd_covars(
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """Generate random SPD 3x3 covariance matrices in upper-tri storage.

    Uses Cholesky decomposition: Sigma = L @ L^T to guarantee SPD.
    Returns in upper-triangular storage: (s00, s01, s02, s11, s12, s22).

    Args:
        N: Number of Gaussians
        device: PyTorch device
        dtype: PyTorch dtype
        seed: Random seed

    Returns:
        covars: (N, 6) in upper-triangular storage
    """
    torch.manual_seed(seed)

    # Generate lower triangular L with positive diagonal
    # Use moderate magnitudes so projected covariance stays within f32 precision.
    l00 = torch.rand(N, device=device, dtype=dtype) * 0.5 + 0.1
    l10 = torch.randn(N, device=device, dtype=dtype) * 0.2
    l11 = torch.rand(N, device=device, dtype=dtype) * 0.5 + 0.1
    l20 = torch.randn(N, device=device, dtype=dtype) * 0.2
    l21 = torch.randn(N, device=device, dtype=dtype) * 0.2
    l22 = torch.rand(N, device=device, dtype=dtype) * 0.5 + 0.1

    # Sigma = L @ L^T
    s00 = l00 * l00
    s01 = l00 * l10
    s02 = l00 * l20
    s11 = l10 * l10 + l11 * l11
    s12 = l10 * l20 + l11 * l21
    s22 = l20 * l20 + l21 * l21 + l22 * l22

    covars = torch.stack([s00, s01, s02, s11, s12, s22], dim=-1)  # (N, 6)
    return covars


def generate_random_viewmats(
    C: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 43,
) -> torch.Tensor:
    """Generate random valid view matrices (C, 4, 4).

    Uses random rotation (via QR decomposition) and random translation.
    Ensures det(R) = +1 (proper rotation).

    Args:
        C: Number of cameras
        device: PyTorch device
        dtype: PyTorch dtype
        seed: Random seed

    Returns:
        viewmats: (C, 4, 4) row-major view matrices
    """
    torch.manual_seed(seed)

    viewmats = torch.zeros(C, 4, 4, device=device, dtype=dtype)
    viewmats[:, 3, 3] = 1.0

    for c in range(C):
        # Random rotation via QR of a random matrix
        A = torch.randn(3, 3, device=device, dtype=dtype)
        Q, R_diag = torch.linalg.qr(A)
        # Ensure proper rotation (det = +1)
        det = torch.det(Q)
        if det < 0:
            Q[:, 0] *= -1
        viewmats[c, :3, :3] = Q

        # Random translation: place camera a bit away
        viewmats[c, :3, 3] = torch.randn(3, device=device, dtype=dtype) * 2.0

    return viewmats


def generate_random_intrinsics(
    C: int,
    device: torch.device,
    dtype: torch.dtype,
    image_width: int = 640,
    image_height: int = 480,
    seed: int = 44,
) -> torch.Tensor:
    """Generate random camera intrinsic matrices (C, 3, 3).

    Args:
        C: Number of cameras
        device: PyTorch device
        dtype: PyTorch dtype
        image_width: Image width
        image_height: Image height
        seed: Random seed

    Returns:
        Ks: (C, 3, 3) intrinsic matrices
    """
    torch.manual_seed(seed)

    Ks = torch.zeros(C, 3, 3, device=device, dtype=dtype)
    for c in range(C):
        fx = 300.0 + torch.rand(1, device=device, dtype=dtype).item() * 200.0
        fy = 300.0 + torch.rand(1, device=device, dtype=dtype).item() * 200.0
        cx = image_width / 2.0 + (torch.rand(1, device=device, dtype=dtype).item() - 0.5) * 20.0
        cy = image_height / 2.0 + (torch.rand(1, device=device, dtype=dtype).item() - 0.5) * 20.0
        Ks[c, 0, 0] = fx
        Ks[c, 1, 1] = fy
        Ks[c, 0, 2] = cx
        Ks[c, 1, 2] = cy
        Ks[c, 2, 2] = 1.0

    return Ks


def generate_test_data(
    C: int = 2,
    N: int = 100,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    image_width: int = 640,
    image_height: int = 480,
    seed: int = 42,
):
    """Generate a complete test data set.

    Ensures all Gaussians are in front of the camera (z > 0.5).
    means is (N, 3) shared across all C cameras.
    covars is (N, 6) shared across all C cameras.

    Returns:
        means (N, 3), covars (N, 6), viewmats (C, 4, 4), Ks (C, 3, 3)
    """
    if device is None:
        device = torch.device("cuda")

    viewmats = generate_random_viewmats(C, device, dtype, seed=seed + 1)
    Ks = generate_random_intrinsics(C, device, dtype, image_width, image_height,
                                     seed=seed + 2)

    # Generate camera-space points with z > 3 (far enough for f32 precision)
    # in front of the first camera, then transform back to world space.
    # With z ~ 3-5 and fx ~ 300-500, J00 = fx/z ~ 60-170, so
    # cov2d ~ (170)^2 * sigma ~ 30000 * sigma, well within f32 range.
    torch.manual_seed(seed + 3)
    p_c = torch.randn(N, 3, device=device, dtype=dtype) * 0.5
    p_c[:, 2] = p_c[:, 2].abs() + 3.0  # ensure z > 3

    # Transform back to world space: p_w = R^T (p_c - t)
    R = viewmats[0, :3, :3]   # (3, 3)
    t = viewmats[0, :3, 3]    # (3,)
    Rt = R.T                   # (3, 3)
    means = torch.einsum('ij,nj->ni', Rt, p_c - t.unsqueeze(0)).contiguous()

    covars = generate_random_spd_covars(N, device, dtype, seed=seed + 4)

    return means, covars, viewmats, Ks


# ============================================================================
# Test Functions
# ============================================================================

def test_forward_consistency(
    C: int = 2,
    N: int = 100,
    dtype: torch.dtype = torch.float32,
    image_width: int = 640,
    image_height: int = 480,
):
    """Test that CUDA and autograd forward passes produce the same results."""
    print(f"\n{'='*60}")
    print(f"Test Forward Consistency: C={C}, N={N}, dtype={dtype}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    means, covars, viewmats, Ks = generate_test_data(
        C, N, device, dtype, image_width, image_height
    )

    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('covars', ParamWrapper(covars))

    config = Config()
    config.set('dim1', image_width)
    config.set('dim0', image_height)

    data = Data()
    data.set('viewmats', viewmats)
    data.set('Ks', Ks)

    # CUDA forward
    cuda_op = WorldToImageProjectionCUDA()
    cuda_op.initialize(params, config)
    cuda_result = cuda_op.forward_only(params, data, config)

    means2d_cuda = cuda_result.outputs['means2d']
    cov2d_cuda = cuda_result.outputs['cov2d_3x3']

    # Autograd forward
    ag_op = WorldToImageProjectionAutograd()
    ag_op.initialize(params, config)
    ag_result = ag_op.forward_only(params, data, config)

    means2d_ag = ag_result.outputs['means2d']
    cov2d_ag = ag_result.outputs['cov2d_3x3']

    # Compare
    means2d_diff = (means2d_cuda - means2d_ag).abs().max().item()
    cov2d_diff = (cov2d_cuda - cov2d_ag).abs().max().item()

    # Use relative comparison: allow rtol-based tolerance for large-valued outputs
    atol = 1e-4 if dtype == torch.float32 else 1e-8
    rtol = 1e-4 if dtype == torch.float32 else 1e-8

    print(f"  means2d max abs diff:   {means2d_diff:.2e}")
    print(f"  cov2d_3x3 max abs diff: {cov2d_diff:.2e}")

    # Per-element cov2d breakdown
    names = ['s00', 's01', 's02', 's11', 's12', 's22']
    for i, name in enumerate(names):
        elem_diff = (cov2d_cuda[..., i] - cov2d_ag[..., i]).abs().max().item()
        print(f"    cov2d {name} max diff: {elem_diff:.2e}")

    means2d_ok = torch.allclose(means2d_cuda, means2d_ag, atol=atol, rtol=rtol)
    cov2d_ok = torch.allclose(cov2d_cuda, cov2d_ag, atol=atol, rtol=rtol)

    print(f"  means2d match:   {'PASS' if means2d_ok else 'FAIL'} (atol={atol:.0e}, rtol={rtol:.0e})")
    print(f"  cov2d_3x3 match: {'PASS' if cov2d_ok else 'FAIL'} (atol={atol:.0e}, rtol={rtol:.0e})")

    return means2d_ok and cov2d_ok


def test_backward_consistency(
    C: int = 2,
    N: int = 100,
    dtype: torch.dtype = torch.float32,
    image_width: int = 640,
    image_height: int = 480,
):
    """Test that CUDA and autograd backward passes produce the same gradients."""
    print(f"\n{'='*60}")
    print(f"Test Backward Consistency: C={C}, N={N}, dtype={dtype}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    means, covars, viewmats, Ks = generate_test_data(
        C, N, device, dtype, image_width, image_height
    )

    # Generate random upstream gradients
    torch.manual_seed(999)
    v_means2d = torch.randn(C, N, 2, device=device, dtype=dtype)
    v_cov2d_3x3 = torch.randn(C, N, 6, device=device, dtype=dtype)

    config = Config()
    config.set('dim1', image_width)
    config.set('dim0', image_height)

    # --- CUDA backward ---
    params_cuda = Params()
    params_cuda.add_param('means', ParamWrapper(means.clone()))
    params_cuda.add_param('covars', ParamWrapper(covars.clone()))
    data_cuda = Data()
    data_cuda.set('viewmats', viewmats.clone())
    data_cuda.set('Ks', Ks.clone())
    data_cuda.set('v_means2d', v_means2d.clone())
    data_cuda.set('v_cov2d_3x3', v_cov2d_3x3.clone())

    cuda_op = WorldToImageProjectionCUDA()
    cuda_op.initialize(params_cuda, config)
    cuda_fwd, cuda_grad = cuda_op.forward_backward1(params_cuda, data_cuda, config)

    v_means_cuda = cuda_grad.param_grads['means']
    v_covars_cuda = cuda_grad.param_grads['covars']

    # --- Autograd backward ---
    params_ag = Params()
    params_ag.add_param('means', ParamWrapper(means.clone()))
    params_ag.add_param('covars', ParamWrapper(covars.clone()))
    data_ag = Data()
    data_ag.set('viewmats', viewmats.clone())
    data_ag.set('Ks', Ks.clone())
    data_ag.set('v_means2d', v_means2d.clone())
    data_ag.set('v_cov2d_3x3', v_cov2d_3x3.clone())

    ag_op = WorldToImageProjectionAutograd()
    ag_op.initialize(params_ag, config)
    ag_fwd, ag_grad = ag_op.forward_backward1(params_ag, data_ag, config)

    v_means_ag = ag_grad.param_grads['means']
    v_covars_ag = ag_grad.param_grads['covars']

    # Compare forward
    fwd_means2d_diff = (cuda_fwd.outputs['means2d'] - ag_fwd.outputs['means2d']).abs().max().item()
    fwd_cov2d_diff = (cuda_fwd.outputs['cov2d_3x3'] - ag_fwd.outputs['cov2d_3x3']).abs().max().item()
    print(f"  Forward means2d max diff:   {fwd_means2d_diff:.2e}")
    print(f"  Forward cov2d_3x3 max diff: {fwd_cov2d_diff:.2e}")

    # Compare gradients
    means_grad_diff = (v_means_cuda - v_means_ag).abs()
    covars_grad_diff = (v_covars_cuda - v_covars_ag).abs()

    max_means = means_grad_diff.max().item()
    max_cov = covars_grad_diff.max().item()
    mean_means = means_grad_diff.mean().item()
    mean_cov = covars_grad_diff.mean().item()

    print(f"  v_means max abs diff:  {max_means:.2e}")
    print(f"  v_means mean abs diff: {mean_means:.2e}")

    for i, name in enumerate(['x', 'y', 'z']):
        d = means_grad_diff[..., i].max().item()
        print(f"    v_means_{name} max diff: {d:.2e}")

    print(f"  v_covars max abs diff:   {max_cov:.2e}")
    print(f"  v_covars mean abs diff:  {mean_cov:.2e}")

    cov_names = ['s00', 's01', 's02', 's11', 's12', 's22']
    for i, name in enumerate(cov_names):
        d = covars_grad_diff[..., i].max().item()
        print(f"    v_covars_{name} max diff: {d:.2e}")

    atol = 1e-3 if dtype == torch.float32 else 1e-8
    rtol = 1e-3 if dtype == torch.float32 else 1e-8
    means_ok = torch.allclose(v_means_cuda, v_means_ag, atol=atol, rtol=rtol)
    cov_ok = torch.allclose(v_covars_cuda, v_covars_ag, atol=atol, rtol=rtol)

    print(f"  v_means match: {'PASS' if means_ok else 'FAIL'} (atol={atol:.0e}, rtol={rtol:.0e})")
    print(f"  v_covars match:  {'PASS' if cov_ok else 'FAIL'} (atol={atol:.0e}, rtol={rtol:.0e})")

    return means_ok and cov_ok


def test_backward_vs_numerical(
    C: int = 1,
    N: int = 5,
    dtype: torch.dtype = torch.float64,
    image_width: int = 640,
    image_height: int = 480,
    fd_eps: float = 1e-6,
):
    """Validate backward using finite differences (gold standard).

    Uses float64 for better numerical accuracy.
    """
    print(f"\n{'='*60}")
    print(f"Test Backward vs Numerical (Finite Diff): C={C}, N={N}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    means, covars, viewmats, Ks = generate_test_data(
        C, N, device, dtype, image_width, image_height
    )

    # Random upstream gradients
    torch.manual_seed(77)
    v_means2d = torch.randn(C, N, 2, device=device, dtype=dtype)
    v_cov2d_3x3 = torch.randn(C, N, 6, device=device, dtype=dtype)

    # Compute autograd gradient
    config = Config()
    config.set('dim1', image_width)
    config.set('dim0', image_height)

    params_ag = Params()
    params_ag.add_param('means', ParamWrapper(means.clone()))
    params_ag.add_param('covars', ParamWrapper(covars.clone()))
    data_ag = Data()
    data_ag.set('viewmats', viewmats.clone())
    data_ag.set('Ks', Ks.clone())
    data_ag.set('v_means2d', v_means2d.clone())
    data_ag.set('v_cov2d_3x3', v_cov2d_3x3.clone())

    ag_op = WorldToImageProjectionAutograd()
    ag_op.initialize(params_ag, config)
    _, ag_grad = ag_op.forward_backward1(params_ag, data_ag, config)
    v_means_ag = ag_grad.param_grads['means']
    v_covars_ag = ag_grad.param_grads['covars']

    # Compute numerical gradient via finite differences
    # f(theta) = v_means2d . means2d(theta) + v_cov2d_3x3 . cov2d_3x3(theta)
    def scalar_fn_means(m_flat):
        m = m_flat.view(N, 3)
        with torch.no_grad():
            means2d, cov2d = _compute_world_to_image_projection(
                m, covars, viewmats, Ks, image_width, image_height
            )
        return (v_means2d * means2d).sum() + (v_cov2d_3x3 * cov2d).sum()

    def scalar_fn_covars(cov_flat):
        cov = cov_flat.view(N, 6)
        with torch.no_grad():
            means2d, cov2d = _compute_world_to_image_projection(
                means, cov, viewmats, Ks, image_width, image_height
            )
        return (v_means2d * means2d).sum() + (v_cov2d_3x3 * cov2d).sum()

    # Numerical gradient for means
    m_flat = means.reshape(-1).clone()
    grad_means_num = torch.zeros_like(m_flat)
    for i in range(m_flat.numel()):
        m_plus = m_flat.clone(); m_plus[i] += fd_eps
        m_minus = m_flat.clone(); m_minus[i] -= fd_eps
        grad_means_num[i] = (scalar_fn_means(m_plus) - scalar_fn_means(m_minus)) / (2.0 * fd_eps)
    grad_means_num = grad_means_num.view(N, 3)

    # Numerical gradient for covars
    cov_flat = covars.reshape(-1).clone()
    grad_cov_num = torch.zeros_like(cov_flat)
    for i in range(cov_flat.numel()):
        c_plus = cov_flat.clone(); c_plus[i] += fd_eps
        c_minus = cov_flat.clone(); c_minus[i] -= fd_eps
        grad_cov_num[i] = (scalar_fn_covars(c_plus) - scalar_fn_covars(c_minus)) / (2.0 * fd_eps)
    grad_cov_num = grad_cov_num.view(N, 6)

    # Compare
    means_diff = (v_means_ag - grad_means_num).abs()
    cov_diff = (v_covars_ag - grad_cov_num).abs()

    max_means = means_diff.max().item()
    max_cov = cov_diff.max().item()

    print(f"  v_means vs numerical max diff: {max_means:.2e}")
    for i, name in enumerate(['x', 'y', 'z']):
        d = means_diff[..., i].max().item()
        print(f"    v_means_{name} max diff: {d:.2e}")

    print(f"  v_covars vs numerical max diff:  {max_cov:.2e}")
    cov_names = ['s00', 's01', 's02', 's11', 's12', 's22']
    for i, name in enumerate(cov_names):
        d = cov_diff[..., i].max().item()
        print(f"    v_covars_{name} max diff: {d:.2e}")

    atol = 1e-4
    means_ok = max_means < atol
    cov_ok = max_cov < atol

    print(f"  means numerical check: {'PASS' if means_ok else 'FAIL'} (atol={atol:.0e})")
    print(f"  covars numerical check:  {'PASS' if cov_ok else 'FAIL'} (atol={atol:.0e})")

    return means_ok and cov_ok


def test_backward2_consistency(
    C: int = 2,
    N: int = 50,
    dtype: torch.dtype = torch.float64,
    image_width: int = 640,
    image_height: int = 480,
):
    """Test that CUDA and autograd second-order backward produce the same Hessians.

    Note: CUDA uses GN approximation for H_{p,p} (drops sum_k v_k * Hess_k),
    so we compare under zero downstream first-order gradients (v_means2d=0, v_cov2d_3x3=0)
    where the GN term dominates and term2 vanishes for the position block.
    The covars block is exact regardless.
    """
    print(f"\n{'='*60}")
    print(f"Test Backward2 Consistency: C={C}, N={N}, dtype={dtype}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    means, covars, viewmats, Ks = generate_test_data(
        C, N, device, dtype, image_width, image_height
    )

    # Set v_means2d and v_cov2d_3x3 to zero so that term2 vanishes
    # This makes CUDA (GN only for means) match autograd exactly
    v_means2d = torch.zeros(C, N, 2, device=device, dtype=dtype)
    v_cov2d_3x3 = torch.zeros(C, N, 6, device=device, dtype=dtype)

    # Generate random downstream Hessians
    torch.manual_seed(555)
    h_means2d = torch.randn(C, N, 3, device=device, dtype=dtype) * 0.1
    h_cov2d_3x3 = torch.randn(C, N, 21, device=device, dtype=dtype) * 0.1
    h_means2d_cov2d_3x3 = torch.randn(C, N, 12, device=device, dtype=dtype) * 0.1
    h_means2d_opacity = torch.randn(C, N, 2, device=device, dtype=dtype) * 0.1
    h_cov2d_3x3_opacity = torch.randn(C, N, 6, device=device, dtype=dtype) * 0.1

    # Make downstream Hessians symmetric where needed
    # h_means2d (3 upper-tri of 2x2): (h00, h01, h11) - already fine
    # h_cov2d_3x3 (21 upper-tri of 6x6) - already fine
    # h_means2d_cov2d_3x3 (12 = 2x6) - no symmetry constraint

    config = Config()
    config.set('dim1', image_width)
    config.set('dim0', image_height)

    # --- CUDA backward2 ---
    params_cuda = Params()
    params_cuda.add_param('means', ParamWrapper(means.clone()))
    params_cuda.add_param('covars', ParamWrapper(covars.clone()))
    data_cuda = Data()
    data_cuda.set('viewmats', viewmats.clone())
    data_cuda.set('Ks', Ks.clone())
    data_cuda.set('v_means2d', v_means2d.clone())
    data_cuda.set('v_cov2d_3x3', v_cov2d_3x3.clone())
    data_cuda.set('h_means2d', h_means2d.clone())
    data_cuda.set('h_cov2d_3x3', h_cov2d_3x3.clone())
    data_cuda.set('h_means2d_cov2d_3x3', h_means2d_cov2d_3x3.clone())
    data_cuda.set('h_means2d_opacity', h_means2d_opacity.clone())
    data_cuda.set('h_cov2d_3x3_opacity', h_cov2d_3x3_opacity.clone())

    cuda_op = WorldToImageProjectionCUDA()
    cuda_op.initialize(params_cuda, config)
    cuda_fwd, cuda_grad, cuda_hess = cuda_op.forward_backward2(
        params_cuda, data_cuda, config
    )

    # --- Autograd backward2 ---
    params_ag = Params()
    params_ag.add_param('means', ParamWrapper(means.clone()))
    params_ag.add_param('covars', ParamWrapper(covars.clone()))
    data_ag = Data()
    data_ag.set('viewmats', viewmats.clone())
    data_ag.set('Ks', Ks.clone())
    data_ag.set('v_means2d', v_means2d.clone())
    data_ag.set('v_cov2d_3x3', v_cov2d_3x3.clone())
    data_ag.set('h_means2d', h_means2d.clone())
    data_ag.set('h_cov2d_3x3', h_cov2d_3x3.clone())
    data_ag.set('h_means2d_cov2d_3x3', h_means2d_cov2d_3x3.clone())
    data_ag.set('h_means2d_opacity', h_means2d_opacity.clone())
    data_ag.set('h_cov2d_3x3_opacity', h_cov2d_3x3_opacity.clone())

    ag_op = WorldToImageProjectionAutograd()
    ag_op.initialize(params_ag, config)
    ag_fwd, ag_grad, ag_hess = ag_op.forward_backward2(params_ag, data_ag, config)

    # Compare first-order gradients
    v_m_diff = (cuda_grad.param_grads['means'] - ag_grad.param_grads['means']).abs().max().item()
    v_cov_diff = (cuda_grad.param_grads['covars'] - ag_grad.param_grads['covars']).abs().max().item()
    print(f"  First-order v_means max diff: {v_m_diff:.2e}")
    print(f"  First-order v_covars max diff:  {v_cov_diff:.2e}")

    # Compare Hessian blocks
    atol = 5e-4 if dtype == torch.float64 else 1e-2
    all_ok = True

    hess_keys = [
        ('means', 6),
        ('covars', 21),
        ('means_covars', 18),
        ('means_opacity', 3),
        ('covars_opacity', 6),
    ]

    for key, size in hess_keys:
        h_cuda = cuda_hess.param_hess[key]
        h_ag = ag_hess.param_hess[key]

        diff = (h_cuda - h_ag).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # Relative error where values are nonzero
        denom = h_ag.abs().clamp(min=1e-10)
        rel_diff = (diff / denom).max().item()

        # Use both absolute and relative tolerance (like torch.allclose)
        ok = torch.allclose(h_cuda, h_ag, atol=atol, rtol=1e-10)
        all_ok = all_ok and ok

        print(f"  H_{key}:")
        print(f"    max abs diff:  {max_diff:.2e}")
        print(f"    mean abs diff: {mean_diff:.2e}")
        print(f"    max rel diff:  {rel_diff:.2e}")
        print(f"    match: {'PASS' if ok else 'FAIL'} (atol={atol:.0e}, rtol=1e-10)")

    return all_ok


def test_backward2_with_nonzero_v(
    C: int = 1,
    N: int = 20,
    dtype: torch.dtype = torch.float64,
    image_width: int = 640,
    image_height: int = 480,
):
    """Test backward2 with nonzero upstream gradients.

    CUDA uses GN approximation for H_{p,p}, so the position Hessian
    may differ from autograd (which computes the exact Hessian).
    The covars and cross-Hessian blocks should still match closely.

    This test reports differences without strict PASS/FAIL for H_{p,p}.
    """
    print(f"\n{'='*60}")
    print(f"Test Backward2 with Nonzero v: C={C}, N={N}, dtype={dtype}")
    print(f"  (CUDA uses GN approx for H_pp; covars & cross should match)")
    print(f"{'='*60}")

    device = torch.device("cuda")
    means, covars, viewmats, Ks = generate_test_data(
        C, N, device, dtype, image_width, image_height
    )

    # Nonzero upstream gradients
    torch.manual_seed(888)
    v_means2d = torch.randn(C, N, 2, device=device, dtype=dtype) * 0.5
    v_cov2d_3x3 = torch.randn(C, N, 6, device=device, dtype=dtype) * 0.5

    # Random downstream Hessians
    torch.manual_seed(777)
    h_means2d = torch.randn(C, N, 3, device=device, dtype=dtype) * 0.1
    h_cov2d_3x3 = torch.randn(C, N, 21, device=device, dtype=dtype) * 0.1
    h_means2d_cov2d_3x3 = torch.randn(C, N, 12, device=device, dtype=dtype) * 0.1
    h_means2d_opacity = torch.randn(C, N, 2, device=device, dtype=dtype) * 0.1
    h_cov2d_3x3_opacity = torch.randn(C, N, 6, device=device, dtype=dtype) * 0.1

    config = Config()
    config.set('dim1', image_width)
    config.set('dim0', image_height)

    # CUDA
    params_cuda = Params()
    params_cuda.add_param('means', ParamWrapper(means.clone()))
    params_cuda.add_param('covars', ParamWrapper(covars.clone()))
    data_cuda = Data()
    for k, v in [('viewmats', viewmats), ('Ks', Ks),
                  ('v_means2d', v_means2d), ('v_cov2d_3x3', v_cov2d_3x3),
                  ('h_means2d', h_means2d), ('h_cov2d_3x3', h_cov2d_3x3),
                  ('h_means2d_cov2d_3x3', h_means2d_cov2d_3x3),
                  ('h_means2d_opacity', h_means2d_opacity),
                  ('h_cov2d_3x3_opacity', h_cov2d_3x3_opacity)]:
        data_cuda.set(k, v.clone())

    cuda_op = WorldToImageProjectionCUDA()
    cuda_op.initialize(params_cuda, config)
    _, _, cuda_hess = cuda_op.forward_backward2(params_cuda, data_cuda, config)

    # Autograd
    params_ag = Params()
    params_ag.add_param('means', ParamWrapper(means.clone()))
    params_ag.add_param('covars', ParamWrapper(covars.clone()))
    data_ag = Data()
    for k, v in [('viewmats', viewmats), ('Ks', Ks),
                  ('v_means2d', v_means2d), ('v_cov2d_3x3', v_cov2d_3x3),
                  ('h_means2d', h_means2d), ('h_cov2d_3x3', h_cov2d_3x3),
                  ('h_means2d_cov2d_3x3', h_means2d_cov2d_3x3),
                  ('h_means2d_opacity', h_means2d_opacity),
                  ('h_cov2d_3x3_opacity', h_cov2d_3x3_opacity)]:
        data_ag.set(k, v.clone())

    ag_op = WorldToImageProjectionAutograd()
    ag_op.initialize(params_ag, config)
    _, _, ag_hess = ag_op.forward_backward2(params_ag, data_ag, config)

    all_ok = True
    for key in ['means', 'covars', 'means_covars',
                'means_opacity', 'covars_opacity']:
        h_cuda = cuda_hess.param_hess[key]
        h_ag = ag_hess.param_hess[key]
        diff = (h_cuda - h_ag).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        # All blocks now use exact Hessian computation (GN Term 1 + Term 2).
        # means (H_pp) and means_covars (H_ps) are no longer GN-only.
        is_gn_approx = False
        if dtype == torch.float64:
            atol_key = 5e-4
        else:
            atol_key = 1e-2  # f32 may have higher numerical error for Hessians
        ok = max_diff < atol_key

        label = f"H_{key}"
        print(f"  {label}:")
        print(f"    max abs diff:  {max_diff:.2e}")
        print(f"    mean abs diff: {mean_diff:.2e}")
        print(f"    match: {'PASS' if ok else 'FAIL'}  (atol={atol_key:.0e})")

        all_ok = all_ok and ok

    return all_ok


def test_factory_function():
    """Test the create_world_to_image_projection factory function."""
    print(f"\n{'='*60}")
    print(f"Test Factory Function")
    print(f"{'='*60}")

    try:
        cuda_op = create_world_to_image_projection("cuda")
        assert isinstance(cuda_op, WorldToImageProjectionCUDA)
        print("  create_world_to_image_projection('cuda'):    PASS")
    except Exception as e:
        print(f"  create_world_to_image_projection('cuda'):    FAIL ({e})")
        return False

    try:
        ag_op = create_world_to_image_projection("autograd")
        assert isinstance(ag_op, WorldToImageProjectionAutograd)
        print("  create_world_to_image_projection('autograd'): PASS")
    except Exception as e:
        print(f"  create_world_to_image_projection('autograd'): FAIL ({e})")
        return False

    try:
        create_world_to_image_projection("invalid")
        print("  create_world_to_image_projection('invalid') should raise: FAIL")
        return False
    except ValueError:
        print("  create_world_to_image_projection('invalid') raises ValueError: PASS")

    return True


def test_edge_cases(dtype: torch.dtype = torch.float32):
    """Test edge cases: single Gaussian, behind camera, large batch."""
    print(f"\n{'='*60}")
    print(f"Test Edge Cases")
    print(f"{'='*60}")

    device = torch.device("cuda")
    all_ok = True
    image_width, image_height = 640, 480

    config = Config()
    config.set('dim1', image_width)
    config.set('dim0', image_height)

    # Test 1: Single Gaussian (C=1, N=1)
    print("  [1] Single Gaussian (C=1, N=1)...")
    means, covars, viewmats, Ks = generate_test_data(
        1, 1, device, dtype, image_width, image_height
    )
    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('covars', ParamWrapper(covars))
    data = Data()
    data.set('viewmats', viewmats)
    data.set('Ks', Ks)

    cuda_op = WorldToImageProjectionCUDA()
    cuda_op.initialize(params, config)
    result = cuda_op.forward_only(params, data, config)

    means2d = result.outputs['means2d']
    cov2d = result.outputs['cov2d_3x3']

    ok = (means2d.shape == (1, 1, 2)
          and cov2d.shape == (1, 1, 6)
          and torch.isfinite(means2d).all().item()
          and torch.isfinite(cov2d).all().item())
    print(f"    Shape & finite check: {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok

    # Test 2: Behind camera (z <= 0)
    print("  [2] Behind camera (should output zeros)...")
    means_behind = torch.tensor([[0.0, 0.0, -5.0]], device=device, dtype=dtype)
    covars_behind = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device=device, dtype=dtype)
    # Create simple viewmat (identity rotation, zero translation)
    viewmat_id = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
    K_simple = torch.tensor([[[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]]], device=device, dtype=dtype)

    params_b = Params()
    params_b.add_param('means', ParamWrapper(means_behind))
    params_b.add_param('covars', ParamWrapper(covars_behind))
    data_b = Data()
    data_b.set('viewmats', viewmat_id)
    data_b.set('Ks', K_simple)

    cuda_op2 = WorldToImageProjectionCUDA()
    cuda_op2.initialize(params_b, config)
    result_b = cuda_op2.forward_only(params_b, data_b, config)

    means2d_b = result_b.outputs['means2d']
    cov2d_b = result_b.outputs['cov2d_3x3']

    ok_behind = (means2d_b.abs().max().item() < 1e-6
                 and cov2d_b.abs().max().item() < 1e-6)
    print(f"    Zeros for behind camera: {'PASS' if ok_behind else 'FAIL'}")
    all_ok = all_ok and ok_behind

    # Test 3: Larger batch
    print("  [3] Larger batch (C=4, N=500)...")
    means_l, covars_l, viewmats_l, Ks_l = generate_test_data(
        4, 500, device, dtype, image_width, image_height, seed=100
    )
    params_l = Params()
    params_l.add_param('means', ParamWrapper(means_l))
    params_l.add_param('covars', ParamWrapper(covars_l))
    data_l = Data()
    data_l.set('viewmats', viewmats_l)
    data_l.set('Ks', Ks_l)

    cuda_op3 = WorldToImageProjectionCUDA()
    cuda_op3.initialize(params_l, config)
    result_l = cuda_op3.forward_only(params_l, data_l, config)

    means2d_l = result_l.outputs['means2d']
    cov2d_l = result_l.outputs['cov2d_3x3']

    ok_l = (means2d_l.shape == (4, 500, 2)
            and cov2d_l.shape == (4, 500, 6)
            and torch.isfinite(means2d_l).all().item()
            and torch.isfinite(cov2d_l).all().item())
    print(f"    Shape & finite check: {'PASS' if ok_l else 'FAIL'}")
    all_ok = all_ok and ok_l

    # Test 4: FOV clipping (extreme off-center Gaussians)
    print("  [4] FOV clipping (extreme positions)...")
    means_clip = torch.tensor([[100.0, 100.0, 1.0]], device=device, dtype=dtype)
    covars_clip = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device=device, dtype=dtype)

    params_c = Params()
    params_c.add_param('means', ParamWrapper(means_clip))
    params_c.add_param('covars', ParamWrapper(covars_clip))
    data_c = Data()
    data_c.set('viewmats', viewmat_id)
    data_c.set('Ks', K_simple)

    cuda_op4 = WorldToImageProjectionCUDA()
    cuda_op4.initialize(params_c, config)
    result_c = cuda_op4.forward_only(params_c, data_c, config)

    cov2d_c = result_c.outputs['cov2d_3x3']
    ok_clip = torch.isfinite(cov2d_c).all().item()
    print(f"    Finite check under clipping: {'PASS' if ok_clip else 'FAIL'}")
    all_ok = all_ok and ok_clip

    return all_ok


def main():
    """Run all tests."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        return

    print("=" * 60)
    print("WorldToImageProjection Module Test Suite")
    print("=" * 60)

    results = {}

    # Test 1: Forward consistency
    results['forward_f32'] = test_forward_consistency(
        C=4, N=500, dtype=torch.float32
    )
    results['forward_f64'] = test_forward_consistency(
        C=4, N=500, dtype=torch.float64
    )

    # Test 2: Backward consistency (CUDA vs autograd)
    results['backward_f32'] = test_backward_consistency(
        C=4, N=500, dtype=torch.float32
    )
    results['backward_f64'] = test_backward_consistency(
        C=4, N=500, dtype=torch.float64
    )

    # Test 3: Backward vs numerical (finite differences)
    results['numerical'] = test_backward_vs_numerical(
        C=1, N=5, dtype=torch.float64
    )

    # Test 4: Backward2 consistency (v=0, no GN difference)
    results['backward2_v0'] = test_backward2_consistency(
        C=2, N=50, dtype=torch.float64
    )

    # Test 5: Backward2 with nonzero v (GN approximation)
    results['backward2_vnonzero'] = test_backward2_with_nonzero_v(
        C=1, N=20, dtype=torch.float64
    )

    # Test 6: Factory function
    results['factory'] = test_factory_function()

    # Test 7: Edge cases
    results['edge_cases'] = test_edge_cases()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s}: {status}")
        if not passed:
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
