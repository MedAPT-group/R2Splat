"""
Test script for Full Projection and Rasterization Pipeline.

This script validates the CUDA chain implementation against PyTorch autograd
for the full pipeline:
    WorldToImageProjection → ConicsMu → RasterizeToPixels → ImageLoss

When lambda_ssim=0 (L2 loss only), the results should match exactly.

Test stages:
1. Forward consistency: CUDA vs Autograd render output and loss
2. First-order backward consistency: Gradients match
3. Second-order backward consistency: Hessians match
4. Loss value sanity check
5. Gradient vs finite difference validation

Usage:
    python tests/test_projection_and_rasterization.py

Author: Dy164
Date: 2026-02-20
"""

import torch
import math
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.data import Params, Data, Config
from gsplat2ct.ops.projection_and_rasterization import (
    ProjectionAndRasterizationCUDA,
    ProjectionAndRasterizationAutograd,
    create_projection_and_rasterization,
)


# ============================================================================
# Utility: ParamWrapper
# ============================================================================

class ParamWrapper:
    """Simple wrapper to satisfy ParameterLike protocol."""
    def __init__(self, data):
        self.data = data


# ============================================================================
# Data Generation Utilities
# ============================================================================

def generate_random_spd_covars(
    N: int,
    device: torch.device, dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """Generate SPD 3x3 covariance in upper-tri storage via Cholesky.

    Returns (N, 6): (s00, s01, s02, s11, s12, s22).
    """
    torch.manual_seed(seed)
    l00 = torch.rand(N, device=device, dtype=dtype) * 0.5 + 0.1
    l10 = torch.randn(N, device=device, dtype=dtype) * 0.2
    l11 = torch.rand(N, device=device, dtype=dtype) * 0.5 + 0.1
    l20 = torch.randn(N, device=device, dtype=dtype) * 0.2
    l21 = torch.randn(N, device=device, dtype=dtype) * 0.2
    l22 = torch.rand(N, device=device, dtype=dtype) * 0.5 + 0.1

    s00 = l00 * l00
    s01 = l00 * l10
    s02 = l00 * l20
    s11 = l10 * l10 + l11 * l11
    s12 = l10 * l20 + l11 * l21
    s22 = l20 * l20 + l21 * l21 + l22 * l22

    return torch.stack([s00, s01, s02, s11, s12, s22], dim=-1)


def generate_random_viewmats(
    C: int,
    device: torch.device, dtype: torch.dtype,
    seed: int = 43,
) -> torch.Tensor:
    """Generate random valid view matrices (C, 4, 4) via QR."""
    torch.manual_seed(seed)
    viewmats = torch.zeros(C, 4, 4, device=device, dtype=dtype)
    viewmats[:, 3, 3] = 1.0

    for c in range(C):
        A = torch.randn(3, 3, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A)
        if torch.det(Q) < 0:
            Q[:, 0] *= -1
        viewmats[c, :3, :3] = Q
        viewmats[c, :3, 3] = torch.randn(3, device=device, dtype=dtype) * 2.0

    return viewmats


def generate_random_intrinsics(
    C: int,
    device: torch.device, dtype: torch.dtype,
    image_width: int, image_height: int,
    seed: int = 44,
) -> torch.Tensor:
    """Generate random camera intrinsics (C, 3, 3)."""
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
    N: int = 5,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    image_width: int = 32,
    image_height: int = 32,
    seed: int = 42,
):
    """Generate a complete test dataset for the integration pipeline.

    Parameters are chosen so that:
    - All Gaussians are in front of the camera (z > 3)
    - Gaussians project within the image
    - Covariances are well-conditioned SPD matrices
    - Opacities are moderate (|opacity * mu| not too large)
    - Small image size to keep the test fast
    """
    if device is None:
        device = torch.device("cuda")

    viewmats = generate_random_viewmats(C, device, dtype, seed=seed + 1)
    Ks = generate_random_intrinsics(C, device, dtype, image_width, image_height,
                                     seed=seed + 2)

    # Generate camera-space points with z > 3 (for first camera)
    torch.manual_seed(seed + 3)
    p_c = torch.randn(N, 3, device=device, dtype=dtype) * 0.5
    p_c[:, 2] = p_c[:, 2].abs() + 3.0

    # Transform from camera space back to world space using the first camera
    R = viewmats[0, :3, :3]
    t = viewmats[0, :3, 3]
    Rt = R.T
    means = torch.einsum('ij,nj->ni', Rt, p_c - t.unsqueeze(0)).contiguous()

    covars = generate_random_spd_covars(N, device, dtype, seed=seed + 4)

    # Opacity: moderate values
    torch.manual_seed(seed + 5)
    opacity = torch.rand(C, N, device=device, dtype=dtype) * 0.2 - 0.1  # [-0.1, 0.1]

    # Target: just zeros (the actual target doesn't matter much for gradient validation)
    target = torch.zeros(C, image_height, image_width, device=device, dtype=dtype)

    return means, covars, opacity, viewmats, Ks, target


# ============================================================================
# Test Functions
# ============================================================================

def test_forward_consistency(
    C: int = 1,
    N: int = 5,
    dim0: int = 32,
    dim1: int = 32,
    eps: float = 0.3,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-4,
) -> bool:
    """Test 1: Forward pass consistency between CUDA and Autograd."""
    print("\n" + "=" * 70)
    print("TEST 1: Forward Pass Consistency (CUDA vs Autograd)")
    print("=" * 70)

    device = torch.device(device)
    means, covars, opacity, viewmats, Ks, target = generate_test_data(
        C, N, device, dtype, dim1, dim0
    )

    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('covars', ParamWrapper(covars))
    params.add_param('opacity', ParamWrapper(opacity))

    config = Config()
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps_blur', eps)
    config.set('lambda_l2', 1.0)
    config.set('lambda_ssim', 0.0)

    data_obj = Data()
    data_obj.set('viewmats', viewmats)
    data_obj.set('Ks', Ks)
    data_obj.set('target', target)

    # CUDA forward
    cuda_op = ProjectionAndRasterizationCUDA()
    cuda_op.initialize(params, config)
    cuda_fwd = cuda_op.forward_only(params, data_obj, config)
    loss_cuda = cuda_fwd.outputs['loss']
    render_cuda = cuda_fwd.outputs['render_output']

    # Autograd forward
    ag_op = ProjectionAndRasterizationAutograd()
    ag_op.initialize(params, config)
    ag_fwd = ag_op.forward_only(params, data_obj, config)
    loss_ag = ag_fwd.outputs['loss']
    render_ag = ag_fwd.outputs['render_output']

    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, eps={eps}")
    print(f"CUDA loss:     {loss_cuda.item():.6f}")
    print(f"Autograd loss: {loss_ag.item():.6f}")

    # Compare
    render_diff = (render_cuda - render_ag).abs()
    max_render_diff = render_diff.max().item()
    loss_diff = abs(loss_cuda.item() - loss_ag.item())

    print(f"\nRender output max diff: {max_render_diff:.6e}")
    print(f"Loss diff: {loss_diff:.6e}")

    render_pass = max_render_diff < rtol
    loss_pass = loss_diff < rtol

    status = "✓" if (render_pass and loss_pass) else "✗"
    print(f"\n{status} TEST 1 {'PASSED' if (render_pass and loss_pass) else 'FAILED'}")

    return render_pass and loss_pass


def test_backward1_consistency(
    C: int = 1,
    N: int = 5,
    dim0: int = 32,
    dim1: int = 32,
    eps: float = 0.3,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.05,
) -> bool:
    """Test 2: First-order backward consistency."""
    print("\n" + "=" * 70)
    print("TEST 2: First-Order Backward Consistency (CUDA vs Autograd)")
    print("=" * 70)

    device = torch.device(device)
    means, covars, opacity, viewmats, Ks, target = generate_test_data(
        C, N, device, dtype, dim1, dim0
    )

    config = Config()
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps_blur', eps)
    config.set('lambda_l2', 1.0)
    config.set('lambda_ssim', 0.0)

    data_obj = Data()
    data_obj.set('viewmats', viewmats)
    data_obj.set('Ks', Ks)
    data_obj.set('target', target)

    # CUDA
    params_cuda = Params()
    params_cuda.add_param('means', ParamWrapper(means.clone()))
    params_cuda.add_param('covars', ParamWrapper(covars.clone()))
    params_cuda.add_param('opacity', ParamWrapper(opacity.clone()))

    cuda_op = ProjectionAndRasterizationCUDA()
    cuda_op.initialize(params_cuda, config)
    _, cuda_grad = cuda_op.forward_backward1(params_cuda, data_obj, config)

    # Autograd
    params_ag = Params()
    params_ag.add_param('means', ParamWrapper(means.clone()))
    params_ag.add_param('covars', ParamWrapper(covars.clone()))
    params_ag.add_param('opacity', ParamWrapper(opacity.clone()))

    ag_op = ProjectionAndRasterizationAutograd()
    ag_op.initialize(params_ag, config)
    _, ag_grad = ag_op.forward_backward1(params_ag, data_obj, config)

    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, eps={eps}")

    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = max(t1.abs().max().item(), t2.abs().max().item(), 1e-10)
        rel_error = max_diff / scale
        passed = rel_error < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: max_diff={max_diff:.6e}, scale={scale:.6e}, rel_error={rel_error:.4f} {status}")
        return passed

    all_pass = True
    all_pass &= compare_tensors("v_means   ", cuda_grad.param_grads['means'], ag_grad.param_grads['means'])
    all_pass &= compare_tensors("v_covars  ", cuda_grad.param_grads['covars'], ag_grad.param_grads['covars'])
    all_pass &= compare_tensors("v_opacity ", cuda_grad.param_grads['opacity'], ag_grad.param_grads['opacity'])

    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 2 {'PASSED' if all_pass else 'FAILED'}")

    return all_pass


def test_backward2_consistency(
    C: int = 1,
    N: int = 3,
    dim0: int = 32,
    dim1: int = 32,
    eps: float = 0.3,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.15,
) -> bool:
    """Test 3: Second-order backward consistency."""
    print("\n" + "=" * 70)
    print("TEST 3: Second-Order Backward Consistency (CUDA vs Autograd)")
    print("=" * 70)

    device = torch.device(device)
    means, covars, opacity, viewmats, Ks, target = generate_test_data(
        C, N, device, dtype, dim1, dim0
    )

    config = Config()
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps_blur', eps)
    config.set('lambda_l2', 1.0)
    config.set('lambda_ssim', 0.0)

    data_obj = Data()
    data_obj.set('viewmats', viewmats)
    data_obj.set('Ks', Ks)
    data_obj.set('target', target)

    # CUDA
    params_cuda = Params()
    params_cuda.add_param('means', ParamWrapper(means.clone()))
    params_cuda.add_param('covars', ParamWrapper(covars.clone()))
    params_cuda.add_param('opacity', ParamWrapper(opacity.clone()))

    cuda_op = ProjectionAndRasterizationCUDA()
    cuda_op.initialize(params_cuda, config)
    _, cuda_grad, cuda_hess = cuda_op.forward_backward2(params_cuda, data_obj, config)

    # Autograd
    params_ag = Params()
    params_ag.add_param('means', ParamWrapper(means.clone()))
    params_ag.add_param('covars', ParamWrapper(covars.clone()))
    params_ag.add_param('opacity', ParamWrapper(opacity.clone()))

    ag_op = ProjectionAndRasterizationAutograd()
    ag_op.initialize(params_ag, config)
    _, ag_grad, ag_hess = ag_op.forward_backward2(params_ag, data_obj, config)

    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, eps={eps}")
    print(f"Using lambda_l2=1.0, lambda_ssim=0.0 (L2 loss only)")

    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        if t1 is None or t2 is None:
            print(f"  {name}: SKIPPED (None)")
            return True
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = max(t1.abs().max().item(), t2.abs().max().item(), 1e-10)
        rel_error = max_diff / scale
        passed = rel_error < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: max_diff={max_diff:.6e}, scale={scale:.6e}, rel_error={rel_error:.4f} {status}")
        if not passed:
            print(f"    t1 range: [{t1.min().item():.6e}, {t1.max().item():.6e}]")
            print(f"    t2 range: [{t2.min().item():.6e}, {t2.max().item():.6e}]")
        return passed

    print("\n--- First-Order Gradients ---")
    all_pass = True
    all_pass &= compare_tensors("v_means   ", cuda_grad.param_grads['means'], ag_grad.param_grads['means'])
    all_pass &= compare_tensors("v_covars  ", cuda_grad.param_grads['covars'], ag_grad.param_grads['covars'])
    all_pass &= compare_tensors("v_opacity ", cuda_grad.param_grads['opacity'], ag_grad.param_grads['opacity'])

    print("\n--- Self Hessians ---")
    all_pass &= compare_tensors("h_means          ", cuda_hess.param_hess['means'], ag_hess.param_hess['means'])
    all_pass &= compare_tensors("h_covars         ", cuda_hess.param_hess['covars'], ag_hess.param_hess['covars'])
    all_pass &= compare_tensors("h_opacity        ", cuda_hess.param_hess['opacity'], ag_hess.param_hess['opacity'])

    print("\n--- Cross-Term Hessians ---")
    all_pass &= compare_tensors("h_means_covars   ", cuda_hess.param_hess['means_covars'], ag_hess.param_hess['means_covars'])
    all_pass &= compare_tensors("h_means_opacity  ", cuda_hess.param_hess['means_opacity'], ag_hess.param_hess['means_opacity'])
    all_pass &= compare_tensors("h_covars_opacity ", cuda_hess.param_hess['covars_opacity'], ag_hess.param_hess['covars_opacity'])

    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 3 {'PASSED' if all_pass else 'FAILED'}")

    return all_pass


def test_loss_value_sanity(
    C: int = 1,
    N: int = 10,
    dim0: int = 64,
    dim1: int = 64,
    eps: float = 0.3,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> bool:
    """Test 4: Verify loss value is reasonable."""
    print("\n" + "=" * 70)
    print("TEST 4: Loss Value Sanity Check")
    print("=" * 70)

    device = torch.device(device)
    means, covars, opacity, viewmats, Ks, target = generate_test_data(
        C, N, device, dtype, dim1, dim0
    )

    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('covars', ParamWrapper(covars))
    params.add_param('opacity', ParamWrapper(opacity))

    config = Config()
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps_blur', eps)
    config.set('lambda_l2', 1.0)
    config.set('lambda_ssim', 0.0)

    data_obj = Data()
    data_obj.set('viewmats', viewmats)
    data_obj.set('Ks', Ks)
    data_obj.set('target', target)

    cuda_op = ProjectionAndRasterizationCUDA()
    cuda_op.initialize(params, config)
    result = cuda_op.forward_only(params, data_obj, config)

    loss = result.outputs['loss'].item()
    render_output = result.outputs['render_output']

    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, eps={eps}")
    print(f"Loss: {loss:.6f}")
    print(f"Render output range: [{render_output.min().item():.4f}, {render_output.max().item():.4f}]")
    print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")

    is_finite = np.isfinite(loss)
    is_positive = loss >= 0

    status = "✓" if (is_finite and is_positive) else "✗"
    print(f"\n{status} TEST 4 {'PASSED' if (is_finite and is_positive) else 'FAILED'}")

    return is_finite and is_positive


def test_gradient_finite_difference(
    C: int = 1,
    N: int = 3,
    dim0: int = 16,
    dim1: int = 16,
    eps: float = 0.3,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
    fd_eps: float = 1e-4,
    rtol: float = 0.05,
) -> bool:
    """Test 5: Verify autograd gradients against finite differences.

    Uses float64 and per-element perturbation for accurate FD comparison.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Gradient vs Finite Difference Validation")
    print("=" * 70)

    device = torch.device(device)
    means, covars, opacity, viewmats, Ks, target = generate_test_data(
        C, N, device, dtype, dim1, dim0, seed=99
    )

    config = Config()
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps_blur', eps)
    config.set('lambda_l2', 1.0)
    config.set('lambda_ssim', 0.0)

    def compute_loss(m, cov, opa):
        """Helper: compute loss via autograd pipeline."""
        p = Params()
        p.add_param('means', ParamWrapper(m))
        p.add_param('covars', ParamWrapper(cov))
        p.add_param('opacity', ParamWrapper(opa))
        d = Data()
        d.set('viewmats', viewmats)
        d.set('Ks', Ks)
        d.set('target', target)
        op = ProjectionAndRasterizationAutograd()
        op.initialize(p, config)
        fwd = op.forward_only(p, d, config)
        return fwd.outputs['loss'].item()

    # Compute autograd gradients
    params = Params()
    params.add_param('means', ParamWrapper(means.clone()))
    params.add_param('covars', ParamWrapper(covars.clone()))
    params.add_param('opacity', ParamWrapper(opacity.clone()))
    data_obj = Data()
    data_obj.set('viewmats', viewmats)
    data_obj.set('Ks', Ks)
    data_obj.set('target', target)

    ag_op = ProjectionAndRasterizationAutograd()
    ag_op.initialize(params, config)
    _, ag_grad = ag_op.forward_backward1(params, data_obj, config)

    ag_v_means = ag_grad.param_grads['means']
    ag_v_covars = ag_grad.param_grads['covars']
    ag_v_opacity = ag_grad.param_grads['opacity']

    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, dtype=float64")

    # Per-element FD for means
    fd_v_means = torch.zeros_like(means)
    for n in range(N):
        for i in range(3):
            m_p = means.clone(); m_p[n, i] += fd_eps
            m_m = means.clone(); m_m[n, i] -= fd_eps
            fd_v_means[n, i] = (compute_loss(m_p, covars, opacity) -
                                 compute_loss(m_m, covars, opacity)) / (2 * fd_eps)

    # Per-element FD for covars
    fd_v_covars = torch.zeros_like(covars)
    for n in range(N):
        for i in range(6):
            c_p = covars.clone(); c_p[n, i] += fd_eps
            c_m = covars.clone(); c_m[n, i] -= fd_eps
            fd_v_covars[n, i] = (compute_loss(means, c_p, opacity) -
                                  compute_loss(means, c_m, opacity)) / (2 * fd_eps)

    # Per-element FD for opacity
    fd_v_opacity = torch.zeros_like(opacity)
    for c in range(C):
        for n in range(N):
            o_p = opacity.clone(); o_p[c, n] += fd_eps
            o_m = opacity.clone(); o_m[c, n] -= fd_eps
            fd_v_opacity[c, n] = (compute_loss(means, covars, o_p) -
                                   compute_loss(means, covars, o_m)) / (2 * fd_eps)

    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = max(t1.abs().max().item(), t2.abs().max().item(), 1e-10)
        rel_error = max_diff / scale
        passed = rel_error < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: max_diff={max_diff:.6e}, rel_error={rel_error:.4f} {status}")
        return passed

    all_pass = True
    all_pass &= compare_tensors("d_means    (autograd vs FD)", ag_v_means, fd_v_means)
    all_pass &= compare_tensors("d_covars   (autograd vs FD)", ag_v_covars, fd_v_covars)
    all_pass &= compare_tensors("d_opacity  (autograd vs FD)", ag_v_opacity, fd_v_opacity)

    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 5 {'PASSED' if all_pass else 'FAILED'}")

    return all_pass


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("ProjectionAndRasterization (Full Pipeline) Test Suite")
    print("=" * 70)

    results = {}

    # Test 1: Forward consistency
    results['forward'] = test_forward_consistency(C=3, N=29, dim0=33, dim1=13, eps=1e-4)

    # Test 2: First-order backward
    results['backward1'] = test_backward1_consistency(C=3, N=29, dim0=13, dim1=33, eps=1e-4)

    # Test 3: Second-order backward
    results['backward2'] = test_backward2_consistency(C=3, N=29, dim0=13, dim1=33, eps=1e-4)

    # Test 4: Loss sanity
    results['loss_sanity'] = test_loss_value_sanity()

    # Test 5: Gradient vs finite difference
    results['grad_vs_fd'] = test_gradient_finite_difference()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_pass = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        all_pass &= passed

    print("\n" + "=" * 70)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    run_all_tests()
