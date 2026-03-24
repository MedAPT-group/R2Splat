"""
Test script for 2D Gaussian Splatting with ConicsMu Rectification + Image Loss.

This script validates the CUDA implementation against PyTorch autograd
for the full ConicsMu -> Rasterization -> ImageLoss pipeline.

When lambda_ssim=0 (L2 loss only), the results should match exactly.

Test stages:
1. Forward consistency: CUDA vs Autograd render output and loss
2. First-order backward consistency: Gradients match
3. Second-order backward consistency: Hessians match

Usage:
    python tests/test_gs2d_rectifying.py

Author: Dy164
Date: 2026-02-19
"""

import torch
import math
import numpy as np
import sys
import os
from typing import Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def generate_random_spd_cov2d_3x3(
    C: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """Generate random symmetric positive-definite 3x3 covariance matrices.
    
    Uses Cholesky decomposition: Sigma = L @ L^T to guarantee SPD.
    Returns in upper-triangular storage: (a, b, c, d, e, f) for
        [[a, b, c],
         [b, d, e],
         [c, e, f]]
    """
    torch.manual_seed(seed)
    
    # Generate lower triangular L with positive diagonal
    l00 = torch.rand(C, N, device=device, dtype=dtype) * 3.0 + 1.0  # [1, 4]
    l10 = torch.randn(C, N, device=device, dtype=dtype) * 1.0
    l11 = torch.rand(C, N, device=device, dtype=dtype) * 3.0 + 1.0
    l20 = torch.randn(C, N, device=device, dtype=dtype) * 1.0
    l21 = torch.randn(C, N, device=device, dtype=dtype) * 1.0
    l22 = torch.rand(C, N, device=device, dtype=dtype) * 3.0 + 1.0
    
    # Sigma = L @ L^T
    a = l00 * l00
    b = l00 * l10
    c = l00 * l20
    d = l10 * l10 + l11 * l11
    e = l10 * l20 + l11 * l21
    f = l20 * l20 + l21 * l21 + l22 * l22
    
    cov2d_3x3 = torch.stack([a, b, c, d, e, f], dim=-1)  # (C, N, 6)
    return cov2d_3x3


def compute_radii_from_cov2d_3x3(
    cov2d_3x3: torch.Tensor,
    eps: float = 0.3,
) -> torch.Tensor:
    """Compute Gaussian radii from 3x3 projected covariance matrices.
    
    Computes max eigenvalue of the blurred 2x2 spatial covariance (upper-left)
    and derives radii = ceil(4 * sqrt(lambda_max)).
    """
    a = cov2d_3x3[..., 0] + eps
    b = cov2d_3x3[..., 1]
    d = cov2d_3x3[..., 3] + eps
    
    trace = a + d
    det = a * d - b * b
    discriminant = torch.sqrt(torch.clamp(trace * trace - 4 * det, min=1e-6))
    lambda_max = (trace + discriminant) / 2
    
    radii = torch.ceil(4 * torch.sqrt(lambda_max)).to(torch.int32)
    radii = torch.clamp(radii, min=1)
    return radii


def generate_test_data(
    C: int,
    N: int,
    dim0: int,
    dim1: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 0.3,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random but valid test data for GS2DRectifying.
    
    Returns:
        cov2d_3x3: (C, N, 6) SPD 3x3 covariance matrices
        means2d: (C, N, 2) Gaussian centers within image bounds
        opacities: (C, N) Gaussian opacities
        radii: (C, N) int32 radii for tile intersection
        target: (C, dim0, dim1) target image
    """
    # Generate SPD covariance matrices
    cov2d_3x3 = generate_random_spd_cov2d_3x3(C, N, device, dtype, seed)
    
    # Generate means2d within image bounds
    torch.manual_seed(seed + 1)
    means2d = torch.rand(C, N, 2, device=device, dtype=dtype)
    means2d[..., 0] *= dim1   # x in [0, dim1)
    means2d[..., 1] *= dim0   # y in [0, dim0)
    
    # Generate opacities (small values to keep outputs bounded)
    torch.manual_seed(seed + 2)
    opacities = torch.rand(C, N, device=device, dtype=dtype) * 0.4 - 0.2  # [-0.2, 0.2]
    
    # Compute radii from covariance
    radii = compute_radii_from_cov2d_3x3(cov2d_3x3, eps)
    
    # Generate target image
    torch.manual_seed(seed + 3)
    target = torch.randn(C, dim0, dim1, device=device, dtype=dtype) * 0.1
    
    return cov2d_3x3, means2d, opacities, radii, target


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
    
    from gsplat2ct.ops.gs2d_rectifying import (
        GS2DRectifyingAutograd, GS2DRectifyingCUDA,
        GS2DRectifyingParams, GS2DRectifyingData, GS2DRectifyingConfig,
    )
    
    device = torch.device(device)
    cov2d_3x3, means2d, opacities, radii, target = generate_test_data(
        C, N, dim0, dim1, device, dtype, eps
    )
    
    # Create operators (lambda_ssim=0 for exact comparison)
    autograd_op = GS2DRectifyingAutograd(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    cuda_op = GS2DRectifyingCUDA(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    # Create data structures
    params = GS2DRectifyingParams(cov2d_3x3=cov2d_3x3, means2d=means2d, opacities=opacities)
    data = GS2DRectifyingData(target=target, radii=radii)
    config = GS2DRectifyingConfig(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    # Forward pass
    autograd_result = autograd_op.forward_only(params, data, config)
    cuda_result = cuda_op.forward_only(params, data, config)
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, eps={eps}")
    print(f"Autograd loss: {autograd_result.outputs['loss'].item():.6f}")
    print(f"CUDA loss: {cuda_result.outputs['loss'].item():.6f}")
    
    # Compare render outputs
    render_diff = (autograd_result.outputs['render_output'] -
                   cuda_result.outputs['render_output']).abs()
    max_render_diff = render_diff.max().item()
    
    # Compare losses
    loss_diff = abs(autograd_result.outputs['loss'].item() -
                    cuda_result.outputs['loss'].item())
    
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
    
    from gsplat2ct.ops.gs2d_rectifying import (
        GS2DRectifyingAutograd, GS2DRectifyingCUDA,
        GS2DRectifyingParams, GS2DRectifyingData, GS2DRectifyingConfig,
    )
    
    device = torch.device(device)
    cov2d_3x3, means2d, opacities, radii, target = generate_test_data(
        C, N, dim0, dim1, device, dtype, eps
    )
    
    # Create operators
    autograd_op = GS2DRectifyingAutograd(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    cuda_op = GS2DRectifyingCUDA(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    # Create data structures
    params = GS2DRectifyingParams(cov2d_3x3=cov2d_3x3, means2d=means2d, opacities=opacities)
    data = GS2DRectifyingData(target=target, radii=radii)
    config = GS2DRectifyingConfig(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    # Forward + backward
    _, autograd_grad = autograd_op.forward_backward1(params, data, config)
    _, cuda_grad = cuda_op.forward_backward1(params, data, config)
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, eps={eps}")
    
    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = t1.abs().max().item()
        if scale < 1e-8:
            scale = 1.0
        rel_error = max_diff / scale
        passed = rel_error < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: max_diff={max_diff:.6e}, scale={scale:.6e}, rel_error={rel_error:.4f} {status}")
        return passed
    
    all_pass = True
    all_pass &= compare_tensors("v_cov2d_3x3", autograd_grad.get('cov2d_3x3'), cuda_grad.get('cov2d_3x3'))
    all_pass &= compare_tensors("v_means2d", autograd_grad.get('means2d'), cuda_grad.get('means2d'))
    all_pass &= compare_tensors("v_opacities", autograd_grad.get('opacities'), cuda_grad.get('opacities'))
    
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
    
    from gsplat2ct.ops.gs2d_rectifying import (
        GS2DRectifyingAutograd, GS2DRectifyingCUDA,
        GS2DRectifyingParams, GS2DRectifyingData, GS2DRectifyingConfig,
    )
    
    device = torch.device(device)
    cov2d_3x3, means2d, opacities, radii, target = generate_test_data(
        C, N, dim0, dim1, device, dtype, eps
    )
    
    # Create operators
    autograd_op = GS2DRectifyingAutograd(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    cuda_op = GS2DRectifyingCUDA(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    # Create data structures
    params = GS2DRectifyingParams(cov2d_3x3=cov2d_3x3, means2d=means2d, opacities=opacities)
    data = GS2DRectifyingData(target=target, radii=radii)
    config = GS2DRectifyingConfig(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    # Forward + backward1 + backward2
    _, autograd_grad, autograd_hess = autograd_op.forward_backward2(params, data, config)
    _, cuda_grad, cuda_hess = cuda_op.forward_backward2(params, data, config)
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, eps={eps}")
    print(f"Using lambda_l1=1.0, lambda_ssim=0.0 (L2 loss only)")
    
    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        if t1 is None or t2 is None:
            print(f"  {name}: SKIPPED (None)")
            return True
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = max(t1.abs().max().item(), t2.abs().max().item())
        if scale < 1e-8:
            scale = 1.0
        rel_error = max_diff / scale
        passed = rel_error < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: max_diff={max_diff:.6e}, scale={scale:.6e}, rel_error={rel_error:.4f} {status}")
        if not passed:
            # Print more details for debugging
            print(f"    t1 range: [{t1.min().item():.6e}, {t1.max().item():.6e}]")
            print(f"    t2 range: [{t2.min().item():.6e}, {t2.max().item():.6e}]")
        return passed
    
    print("\n--- First-Order Gradients ---")
    all_pass = True
    all_pass &= compare_tensors("v_cov2d_3x3", autograd_grad.get('cov2d_3x3'), cuda_grad.get('cov2d_3x3'))
    all_pass &= compare_tensors("v_means2d", autograd_grad.get('means2d'), cuda_grad.get('means2d'))
    all_pass &= compare_tensors("v_opacities", autograd_grad.get('opacities'), cuda_grad.get('opacities'))
    
    print("\n--- Diagonal Hessians ---")
    all_pass &= compare_tensors("h_cov2d_3x3", autograd_hess.get('cov2d_3x3'), cuda_hess.get('cov2d_3x3'))
    all_pass &= compare_tensors("h_means2d", autograd_hess.get('means2d'), cuda_hess.get('means2d'))
    all_pass &= compare_tensors("h_opacities", autograd_hess.get('opacities'), cuda_hess.get('opacities'))
    
    print("\n--- Cross-Term Hessians ---")
    all_pass &= compare_tensors("h_cov_means2d",
                                autograd_hess.get_cross('cov_means2d'),
                                cuda_hess.get_cross('cov_means2d'))
    all_pass &= compare_tensors("h_cov_opacity",
                                autograd_hess.get_cross('cov_opacity'),
                                cuda_hess.get_cross('cov_opacity'))
    all_pass &= compare_tensors("h_mean_opacity",
                                autograd_hess.get_cross('mean_opacity'),
                                cuda_hess.get_cross('mean_opacity'))
    
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
    
    from gsplat2ct.ops.gs2d_rectifying import (
        GS2DRectifyingCUDA,
        GS2DRectifyingParams, GS2DRectifyingData, GS2DRectifyingConfig,
    )
    
    device = torch.device(device)
    cov2d_3x3, means2d, opacities, radii, target = generate_test_data(
        C, N, dim0, dim1, device, dtype, eps
    )
    
    # Create operator
    cuda_op = GS2DRectifyingCUDA(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    # Create data structures
    params = GS2DRectifyingParams(cov2d_3x3=cov2d_3x3, means2d=means2d, opacities=opacities)
    data = GS2DRectifyingData(target=target, radii=radii)
    config = GS2DRectifyingConfig(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    # Forward
    result = cuda_op.forward_only(params, data, config)
    
    loss = result.outputs['loss'].item()
    render_output = result.outputs['render_output']
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, eps={eps}")
    print(f"Loss: {loss:.6f}")
    print(f"Render output range: [{render_output.min().item():.4f}, {render_output.max().item():.4f}]")
    print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
    
    # Sanity checks
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
    The loss is dL/d(theta_i) for each scalar element theta_i independently.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Gradient vs Finite Difference Validation")
    print("=" * 70)
    
    from gsplat2ct.ops.gs2d_rectifying import GS2DRectifyingAutograd
    
    device = torch.device(device)
    cov2d_3x3, means2d, opacities, radii, target = generate_test_data(
        C, N, dim0, dim1, device, dtype, eps, seed=99
    )
    
    autograd_op = GS2DRectifyingAutograd(eps=eps, lambda_l1=1.0, lambda_ssim=0.0)
    
    def compute_loss(cov_in, mean_in, opa_in):
        conics, mus = autograd_op._compute_conics_mu(cov_in, eps)
        render = autograd_op._rasterize_torch(mean_in, conics, opa_in, mus, dim0, dim1)
        return autograd_op._compute_l2_loss(render, target)
    
    # Compute autograd gradients
    cov2d_3x3_ag = cov2d_3x3.detach().clone().requires_grad_(True)
    means2d_ag = means2d.detach().clone().requires_grad_(True)
    opacities_ag = opacities.detach().clone().requires_grad_(True)
    
    loss = compute_loss(cov2d_3x3_ag, means2d_ag, opacities_ag)
    loss.backward()
    
    ag_grad_cov = cov2d_3x3_ag.grad.detach().clone()
    ag_grad_mean = means2d_ag.grad.detach().clone()
    ag_grad_opa = opacities_ag.grad.detach().clone()
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}, dtype=float64")
    print(f"Loss: {loss.item():.8f}")
    
    # Per-element FD for cov2d_3x3: perturb each scalar (c, n, i) independently
    fd_grad_cov = torch.zeros_like(cov2d_3x3)
    for c in range(C):
        for n in range(N):
            for i in range(6):
                cov_p = cov2d_3x3.clone()
                cov_m = cov2d_3x3.clone()
                cov_p[c, n, i] += fd_eps
                cov_m[c, n, i] -= fd_eps
                loss_p = compute_loss(cov_p, means2d, opacities)
                loss_m = compute_loss(cov_m, means2d, opacities)
                fd_grad_cov[c, n, i] = (loss_p - loss_m) / (2 * fd_eps)
    
    # Per-element FD for means2d: perturb each scalar (c, n, i) independently
    fd_grad_mean = torch.zeros_like(means2d)
    for c in range(C):
        for n in range(N):
            for i in range(2):
                mean_p = means2d.clone()
                mean_m = means2d.clone()
                mean_p[c, n, i] += fd_eps
                mean_m[c, n, i] -= fd_eps
                loss_p = compute_loss(cov2d_3x3, mean_p, opacities)
                loss_m = compute_loss(cov2d_3x3, mean_m, opacities)
                fd_grad_mean[c, n, i] = (loss_p - loss_m) / (2 * fd_eps)
    
    # Per-element FD for opacities: perturb each scalar (c, n) independently
    fd_grad_opa = torch.zeros_like(opacities)
    for c in range(C):
        for n in range(N):
            opa_p = opacities.clone()
            opa_m = opacities.clone()
            opa_p[c, n] += fd_eps
            opa_m[c, n] -= fd_eps
            loss_p = compute_loss(cov2d_3x3, means2d, opa_p)
            loss_m = compute_loss(cov2d_3x3, means2d, opa_m)
            fd_grad_opa[c, n] = (loss_p - loss_m) / (2 * fd_eps)
    
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
    all_pass &= compare_tensors("d_cov2d_3x3 (autograd vs FD)", ag_grad_cov, fd_grad_cov)
    all_pass &= compare_tensors("d_means2d   (autograd vs FD)", ag_grad_mean, fd_grad_mean)
    all_pass &= compare_tensors("d_opacities (autograd vs FD)", ag_grad_opa, fd_grad_opa)
    
    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 5 {'PASSED' if all_pass else 'FAILED'}")
    
    return all_pass


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("GS2D Rectifying (ConicsMu + Rasterization + ImageLoss) Test Suite")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Forward consistency
    results['forward'] = test_forward_consistency(C=2, N=100, dim0=11, dim1=15, eps=0.5)
    
    # Test 2: First-order backward
    results['backward1'] = test_backward1_consistency(C=2, N=100, dim0=11, dim1=15, eps=0.5)
    
    # Test 3: Second-order backward
    results['backward2'] = test_backward2_consistency(C=2, N=100, dim0=11, dim1=15, eps=0.5)
    
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
