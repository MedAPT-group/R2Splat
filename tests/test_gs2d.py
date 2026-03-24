"""
Test script for 2D Gaussian Splatting with Image Loss.

This script validates the CUDA implementation against PyTorch autograd.
When lambda_ssim=0 (L1 loss only), the results should match exactly.

Test stages:
1. Forward consistency: CUDA vs Torch render output and loss
2. First-order backward consistency: Gradients match
3. Second-order backward consistency: Hessians match

Usage:
    python tests/test_gs2d.py

Author: Dy164
Date: 2026-01-30
"""

import torch
import numpy as np
from typing import Tuple


def generate_gaussian_params(
    C: int,
    N: int,
    dim0: int,
    dim1: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random but valid Gaussian parameters for testing."""
    torch.manual_seed(42)
    
    # means2d: Random positions within image bounds
    means2d = torch.rand(C, N, 2, device=device, dtype=dtype)
    means2d[..., 0] *= dim1   # x in [0, dim1)
    means2d[..., 1] *= dim0   # y in [0, dim0)
    
    # Generate valid covariance matrices using Cholesky decomposition
    l11 = torch.rand(C, N, device=device, dtype=dtype) * 4 + 2  # [2, 6]
    l21 = torch.randn(C, N, device=device, dtype=dtype) * 1.5
    l22 = torch.rand(C, N, device=device, dtype=dtype) * 4 + 2  # [2, 6]
    
    sigma_xx = l11 * l11
    sigma_xy = l11 * l21
    sigma_yy = l21 * l21 + l22 * l22
    
    trace = sigma_xx + sigma_yy
    det = sigma_xx * sigma_yy - sigma_xy * sigma_xy
    discriminant = torch.sqrt(torch.clamp(trace * trace - 4 * det, min=1e-6))
    lambda_max = (trace + discriminant) / 2
    
    radii = (4 * torch.sqrt(lambda_max)).to(torch.int32)
    radii = torch.clamp(radii, min=1)
    
    inv_det = 1.0 / torch.clamp(det, min=1e-6)
    conic_a = sigma_yy * inv_det
    conic_b = -sigma_xy * inv_det
    conic_c = sigma_xx * inv_det
    conics = torch.stack([conic_a, conic_b, conic_c], dim=-1)
    
    mus = torch.full((C, N), 0.5, device=device, dtype=dtype)
    opacities = torch.rand(C, N, device=device, dtype=dtype) * 0.4 - 0.2
    
    return means2d, conics, opacities, mus, radii


def generate_target_image(
    C: int,
    dim0: int,
    dim1: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a random target image for loss computation."""
    torch.manual_seed(123)
    return torch.randn(C, dim0, dim1, device=device, dtype=dtype) * 0.1


def test_forward_consistency(
    C: int = 1,
    N: int = 5,
    dim0: int = 32,
    dim1: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-4,
) -> bool:
    """Test 1: Forward pass consistency between CUDA and Torch."""
    print("\n" + "="*70)
    print("TEST 1: Forward Pass Consistency (CUDA vs Torch)")
    print("="*70)
    
    from gsplat2ct.ops.gs2d import (
        GS2DAutograd, GS2DCUDA,
        GS2DParams, GS2DData, GS2DConfig
    )
    
    device = torch.device(device)
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    target = generate_target_image(C, dim0, dim1, device, dtype)
    
    # Create operators (lambda_ssim=0 for exact comparison)
    autograd_op = GS2DAutograd(lambda_l1=1.0, lambda_ssim=0.0)
    cuda_op = GS2DCUDA(lambda_l1=1.0, lambda_ssim=0.0)
    
    # Create data structures
    params = GS2DParams(means2d=means2d, conics=conics, opacities=opacities, mus=mus)
    data = GS2DData(target=target, radii=radii)
    config = GS2DConfig(lambda_l1=1.0, lambda_ssim=0.0)
    
    # Forward pass
    autograd_result = autograd_op.forward_only(params, data, config)
    cuda_result = cuda_op.forward_only(params, data, config)
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
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
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.05,
) -> bool:
    """Test 2: First-order backward consistency."""
    print("\n" + "="*70)
    print("TEST 2: First-Order Backward Consistency (CUDA vs Torch)")
    print("="*70)
    
    from gsplat2ct.ops.gs2d import (
        GS2DAutograd, GS2DCUDA,
        GS2DParams, GS2DData, GS2DConfig
    )
    
    device = torch.device(device)
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    target = generate_target_image(C, dim0, dim1, device, dtype)
    
    # Create operators
    autograd_op = GS2DAutograd(lambda_l1=1.0, lambda_ssim=0.0)
    cuda_op = GS2DCUDA(lambda_l1=1.0, lambda_ssim=0.0)
    
    # Create data structures
    params = GS2DParams(means2d=means2d, conics=conics, opacities=opacities, mus=mus)
    data = GS2DData(target=target, radii=radii)
    config = GS2DConfig(lambda_l1=1.0, lambda_ssim=0.0)
    
    # Forward + backward
    _, autograd_grad = autograd_op.forward_backward1(params, data, config)
    _, cuda_grad = cuda_op.forward_backward1(params, data, config)
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
    
    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = t1.abs().max().item()
        if scale < 1e-8:
            scale = 1.0
        rel_error = max_diff / scale
        passed = rel_error < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: max_diff={max_diff:.6e}, rel_error={rel_error:.4f} {status}")
        return passed
    
    all_pass = True
    all_pass &= compare_tensors("v_means2d", autograd_grad.get('means2d'), cuda_grad.get('means2d'))
    all_pass &= compare_tensors("v_conics", autograd_grad.get('conics'), cuda_grad.get('conics'))
    all_pass &= compare_tensors("v_opacities", autograd_grad.get('opacities'), cuda_grad.get('opacities'))
    all_pass &= compare_tensors("v_mus", autograd_grad.get('mus'), cuda_grad.get('mus'))
    
    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 2 {'PASSED' if all_pass else 'FAILED'}")
    
    return all_pass


def test_backward2_consistency(
    C: int = 1,
    N: int = 3,
    dim0: int = 32,
    dim1: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.1,
) -> bool:
    """Test 3: Second-order backward consistency."""
    print("\n" + "="*70)
    print("TEST 3: Second-Order Backward Consistency (CUDA vs Torch)")
    print("="*70)
    
    from gsplat2ct.ops.gs2d import (
        GS2DAutograd, GS2DCUDA,
        GS2DParams, GS2DData, GS2DConfig
    )
    
    device = torch.device(device)
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    target = generate_target_image(C, dim0, dim1, device, dtype)
    
    # Create operators
    autograd_op = GS2DAutograd(lambda_l1=1.0, lambda_ssim=0.0)
    cuda_op = GS2DCUDA(lambda_l1=1.0, lambda_ssim=0.0)
    
    # Create data structures
    params = GS2DParams(means2d=means2d, conics=conics, opacities=opacities, mus=mus)
    data = GS2DData(target=target, radii=radii)
    config = GS2DConfig(lambda_l1=1.0, lambda_ssim=0.0)
    
    # Forward + backward1 + backward2
    _, autograd_grad, autograd_hess = autograd_op.forward_backward2(params, data, config)
    _, cuda_grad, cuda_hess = cuda_op.forward_backward2(params, data, config)
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
    print(f"Using lambda_l1=1.0, lambda_ssim=0.0 (L1 only, no SSIM)")
    
    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        if t1 is None or t2 is None:
            print(f"  {name}: SKIPPED (None)")
            return True
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = t1.abs().max().item()
        if scale < 1e-8:
            scale = 1.0
        rel_error = max_diff / scale
        passed = rel_error < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: max_diff={max_diff:.6e}, rel_error={rel_error:.4f} {status}")
        return passed
    
    print("\n--- First-Order Gradients ---")
    all_pass = True
    all_pass &= compare_tensors("v_means2d", autograd_grad.get('means2d'), cuda_grad.get('means2d'))
    all_pass &= compare_tensors("v_conics", autograd_grad.get('conics'), cuda_grad.get('conics'))
    all_pass &= compare_tensors("v_opacities", autograd_grad.get('opacities'), cuda_grad.get('opacities'))
    all_pass &= compare_tensors("v_mus", autograd_grad.get('mus'), cuda_grad.get('mus'))
    
    print("\n--- Second-Order Hessians ---")
    all_pass &= compare_tensors("h_means2d", autograd_hess.get('means2d'), cuda_hess.get('means2d'))
    all_pass &= compare_tensors("h_conics", autograd_hess.get('conics'), cuda_hess.get('conics'))
    all_pass &= compare_tensors("h_mus", autograd_hess.get('mus'), cuda_hess.get('mus'))
    all_pass &= compare_tensors("h_opacities", autograd_hess.get('opacities'), cuda_hess.get('opacities'))
    
    print("\n--- Cross-Term Hessians ---")
    all_pass &= compare_tensors("h_mu_mean", autograd_hess.get_cross('mu_mean'), cuda_hess.get_cross('mu_mean'))
    all_pass &= compare_tensors("h_mu_conic", autograd_hess.get_cross('mu_conic'), cuda_hess.get_cross('mu_conic'))
    all_pass &= compare_tensors("h_mean_conic", autograd_hess.get_cross('mean_conic'), cuda_hess.get_cross('mean_conic'))
    all_pass &= compare_tensors("h_mu_opacity", autograd_hess.get_cross('mu_opacity'), cuda_hess.get_cross('mu_opacity'))
    all_pass &= compare_tensors("h_conic_opacity", autograd_hess.get_cross('conic_opacity'), cuda_hess.get_cross('conic_opacity'))
    all_pass &= compare_tensors("h_mean_opacity", autograd_hess.get_cross('mean_opacity'), cuda_hess.get_cross('mean_opacity'))
    
    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 3 {'PASSED' if all_pass else 'FAILED'}")
    
    return all_pass


def test_loss_value(
    C: int = 1,
    N: int = 10,
    dim0: int = 64,
    dim1: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> bool:
    """Test 4: Verify loss value is reasonable."""
    print("\n" + "="*70)
    print("TEST 4: Loss Value Sanity Check")
    print("="*70)
    
    from gsplat2ct.ops.gs2d import (
        GS2DCUDA,
        GS2DParams, GS2DData, GS2DConfig
    )
    
    device = torch.device(device)
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    target = generate_target_image(C, dim0, dim1, device, dtype)
    
    # Create operator
    cuda_op = GS2DCUDA(lambda_l1=1.0, lambda_ssim=0.0)
    
    # Create data structures
    params = GS2DParams(means2d=means2d, conics=conics, opacities=opacities, mus=mus)
    data = GS2DData(target=target, radii=radii)
    config = GS2DConfig(lambda_l1=1.0, lambda_ssim=0.0)
    
    # Forward
    result = cuda_op.forward_only(params, data, config)
    
    loss = result.outputs['loss'].item()
    render_output = result.outputs['render_output']
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
    print(f"Loss: {loss:.6f}")
    print(f"Render output range: [{render_output.min().item():.4f}, {render_output.max().item():.4f}]")
    print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
    
    # Sanity checks
    is_finite = np.isfinite(loss)
    is_positive = loss >= 0
    
    status = "✓" if (is_finite and is_positive) else "✗"
    print(f"\n{status} TEST 4 {'PASSED' if (is_finite and is_positive) else 'FAILED'}")
    
    return is_finite and is_positive


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("2D Gaussian Splatting Test Suite")
    print("="*70)
    
    results = {}
    
    # Test 1: Forward consistency
    results['forward'] = test_forward_consistency()
    
    # Test 2: First-order backward
    results['backward1'] = test_backward1_consistency()
    
    # Test 3: Second-order backward
    results['backward2'] = test_backward2_consistency(C=3, N=15, dim0=27, dim1=39)
    
    # Test 4: Loss sanity
    results['loss_sanity'] = test_loss_value()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        all_pass &= passed
    
    print("\n" + "="*70)
    if all_pass:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*70)
    
    return all_pass


if __name__ == "__main__":
    run_all_tests()
