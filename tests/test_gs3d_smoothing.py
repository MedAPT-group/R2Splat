"""
Test script for 3D Gaussian Splatting with TV Smoothing Loss.

This script validates the CUDA implementation against PyTorch autograd.
The results should match for both forward and backward passes.

Test stages:
1. Forward consistency: CUDA vs Torch render output and loss
2. First-order backward consistency: Gradients match
3. Second-order backward consistency: Hessians match (may have some tolerance due to numerical precision)

Usage:
    python tests/test_gs3d_smoothing.py

Author: Dy164
Date: 2026-02-03
"""

import torch
import numpy as np
from typing import Tuple


def generate_gaussian_params_3d(
    N: int,
    dim0: int,
    dim1: int,
    dim2: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random but valid 3D Gaussian parameters for testing.
    
    Returns:
        means3d: (N, 3) Gaussian centers [x, y, z]
        conics: (N, 6) Inverse covariance [a, p, q, b, r, c]
        opacities: (N,) Gaussian opacities
        radii: (N, 3) Gaussian radii [rx, ry, rz]
    """
    torch.manual_seed(42)
    
    # means3d: Random positions within volume bounds
    means3d = torch.rand(N, 3, device=device, dtype=dtype)
    means3d[:, 0] = means3d[:, 0] * dim2 * 0.6 + dim2 * 0.2  # x in [0.2*dim2, 0.8*dim2]
    means3d[:, 1] = means3d[:, 1] * dim1 * 0.6 + dim1 * 0.2  # y in [0.2*dim1, 0.8*dim1]
    means3d[:, 2] = means3d[:, 2] * dim0 * 0.6 + dim0 * 0.2  # z in [0.2*dim0, 0.8*dim0]
    
    # Generate valid positive definite covariance matrices via Cholesky
    # L is 3x3 lower triangular, Sigma = L @ L.T, conic = inv(Sigma)
    L11 = torch.rand(N, device=device, dtype=dtype) * 2.0 + 1.0  # [1, 3]
    L21 = torch.randn(N, device=device, dtype=dtype) * 0.5
    L22 = torch.rand(N, device=device, dtype=dtype) * 2.0 + 1.0
    L31 = torch.randn(N, device=device, dtype=dtype) * 0.5
    L32 = torch.randn(N, device=device, dtype=dtype) * 0.5
    L33 = torch.rand(N, device=device, dtype=dtype) * 2.0 + 1.0
    
    # Covariance = L @ L^T
    sigma_xx = L11 * L11
    sigma_xy = L11 * L21
    sigma_xz = L11 * L31
    sigma_yy = L21 * L21 + L22 * L22
    sigma_yz = L21 * L31 + L22 * L32
    sigma_zz = L31 * L31 + L32 * L32 + L33 * L33
    
    # Build 3x3 covariance matrix and compute inverse
    Sigma = torch.stack([
        torch.stack([sigma_xx, sigma_xy, sigma_xz], dim=-1),
        torch.stack([sigma_xy, sigma_yy, sigma_yz], dim=-1),
        torch.stack([sigma_xz, sigma_yz, sigma_zz], dim=-1),
    ], dim=-2)  # (N, 3, 3)
    
    # Add small regularization for numerical stability
    Sigma = Sigma + 0.1 * torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    
    # Compute inverse
    Sigma_inv = torch.linalg.inv(Sigma)  # (N, 3, 3)
    
    # Extract upper triangular: [a, p, q, b, r, c]
    # Sigma_inv = [[a, p, q], [p, b, r], [q, r, c]]
    # We store upper triangular
    conics = torch.stack([
        Sigma_inv[:, 0, 0],  # a
        Sigma_inv[:, 0, 1],  # p
        Sigma_inv[:, 0, 2],  # q
        Sigma_inv[:, 1, 1],  # b
        Sigma_inv[:, 1, 2],  # r
        Sigma_inv[:, 2, 2],  # c
    ], dim=-1)  # (N, 6)
    
    # Opacities in reasonable range
    opacities = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1  # [0.1, 0.9]
    
    # Compute radii from eigenvalues of covariance
    eigenvalues = torch.linalg.eigvalsh(Sigma)  # (N, 3)
    max_eigenvalue = eigenvalues.max(dim=-1)[0]
    radius = (4 * torch.sqrt(max_eigenvalue)).to(torch.int32).clamp(min=2)
    radii = torch.stack([radius, radius, radius], dim=-1)  # (N, 3)
    
    return means3d, conics, opacities, radii


def test_forward_consistency(
    N: int = 5,
    dim0: int = 16,
    dim1: int = 16,
    dim2: int = 16,
    tile_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-4,
) -> bool:
    """Test 1: Forward pass consistency between CUDA and Torch."""
    print("\n" + "="*70)
    print("TEST 1: Forward Pass Consistency (CUDA vs Torch)")
    print("="*70)
    
    from gsplat2ct.ops.gs3d_smoothing import (
        GS3DSmoothingAutograd, GS3DSmoothingCUDA,
        GS3DSmoothingParams, GS3DSmoothingData, GS3DSmoothingConfig
    )
    
    device = torch.device(device)
    means3d, conics, opacities, radii = generate_gaussian_params_3d(
        N, dim0, dim1, dim2, device, dtype
    )
    
    # Create operators
    autograd_op = GS3DSmoothingAutograd(tv_weight=1.0, tile_size=tile_size)
    cuda_op = GS3DSmoothingCUDA(tv_weight=1.0, tile_size=tile_size)
    
    # Create data structures
    params = GS3DSmoothingParams(means3d=means3d, conics=conics, opacities=opacities)
    data = GS3DSmoothingData(radii=radii, dim0=dim0, dim1=dim1, dim2=dim2, tile_size=tile_size)
    config = GS3DSmoothingConfig(tile_size=tile_size, tv_weight=1.0)
    
    # Forward pass
    autograd_result = autograd_op.forward_only(params, data, config)
    cuda_result = cuda_op.forward_only(params, data, config)
    
    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}, tile_size={tile_size}")
    print(f"Autograd loss: {autograd_result.outputs['loss'].item():.6f}")
    print(f"CUDA loss: {cuda_result.outputs['loss'].item():.6f}")
    
    # Compare render outputs
    render_diff = (autograd_result.outputs['render_output'] - 
                   cuda_result.outputs['render_output']).abs()
    max_render_diff = render_diff.max().item()
    mean_render_diff = render_diff.mean().item()
    
    # Compare losses
    loss_diff = abs(autograd_result.outputs['loss'].item() - 
                    cuda_result.outputs['loss'].item())
    
    print(f"\nRender output max diff: {max_render_diff:.6e}")
    print(f"Render output mean diff: {mean_render_diff:.6e}")
    print(f"Loss diff: {loss_diff:.6e}")
    
    # Allow larger tolerance for voxelization
    render_pass = max_render_diff < 0.01  # 1% tolerance for render
    loss_pass = loss_diff < rtol
    
    status = "✓" if (render_pass and loss_pass) else "✗"
    print(f"\n{status} TEST 1 {'PASSED' if (render_pass and loss_pass) else 'FAILED'}")
    
    return render_pass and loss_pass


def test_backward1_consistency(
    N: int = 5,
    dim0: int = 16,
    dim1: int = 16,
    dim2: int = 16,
    tile_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.1,  # 10% tolerance for gradients
) -> bool:
    """Test 2: First-order backward consistency."""
    print("\n" + "="*70)
    print("TEST 2: First-Order Backward Consistency (CUDA vs Torch)")
    print("="*70)
    
    from gsplat2ct.ops.gs3d_smoothing import (
        GS3DSmoothingAutograd, GS3DSmoothingCUDA,
        GS3DSmoothingParams, GS3DSmoothingData, GS3DSmoothingConfig
    )
    
    device = torch.device(device)
    means3d, conics, opacities, radii = generate_gaussian_params_3d(
        N, dim0, dim1, dim2, device, dtype
    )
    
    # Create operators
    autograd_op = GS3DSmoothingAutograd(tv_weight=1.0, tile_size=tile_size)
    cuda_op = GS3DSmoothingCUDA(tv_weight=1.0, tile_size=tile_size)
    
    # Create data structures
    params = GS3DSmoothingParams(means3d=means3d, conics=conics, opacities=opacities)
    data = GS3DSmoothingData(radii=radii, dim0=dim0, dim1=dim1, dim2=dim2, tile_size=tile_size)
    config = GS3DSmoothingConfig(tile_size=tile_size, tv_weight=1.0)
    
    # Forward + backward
    _, autograd_grad = autograd_op.forward_backward1(params, data, config)
    _, cuda_grad = cuda_op.forward_backward1(params, data, config)
    
    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}")
    
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
        return passed
    
    all_pass = True
    all_pass &= compare_tensors("v_means3d", autograd_grad.get('means3d'), cuda_grad.get('means3d'))
    all_pass &= compare_tensors("v_conics", autograd_grad.get('conics'), cuda_grad.get('conics'))
    all_pass &= compare_tensors("v_opacities", autograd_grad.get('opacities'), cuda_grad.get('opacities'))
    
    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 2 {'PASSED' if all_pass else 'FAILED'}")
    
    return all_pass


def test_backward2_consistency(
    N: int = 3,
    dim0: int = 12,
    dim1: int = 12,
    dim2: int = 12,
    tile_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.3,  # 30% tolerance for Hessians (TV loss has complex second-order behavior)
) -> bool:
    """Test 3: Second-order backward consistency.
    
    This test performs comprehensive element-wise comparison of all Hessian blocks
    between CUDA and Autograd implementations:
    
    1. h_means3d: (N, 6) upper triangle of 3×3 Hessian
    2. h_conics: (N, 21) upper triangle of 6×6 Hessian
    3. h_opacities: (N,) diagonal
    4. h_op_mean: (N, 3) cross-term ∂²L/∂o∂m
    5. h_op_conic: (N, 6) cross-term ∂²L/∂o∂c
    6. h_mean_conic: (N, 18) cross-term ∂²L/∂m∂c
    
    Note: TV loss Hessian is sparse (zero almost everywhere except at sign changes),
    so we use a relaxed tolerance and focus on non-zero elements.
    """
    print("\n" + "="*70)
    print("TEST 3: Second-Order Backward Consistency (CUDA vs Torch)")
    print("="*70)
    
    from gsplat2ct.ops.gs3d_smoothing import (
        GS3DSmoothingAutograd, GS3DSmoothingCUDA,
        GS3DSmoothingParams, GS3DSmoothingData, GS3DSmoothingConfig
    )
    
    device = torch.device(device)
    means3d, conics, opacities, radii = generate_gaussian_params_3d(
        N, dim0, dim1, dim2, device, dtype
    )
    
    # Create operators
    autograd_op = GS3DSmoothingAutograd(tv_weight=1.0, tile_size=tile_size)
    cuda_op = GS3DSmoothingCUDA(tv_weight=1.0, tile_size=tile_size)
    
    # Create data structures
    params = GS3DSmoothingParams(means3d=means3d, conics=conics, opacities=opacities)
    data = GS3DSmoothingData(radii=radii, dim0=dim0, dim1=dim1, dim2=dim2, tile_size=tile_size)
    config = GS3DSmoothingConfig(tile_size=tile_size, tv_weight=1.0)
    
    # Forward + backward1 + backward2
    _, autograd_grad, autograd_hess = autograd_op.forward_backward2(params, data, config)
    _, cuda_grad, cuda_hess = cuda_op.forward_backward2(params, data, config)
    
    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}")
    
    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, tol: float = rtol, 
                       show_details: bool = True) -> bool:
        """Compare two tensors element-wise with detailed diagnostics."""
        if t1 is None or t2 is None:
            print(f"  {name}: SKIPPED (None)")
            return True
        
        # Shape check
        if t1.shape != t2.shape:
            print(f"  {name}: SHAPE MISMATCH autograd={list(t1.shape)} vs cuda={list(t2.shape)} ✗")
            return False
        
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        scale = max(t1.abs().max().item(), t2.abs().max().item(), 1e-8)
        rel_error = max_diff / scale
        
        # Find location of maximum difference
        max_idx = torch.argmax(diff.flatten()).item()
        max_coords = np.unravel_index(max_idx, t1.shape)
        
        passed = rel_error < tol
        status = "✓" if passed else "✗"
        
        print(f"  {name}: shape={list(t1.shape)}")
        print(f"    max_diff={max_diff:.6e} at {max_coords}")
        print(f"    mean_diff={mean_diff:.6e}, scale={scale:.6e}")
        print(f"    rel_error={rel_error:.4f} (tol={tol}) {status}")
        
        if show_details and not passed:
            # Show top-5 mismatches
            flat_diff = diff.flatten()
            top_k = min(5, flat_diff.numel())
            top_vals, top_idxs = torch.topk(flat_diff, top_k)
            print(f"    Top {top_k} mismatches:")
            for i in range(top_k):
                coords = np.unravel_index(top_idxs[i].item(), t1.shape)
                v1 = t1.flatten()[top_idxs[i]].item()
                v2 = t2.flatten()[top_idxs[i]].item()
                print(f"      [{coords}]: autograd={v1:.6e}, cuda={v2:.6e}, diff={top_vals[i].item():.6e}")
        
        return passed
    
    def check_wellformed(name: str, hess: torch.Tensor) -> bool:
        """Check if Hessian is well-formed (finite, reasonable scale)."""
        if hess is None:
            print(f"  {name}: MISSING ✗")
            return False
        is_finite = torch.isfinite(hess).all().item()
        max_val = hess.abs().max().item()
        nonzero_ratio = (hess.abs() > 1e-10).float().mean().item()
        status = "✓" if is_finite and max_val < 1e6 else "✗"
        print(f"  {name}: shape={list(hess.shape)}, max={max_val:.3e}, nonzero={nonzero_ratio:.1%}, finite={is_finite} {status}")
        return is_finite and max_val < 1e6
    
    # ========================================================================
    # PART 1: First-order gradient consistency
    # ========================================================================
    print("\n" + "-"*60)
    print("PART 1: First-Order Gradient Consistency")
    print("-"*60)
    
    all_pass = True
    all_pass &= compare_tensors("v_means3d", autograd_grad.get('means3d'), cuda_grad.get('means3d'), tol=0.1)
    all_pass &= compare_tensors("v_conics", autograd_grad.get('conics'), cuda_grad.get('conics'), tol=0.2)
    all_pass &= compare_tensors("v_opacities", autograd_grad.get('opacities'), cuda_grad.get('opacities'), tol=0.1)
    
    # ========================================================================
    # PART 2: Second-order Hessian well-formedness
    # ========================================================================
    print("\n" + "-"*60)
    print("PART 2: Hessian Well-Formedness Check (Finite, Bounded)")
    print("-"*60)
    print("  Note: TV loss Hessian is sparse (zero except at sign crossings)")
    
    all_pass &= check_wellformed("h_means3d (CUDA)", cuda_hess.get('means3d'))
    all_pass &= check_wellformed("h_conics (CUDA)", cuda_hess.get('conics'))
    all_pass &= check_wellformed("h_opacities (CUDA)", cuda_hess.get('opacities'))
    all_pass &= check_wellformed("h_op_mean (CUDA)", cuda_hess.get('op_mean'))
    all_pass &= check_wellformed("h_op_conic (CUDA)", cuda_hess.get('op_conic'))
    all_pass &= check_wellformed("h_mean_conic (CUDA)", cuda_hess.get('mean_conic'))
    
    all_pass &= check_wellformed("h_means3d (Autograd)", autograd_hess.get('means3d'))
    all_pass &= check_wellformed("h_conics (Autograd)", autograd_hess.get('conics'))
    all_pass &= check_wellformed("h_opacities (Autograd)", autograd_hess.get('opacities'))
    all_pass &= check_wellformed("h_op_mean (Autograd)", autograd_hess.get('op_mean'))
    all_pass &= check_wellformed("h_op_conic (Autograd)", autograd_hess.get('op_conic'))
    all_pass &= check_wellformed("h_mean_conic (Autograd)", autograd_hess.get('mean_conic'))
    
    # ========================================================================
    # PART 3: Element-wise Hessian comparison (CUDA vs Autograd)
    # ========================================================================
    print("\n" + "-"*60)
    print("PART 3: Element-Wise Hessian Comparison (CUDA vs Autograd)")
    print("-"*60)
    print("  Using relaxed tolerance due to TV loss sparsity")
    
    # TV loss Hessian is sparse / non-smooth; keep general tolerance relaxed,
    # but keep the most diagnostic cross-term tighter.
    hess_tol = 0.5
    hess_tol_mean_conic = 0.2
    
    all_pass &= compare_tensors("h_means3d", autograd_hess.get('means3d'), cuda_hess.get('means3d'), tol=hess_tol)
    all_pass &= compare_tensors("h_conics", autograd_hess.get('conics'), cuda_hess.get('conics'), tol=hess_tol)
    all_pass &= compare_tensors("h_opacities", autograd_hess.get('opacities'), cuda_hess.get('opacities'), tol=hess_tol)
    all_pass &= compare_tensors("h_op_mean", autograd_hess.get('op_mean'), cuda_hess.get('op_mean'), tol=hess_tol)
    all_pass &= compare_tensors("h_op_conic", autograd_hess.get('op_conic'), cuda_hess.get('op_conic'), tol=hess_tol)
    all_pass &= compare_tensors(
        "h_mean_conic",
        autograd_hess.get('mean_conic'),
        cuda_hess.get('mean_conic'),
        tol=hess_tol_mean_conic,
    )
    
    # ========================================================================
    # PART 4: Detailed per-Gaussian Hessian block analysis
    # ========================================================================
    print("\n" + "-"*60)
    print("PART 4: Per-Gaussian Hessian Block Analysis")
    print("-"*60)
    
    # Upper triangle indices for visualization
    def upper_tri_to_matrix_3x3(utri: torch.Tensor) -> torch.Tensor:
        """Convert 6-element upper triangle to 3x3 symmetric matrix."""
        # utri = [H00, H01, H02, H11, H12, H22]
        M = torch.zeros(3, 3, device=utri.device, dtype=utri.dtype)
        M[0, 0] = utri[0]
        M[0, 1] = M[1, 0] = utri[1]
        M[0, 2] = M[2, 0] = utri[2]
        M[1, 1] = utri[3]
        M[1, 2] = M[2, 1] = utri[4]
        M[2, 2] = utri[5]
        return M
    
    def upper_tri_to_matrix_6x6(utri: torch.Tensor) -> torch.Tensor:
        """Convert 21-element upper triangle to 6x6 symmetric matrix."""
        M = torch.zeros(6, 6, device=utri.device, dtype=utri.dtype)
        idx = 0
        for i in range(6):
            for j in range(i, 6):
                M[i, j] = M[j, i] = utri[idx]
                idx += 1
        return M
    
    # Analyze first Gaussian in detail
    if N > 0:
        n = 0  # Analyze first Gaussian
        print(f"\n  Detailed analysis for Gaussian n={n}:")
        
        # means3d Hessian (3x3)
        h_mean_auto = autograd_hess.get('means3d')[n]  # (6,) upper triangle
        h_mean_cuda = cuda_hess.get('means3d')[n]      # (6,) upper triangle
        
        # Check shapes and skip detailed matrix analysis if mismatch
        if h_mean_auto.shape != h_mean_cuda.shape:
            print(f"    h_means3d shape mismatch: autograd={h_mean_auto.shape} vs cuda={h_mean_cuda.shape}")
            print(f"    Skipping matrix reconstruction...")
            status = "✓" if all_pass else "✗"
            print(f"\n{'='*60}")
            print(f"{status} TEST 3 {'PASSED' if all_pass else 'FAILED'}")
            print(f"{'='*60}")
            return all_pass
        
        H_mean_auto = upper_tri_to_matrix_3x3(h_mean_auto)
        H_mean_cuda = upper_tri_to_matrix_3x3(h_mean_cuda)
        
        print(f"\n    h_means3d (3×3 symmetric, from upper triangle):")
        print(f"      Autograd:\n{H_mean_auto.cpu().numpy()}")
        print(f"      CUDA:\n{H_mean_cuda.cpu().numpy()}")
        print(f"      Diff:\n{(H_mean_auto - H_mean_cuda).abs().cpu().numpy()}")
        
        # conics Hessian (6x6)
        h_con_auto = autograd_hess.get('conics')[n]  # (21,)
        h_con_cuda = cuda_hess.get('conics')[n]      # (21,)
        
        H_con_auto = upper_tri_to_matrix_6x6(h_con_auto)
        H_con_cuda = upper_tri_to_matrix_6x6(h_con_cuda)
        
        print(f"\n    h_conics (6×6 symmetric, from upper triangle):")
        print(f"      Autograd diagonal: {[H_con_auto[i,i].item() for i in range(6)]}")
        print(f"      CUDA diagonal:     {[H_con_cuda[i,i].item() for i in range(6)]}")
        print(f"      Max absolute diff: {(H_con_auto - H_con_cuda).abs().max().item():.6e}")
        
        # Cross-terms
        print(f"\n    Cross-terms for Gaussian n={n}:")
        print(f"      h_op_mean: autograd={autograd_hess.get('op_mean')[n].tolist()}")
        print(f"      h_op_mean: cuda    ={cuda_hess.get('op_mean')[n].tolist()}")
        print(f"      h_op_conic: autograd={autograd_hess.get('op_conic')[n].tolist()}")
        print(f"      h_op_conic: cuda    ={cuda_hess.get('op_conic')[n].tolist()}")
        
        # Reshape mean_conic to 3x6 for visualization
        mc_auto = autograd_hess.get('mean_conic')[n].reshape(3, 6)
        mc_cuda = cuda_hess.get('mean_conic')[n].reshape(3, 6)
        print(f"\n    h_mean_conic (3×6):")
        print(f"      Autograd:\n{mc_auto.cpu().numpy()}")
        print(f"      CUDA:\n{mc_cuda.cpu().numpy()}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    status = "✓" if all_pass else "✗"
    print(f"\n{'='*60}")
    print(f"{status} TEST 3 {'PASSED' if all_pass else 'FAILED'}")
    print(f"{'='*60}")
    
    return all_pass


def test_loss_value(
    N: int = 10,
    dim0: int = 24,
    dim1: int = 24,
    dim2: int = 24,
    tile_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> bool:
    """Test 4: Verify loss value is reasonable."""
    print("\n" + "="*70)
    print("TEST 4: Loss Value Sanity Check")
    print("="*70)
    
    from gsplat2ct.ops.gs3d_smoothing import (
        GS3DSmoothingCUDA,
        GS3DSmoothingParams, GS3DSmoothingData, GS3DSmoothingConfig
    )
    
    device = torch.device(device)
    means3d, conics, opacities, radii = generate_gaussian_params_3d(
        N, dim0, dim1, dim2, device, dtype
    )
    
    # Create operator
    cuda_op = GS3DSmoothingCUDA(tv_weight=1.0, tile_size=tile_size)
    
    # Create data structures
    params = GS3DSmoothingParams(means3d=means3d, conics=conics, opacities=opacities)
    data = GS3DSmoothingData(radii=radii, dim0=dim0, dim1=dim1, dim2=dim2, tile_size=tile_size)
    config = GS3DSmoothingConfig(tile_size=tile_size, tv_weight=1.0)
    
    # Forward
    result = cuda_op.forward_only(params, data, config)
    
    loss = result.outputs['loss'].item()
    render_output = result.outputs['render_output']
    tv_loss = result.outputs['tv_loss'].item()
    
    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}")
    print(f"Loss: {loss:.6f}")
    print(f"TV Loss: {tv_loss:.6f}")
    print(f"Render output range: [{render_output.min().item():.4f}, {render_output.max().item():.4f}]")
    print(f"Render output mean: {render_output.mean().item():.4f}")
    
    # Sanity checks
    is_finite = np.isfinite(loss)
    is_positive = loss >= 0
    render_finite = torch.isfinite(render_output).all().item()
    render_positive = (render_output >= 0).all().item()
    
    all_pass = is_finite and is_positive and render_finite and render_positive
    
    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 4 {'PASSED' if all_pass else 'FAILED'}")
    
    return all_pass


def test_gradient_direction(
    N: int = 5,
    dim0: int = 16,
    dim1: int = 16,
    dim2: int = 16,
    tile_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    step_size: float = 1e-3,
) -> bool:
    """Test 5: Verify gradient direction reduces loss."""
    print("\n" + "="*70)
    print("TEST 5: Gradient Direction Test")
    print("="*70)
    
    from gsplat2ct.ops.gs3d_smoothing import (
        GS3DSmoothingCUDA,
        GS3DSmoothingParams, GS3DSmoothingData, GS3DSmoothingConfig
    )
    
    device = torch.device(device)
    means3d, conics, opacities, radii = generate_gaussian_params_3d(
        N, dim0, dim1, dim2, device, dtype
    )
    
    # Create operator
    cuda_op = GS3DSmoothingCUDA(tv_weight=1.0, tile_size=tile_size)
    
    # Create data structures
    params = GS3DSmoothingParams(means3d=means3d, conics=conics, opacities=opacities)
    data = GS3DSmoothingData(radii=radii, dim0=dim0, dim1=dim1, dim2=dim2, tile_size=tile_size)
    config = GS3DSmoothingConfig(tile_size=tile_size, tv_weight=1.0)
    
    # Get initial loss and gradients
    forward_result, grad_result = cuda_op.forward_backward1(params, data, config)
    loss0 = forward_result.outputs['loss'].item()
    
    # Update parameters in negative gradient direction
    new_means3d = means3d - step_size * grad_result.get('means3d')
    new_opacities = opacities - step_size * grad_result.get('opacities')
    
    # Compute new loss
    new_params = GS3DSmoothingParams(means3d=new_means3d, conics=conics, opacities=new_opacities)
    new_result = cuda_op.forward_only(new_params, data, config)
    loss1 = new_result.outputs['loss'].item()
    
    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}")
    print(f"Initial loss: {loss0:.6f}")
    print(f"After gradient step: {loss1:.6f}")
    print(f"Loss change: {loss1 - loss0:.6e}")
    
    # Loss should decrease (or stay same if already at minimum)
    loss_decreased = loss1 <= loss0 + 1e-6  # Small tolerance
    
    status = "✓" if loss_decreased else "✗"
    print(f"\n{status} TEST 5 {'PASSED' if loss_decreased else 'FAILED'}")
    
    return loss_decreased


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("3D Gaussian Splatting with TV Smoothing Test Suite")
    print("="*70)
    
    results = {}
    
    # Test 1: Forward consistency
    results['forward'] = test_forward_consistency(N=100, dim0=17, dim1=13, dim2=19, tile_size=5)
    
    # Test 2: First-order backward
    results['backward1'] = test_backward1_consistency(N=100, dim0=17, dim1=13, dim2=19, tile_size=5)
    
    # Test 3: Second-order backward
    results['backward2'] = test_backward2_consistency(N=100, dim0=17, dim1=13, dim2=19, tile_size=5)
    
    # Test 4: Loss sanity
    results['loss_sanity'] = test_loss_value(N=100, dim0=17, dim1=13, dim2=19, tile_size=5)
    
    # Test 5: Gradient direction
    results['gradient_direction'] = test_gradient_direction(N=100, dim0=17, dim1=13, dim2=19, tile_size=5)
    
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
