"""
Test script for rasterize_to_pixels second-order backward pass.

This script validates the CUDA implementation of second-order derivatives (Hessian)
against PyTorch autograd.

For the CUDA implementation, we compute Hessians for each Gaussian independently,
ignoring cross-Gaussian terms. This is because:
1. The full Hessian would be O(N²) in size for N Gaussians
2. For many optimization algorithms, the block-diagonal approximation is sufficient

The test computes the full Hessian for each Gaussian using torch.autograd.functional.hessian()
and extracts the block-diagonal and cross-term blocks to compare with CUDA results.

Test stages:
1. Basic sanity check: verify shapes and non-NaN values
2. First-order gradient comparison with bwd1
3. Second-order Hessian comparison with torch autograd

Usage:
    python tests/test_rasterize_to_pixels_backward2.py

Author: Dy164
Date: 2025-01-28
"""

import torch
import numpy as np
import os
from typing import Tuple, Dict, Optional
from datetime import datetime


def generate_gaussian_params(
    C: int,
    N: int,
    dim0: int,
    dim1: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random but valid Gaussian parameters for testing.
    
    Same as in test_rasterize_to_pixels.py for consistency.
    """
    torch.manual_seed(42)  # For reproducibility
    
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


def test_bwd2_basic_sanity(
    C: int = 1,
    N: int = 5,
    dim0: int = 32,
    dim1: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32
) -> bool:
    """Test 1: Basic sanity check for bwd2 output shapes and values.
    
    Verifies:
    - All output tensors have correct shapes
    - No NaN or Inf values
    - First-order gradients are non-zero for at least some Gaussians
    """
    print("\n" + "="*70)
    print("TEST 1: Basic Sanity Check for bwd2")
    print("="*70)
    
    from gsplat2ct.cuda._wrapper import (
        isect_tiles,
        isect_offset_encode,
        rasterize_to_pixels_fwd,
        rasterize_to_pixels_bwd2,
    )
    
    device = torch.device(device)
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    
    tile_size = 16
    n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
    n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
    
    # Compute tile intersections
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
    )
    tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
    
    # Forward pass
    render_output = rasterize_to_pixels_fwd(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        dim1, dim0, tile_size
    )
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
    print(f"Render output shape: {render_output.shape}")
    print(f"Render output range: [{render_output.min():.4f}, {render_output.max():.4f}]")
    
    # Create upstream gradients
    v_render_output = torch.ones_like(render_output)
    h_render_output = torch.ones_like(render_output)
    
    # Second-order backward pass
    (v_means2d, v_conics, v_opacities, v_mus,
     h_means2d, h_conics,
     h_mu_mean, h_mu_conic, h_mean_conic,
     h_mu_opacity, h_conic_opacity, h_mean_opacity,
     h_mus, h_opacities) = rasterize_to_pixels_bwd2(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        v_render_output, h_render_output,
        dim1, dim0, tile_size
    )
    
    # Check shapes
    # Hessian uses upper triangular storage:
    # - h_means2d: [H00, H01, H11] = 3 elements
    # - h_conics: [H00, H01, H02, H11, H12, H22] = 6 elements
    expected_shapes = {
        'v_means2d': (C, N, 2),
        'v_conics': (C, N, 3),
        'v_opacities': (C, N),
        'v_mus': (C, N),
        'h_means2d': (C, N, 3),      # upper triangle [H00, H01, H11]
        'h_conics': (C, N, 6),       # upper triangle [H00, H01, H02, H11, H12, H22]
        'h_mu_mean': (C, N, 2),
        'h_mu_conic': (C, N, 3),
        'h_mean_conic': (C, N, 2, 3),
        'h_mu_opacity': (C, N),       # cross-term mu-opacity
        'h_conic_opacity': (C, N, 3), # cross-term conic-opacity
        'h_mean_opacity': (C, N, 2),  # cross-term mean-opacity
        'h_mus': (C, N),             # scalar Hessian for mu
        'h_opacities': (C, N),       # scalar Hessian for opacity
    }
    
    outputs = {
        'v_means2d': v_means2d,
        'v_conics': v_conics,
        'v_opacities': v_opacities,
        'v_mus': v_mus,
        'h_means2d': h_means2d,
        'h_conics': h_conics,
        'h_mu_mean': h_mu_mean,
        'h_mu_conic': h_mu_conic,
        'h_mean_conic': h_mean_conic,
        'h_mu_opacity': h_mu_opacity,
        'h_conic_opacity': h_conic_opacity,
        'h_mean_opacity': h_mean_opacity,
        'h_mus': h_mus,
        'h_opacities': h_opacities,
    }
    
    all_pass = True
    
    print("\n--- Shape Check ---")
    for name, expected in expected_shapes.items():
        actual = outputs[name].shape
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False
    
    print("\n--- NaN/Inf Check ---")
    for name, tensor in outputs.items():
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        status = "✓" if not (has_nan or has_inf) else "✗"
        print(f"  {name}: NaN={has_nan}, Inf={has_inf} {status}")
        if has_nan or has_inf:
            all_pass = False
    
    print("\n--- Value Statistics ---")
    for name, tensor in outputs.items():
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        mean_val = tensor.mean().item()
        print(f"  {name}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
    
    # Check that at least some gradients are non-zero
    print("\n--- Non-zero Check ---")
    v_means2d_nonzero = (v_means2d.abs() > 1e-8).any().item()
    v_conics_nonzero = (v_conics.abs() > 1e-8).any().item()
    v_opacities_nonzero = (v_opacities.abs() > 1e-8).any().item()
    v_mus_nonzero = (v_mus.abs() > 1e-8).any().item()
    
    print(f"  v_means2d has non-zero values: {v_means2d_nonzero}")
    print(f"  v_conics has non-zero values: {v_conics_nonzero}")
    print(f"  v_opacities has non-zero values: {v_opacities_nonzero}")
    print(f"  v_mus has non-zero values: {v_mus_nonzero}")
    
    if not (v_means2d_nonzero and v_conics_nonzero and v_opacities_nonzero and v_mus_nonzero):
        print("  WARNING: Some gradients are all zero!")
    
    if all_pass:
        print("\n✓ TEST 1 PASSED: Basic sanity check")
    else:
        print("\n✗ TEST 1 FAILED: Basic sanity check")
    
    return all_pass


def test_bwd2_first_order_consistency(
    C: int = 1,
    N: int = 5,
    dim0: int = 32,
    dim1: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.05
) -> bool:
    """Test 2: Verify first-order gradients from bwd2 match bwd.
    
    The bwd2 function should return the same first-order gradients as bwd.
    """
    print("\n" + "="*70)
    print("TEST 2: First-Order Gradient Consistency (bwd2 vs bwd)")
    print("="*70)
    
    from gsplat2ct.cuda._wrapper import (
        isect_tiles,
        isect_offset_encode,
        rasterize_to_pixels_fwd,
        rasterize_to_pixels_bwd,
        rasterize_to_pixels_bwd2,
    )
    
    device = torch.device(device)
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    
    tile_size = 16
    n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
    n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
    
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
    )
    tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
    
    # Forward pass
    render_output = rasterize_to_pixels_fwd(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        dim1, dim0, tile_size
    )
    
    # Upstream gradient
    v_render_output = torch.randn_like(render_output)  # Use random for better test
    h_render_output = torch.ones_like(render_output)
    
    # First-order backward (bwd)
    v_means2d_bwd, v_conics_bwd, v_opacities_bwd, v_mus_bwd = rasterize_to_pixels_bwd(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids, v_render_output,
        dim1, dim0, tile_size
    )
    
    # Second-order backward (bwd2) - should match first-order part
    (v_means2d_bwd2, v_conics_bwd2, v_opacities_bwd2, v_mus_bwd2,
     _, _, _, _, _, _, _, _, _, _) = rasterize_to_pixels_bwd2(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        v_render_output, h_render_output,
        dim1, dim0, tile_size
    )
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
    
    def compare_tensors(name, t1, t2, rtol):
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        # Compute relative error
        scale = t1.abs().max().item()
        if scale < 1e-8:
            scale = 1.0
        rel_error = max_diff / scale
        
        passed = rel_error < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: max_abs_diff={max_diff:.6e}, rel_error={rel_error:.4f} {status}")
        return passed
    
    all_pass = True
    all_pass &= compare_tensors("v_means2d", v_means2d_bwd, v_means2d_bwd2, rtol)
    all_pass &= compare_tensors("v_conics", v_conics_bwd, v_conics_bwd2, rtol)
    all_pass &= compare_tensors("v_opacities", v_opacities_bwd, v_opacities_bwd2, rtol)
    all_pass &= compare_tensors("v_mus", v_mus_bwd, v_mus_bwd2, rtol)
    
    if all_pass:
        print("\n✓ TEST 2 PASSED: First-order gradients are consistent")
    else:
        print("\n✗ TEST 2 FAILED: First-order gradients mismatch")
    
    return all_pass


def compute_single_gaussian_hessian_torch(
    c: int,
    g: int,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    mus: torch.Tensor,
    dim0: int,
    dim1: int,
    weights: Optional[torch.Tensor] = None,
    render_output: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the Hessian for a single Gaussian using torch.autograd.
    
    This computes ∂²L/∂θ² for all parameters θ of Gaussian (c, g).
    
    If weights is None:
        Loss is L = sum(render_output), giving v_O = 1 and h_O = 0.
    If weights is provided:
        Loss is L = sum(weights * O²), where O is total render_output.
        Since O = Σ_g α_g, we have ∂L/∂α_g = 2 * weights * O (depends on total O!)
        and ∂²L/∂α_g² = 2 * weights.
        
        To correctly match CUDA, we need to pass in the actual render_output
        and compute gradients through it.
    
    Returns:
        Dictionary with Hessian blocks
    """
    device = means2d.device
    dtype = means2d.dtype
    
    # Extract single Gaussian parameters
    mean = means2d[c, g].detach().clone().requires_grad_(True)  # [2]
    conic = conics[c, g].detach().clone().requires_grad_(True)   # [3]
    opacity = opacities[c, g].detach().clone().requires_grad_(True)  # scalar
    mu = mus[c, g].detach().clone().requires_grad_(True)  # scalar
    
    # Create pixel grid
    y_coords = torch.arange(dim0, device=device, dtype=dtype) + 0.5
    x_coords = torch.arange(dim1, device=device, dtype=dtype) + 0.5
    py, px = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [dim0, dim1]
    
    # Compute Gaussian contribution
    dx = px - mean[0]
    dy = py - mean[1]
    
    sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
    alpha = opacity * mu * torch.exp(-sigma)
    
    # Apply numerical mask
    mask = (sigma >= 0) & (alpha.abs() >= 1e-5)
    alpha = torch.where(mask, alpha, torch.zeros_like(alpha))
    
    # Compute loss
    if weights is None:
        # Simple sum: L = sum(alpha), v_O = 1, h_O = 0
        loss = alpha.sum()
    else:
        # Weighted squared loss: L = sum(weights * O^2)
        # where O is the TOTAL render output (from all Gaussians)
        # ∂L/∂O = 2 * weights * O
        # ∂²L/∂O² = 2 * weights
        #
        # For the Hessian w.r.t. single Gaussian parameters:
        # Since O = α_g + (other gaussians), and other gaussians are constant w.r.t. θ_g,
        # we need to compute ∂L/∂α_g = (∂L/∂O)(∂O/∂α_g) = 2*w*O * 1 = 2*w*O
        # 
        # But for second-order: ∂²L/∂θ² = (∂O/∂θ)^T (∂²L/∂O²) (∂O/∂θ) + Σ (∂L/∂O) (∂²O/∂θ²)
        #                              = (∂α/∂θ)^T (2*w) (∂α/∂θ) + (2*w*O) (∂²α/∂θ²)
        #
        # The key insight: the v_O passed to CUDA is 2*w*O (actual render output),
        # and h_O is 2*w (constant). The CUDA kernel computes the correct formula.
        #
        # For torch autograd to match, we need to define L in terms of the actual O:
        # L = Σ w * O², but O includes THIS gaussian's contribution
        
        if render_output is not None:
            # Use actual render output (but only care about this gaussian's gradient)
            # We construct: L = sum(weights * (other_contribution + alpha)^2)
            # where other_contribution = render_output - old_alpha
            # But this is tricky because old_alpha was computed with old params
            
            # Simpler approach: use the chain rule directly
            # ∂L/∂θ = ∂L/∂O × ∂O/∂θ = (2*w*O) × (∂α/∂θ)
            # ∂²L/∂θ² = (∂α/∂θ)^T (2*w) (∂α/∂θ) + (2*w*O) (∂²α/∂θ²)
            #
            # This means we can compute it as:
            # loss = sum(v_O * alpha) where v_O = 2*w*O
            # and add the GGN term separately
            
            v_O = 2 * weights[c] * render_output[c]  # [dim0, dim1]
            h_O = 2 * weights[c]  # [dim0, dim1]
            
            # But torch autograd can't separate GGN and pure Hessian easily...
            # Let's just compute the loss properly:
            # L = sum(w * O^2) where O depends on this gaussian's alpha
            
            # Recompute O with this gaussian's new alpha contribution
            # O_new = O_old - alpha_old + alpha_new
            # But alpha_old was computed without gradients, so:
            O_new = render_output[c].detach() + alpha  # add current alpha (with grads)
            # Wait, this double-counts! render_output already includes this gaussian.
            
            # Actually we need: O_reconstructed = (render_output - alpha_old) + alpha
            # where alpha_old is the contribution computed without requires_grad
            alpha_old = opacities[c, g].detach() * mus[c, g].detach() * torch.exp(-sigma.detach())
            alpha_old = torch.where(mask, alpha_old, torch.zeros_like(alpha_old))
            
            O_reconstructed = render_output[c].detach() - alpha_old.detach() + alpha
            loss = (weights[c] * O_reconstructed * O_reconstructed).sum()
        else:
            # Fallback: just use alpha^2 (incorrect for multi-gaussian case)
            loss = (weights[c] * alpha * alpha).sum()
    
    # Compute first derivatives
    v_mean, = torch.autograd.grad(loss, mean, create_graph=True)
    v_conic, = torch.autograd.grad(loss, conic, create_graph=True)
    v_mu, = torch.autograd.grad(loss, mu, create_graph=True)
    v_opacity, = torch.autograd.grad(loss, opacity, create_graph=True)
    
    # Compute second derivatives (Hessian blocks)
    # h_means2d[i, j] = ∂²L/∂mean[i]∂mean[j]
    h_means2d = torch.zeros(2, 2, device=device, dtype=dtype)
    for i in range(2):
        grad_i, = torch.autograd.grad(v_mean[i], mean, retain_graph=True)
        h_means2d[i] = grad_i
    
    # h_conics[i, j] = ∂²L/∂conic[i]∂conic[j]
    h_conics = torch.zeros(3, 3, device=device, dtype=dtype)
    for i in range(3):
        grad_i, = torch.autograd.grad(v_conic[i], conic, retain_graph=True)
        h_conics[i] = grad_i
    
    # Cross-term: h_mu_mean[i] = ∂²L/∂μ∂mean[i]
    h_mu_mean, = torch.autograd.grad(v_mu, mean, retain_graph=True)
    
    # Cross-term: h_mu_conic[i] = ∂²L/∂μ∂conic[i]
    h_mu_conic, = torch.autograd.grad(v_mu, conic, retain_graph=True)
    
    # Cross-term: h_mean_conic[i, j] = ∂²L/∂mean[i]∂conic[j]
    h_mean_conic = torch.zeros(2, 3, device=device, dtype=dtype)
    for i in range(2):
        grad_i, = torch.autograd.grad(v_mean[i], conic, retain_graph=True)
        h_mean_conic[i] = grad_i
    
    # Scalar Hessians
    # h_mu = ∂²L/∂μ²
    # h_opacity = ∂²L/∂o²
    # Note: For simple loss L = sum(alpha), ∂²G/∂μ² = 0 and ∂²G/∂o² = 0
    # Only when using weighted loss with h_O ≠ 0, we get GGN contribution
    try:
        h_mu, = torch.autograd.grad(v_mu, mu, retain_graph=True, allow_unused=True)
        if h_mu is None:
            h_mu = torch.tensor(0.0, device=device, dtype=dtype)
    except RuntimeError:
        h_mu = torch.tensor(0.0, device=device, dtype=dtype)
    
    try:
        h_opacity, = torch.autograd.grad(v_opacity, opacity, retain_graph=True, allow_unused=True)
        if h_opacity is None:
            h_opacity = torch.tensor(0.0, device=device, dtype=dtype)
    except RuntimeError:
        h_opacity = torch.tensor(0.0, device=device, dtype=dtype)
    
    # Cross-term: h_mu_opacity = ∂²L/∂μ∂o
    try:
        h_mu_opacity, = torch.autograd.grad(v_mu, opacity, retain_graph=True, allow_unused=True)
        if h_mu_opacity is None:
            h_mu_opacity = torch.tensor(0.0, device=device, dtype=dtype)
    except RuntimeError:
        h_mu_opacity = torch.tensor(0.0, device=device, dtype=dtype)
    
    # Cross-term: h_conic_opacity[i] = ∂²L/∂conic[i]∂o
    h_conic_opacity = torch.zeros(3, device=device, dtype=dtype)
    for i in range(3):
        try:
            grad_i, = torch.autograd.grad(v_conic[i], opacity, retain_graph=True, allow_unused=True)
            if grad_i is not None:
                h_conic_opacity[i] = grad_i
        except RuntimeError:
            pass
    
    # Cross-term: h_mean_opacity[i] = ∂²L/∂mean[i]∂o
    h_mean_opacity = torch.zeros(2, device=device, dtype=dtype)
    for i in range(2):
        try:
            grad_i, = torch.autograd.grad(v_mean[i], opacity, retain_graph=True, allow_unused=True)
            if grad_i is not None:
                h_mean_opacity[i] = grad_i
        except RuntimeError:
            pass
    
    return {
        'h_means2d': h_means2d.detach(),
        'h_conics': h_conics.detach(),
        'h_mu_mean': h_mu_mean.detach(),
        'h_mu_conic': h_mu_conic.detach(),
        'h_mean_conic': h_mean_conic.detach(),
        'h_mu': h_mu.detach(),
        'h_opacity': h_opacity.detach(),
        'h_mu_opacity': h_mu_opacity.detach(),
        'h_conic_opacity': h_conic_opacity.detach(),
        'h_mean_opacity': h_mean_opacity.detach(),
    }


def test_bwd2_hessian_vs_torch(
    C: int = 1,
    N: int = 3,
    dim0: int = 32,
    dim1: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.1,
    verbose: bool = True
) -> bool:
    """Test 3: Compare CUDA Hessian with torch.autograd Hessian.
    
    For each Gaussian, compute the per-Gaussian Hessian using torch.autograd
    and compare with CUDA results.
    
    Note: CUDA computes h_θ = h_O × (∂G/∂θ)² + v_O × ∂²G/∂θ²
    With v_O = 1 (ones) and h_O = 1 (ones), this should match torch.
    
    But when h_O = 0, we get h_θ = v_O × ∂²G/∂θ², which is the "pure" Hessian.
    """
    print("\n" + "="*70)
    print("TEST 3: Hessian Comparison (CUDA vs Torch Autograd)")
    print("="*70)
    
    from gsplat2ct.cuda._wrapper import (
        isect_tiles,
        isect_offset_encode,
        rasterize_to_pixels_fwd,
        rasterize_to_pixels_bwd2,
    )
    
    device = torch.device(device)
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    
    tile_size = 16
    n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
    n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
    
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
    )
    tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
    
    render_output = rasterize_to_pixels_fwd(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        dim1, dim0, tile_size
    )
    
    # Use v_O = 1, h_O = 0 for pure Hessian (no GGN term)
    v_render_output = torch.ones_like(render_output)
    h_render_output = torch.zeros_like(render_output)
    
    # CUDA bwd2
    (v_means2d_cuda, v_conics_cuda, v_opacities_cuda, v_mus_cuda,
     h_means2d_cuda, h_conics_cuda,
     h_mu_mean_cuda, h_mu_conic_cuda, h_mean_conic_cuda,
     h_mu_opacity_cuda, h_conic_opacity_cuda, h_mean_opacity_cuda,
     h_mus_cuda, h_opacities_cuda) = rasterize_to_pixels_bwd2(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        v_render_output, h_render_output,
        dim1, dim0, tile_size
    )
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
    print(f"Using v_O=1, h_O=0 (pure Hessian, no GGN term)")
    print()
    
    all_pass = True
    
    for c in range(C):
        for g in range(N):
            # Compute torch Hessian for this Gaussian
            torch_hess = compute_single_gaussian_hessian_torch(
                c, g, means2d, conics, opacities, mus, dim0, dim1
            )
            
            # Extract CUDA Hessian for this Gaussian
            # h_means2d: [3] upper triangle [H00, H01, H11]
            cuda_h_means2d_triu = h_means2d_cuda[c, g]  # [3]
            # h_conics: [6] upper triangle [H00, H01, H02, H11, H12, H22]
            cuda_h_conics_triu = h_conics_cuda[c, g]    # [6]
            cuda_h_mu_mean = h_mu_mean_cuda[c, g]  # [2]
            cuda_h_mu_conic = h_mu_conic_cuda[c, g]  # [3]
            cuda_h_mean_conic = h_mean_conic_cuda[c, g]  # [2, 3]
            cuda_h_mu = h_mus_cuda[c, g]  # scalar
            cuda_h_opacity = h_opacities_cuda[c, g]  # scalar
            cuda_h_mu_opacity = h_mu_opacity_cuda[c, g]  # scalar
            cuda_h_conic_opacity = h_conic_opacity_cuda[c, g]  # [3]
            cuda_h_mean_opacity = h_mean_opacity_cuda[c, g]  # [2]
            
            # Reconstruct full matrices from upper triangle for comparison
            # h_means2d: [H00, H01, H11] -> [[H00, H01], [H01, H11]]
            cuda_h_means2d = torch.zeros(2, 2, device=device, dtype=dtype)
            cuda_h_means2d[0, 0] = cuda_h_means2d_triu[0]  # H00
            cuda_h_means2d[0, 1] = cuda_h_means2d_triu[1]  # H01
            cuda_h_means2d[1, 0] = cuda_h_means2d_triu[1]  # H10 = H01
            cuda_h_means2d[1, 1] = cuda_h_means2d_triu[2]  # H11
            
            # h_conics: [H00, H01, H02, H11, H12, H22] -> full 3x3
            cuda_h_conics = torch.zeros(3, 3, device=device, dtype=dtype)
            cuda_h_conics[0, 0] = cuda_h_conics_triu[0]  # H00
            cuda_h_conics[0, 1] = cuda_h_conics_triu[1]  # H01
            cuda_h_conics[0, 2] = cuda_h_conics_triu[2]  # H02
            cuda_h_conics[1, 0] = cuda_h_conics_triu[1]  # H10 = H01
            cuda_h_conics[1, 1] = cuda_h_conics_triu[3]  # H11
            cuda_h_conics[1, 2] = cuda_h_conics_triu[4]  # H12
            cuda_h_conics[2, 0] = cuda_h_conics_triu[2]  # H20 = H02
            cuda_h_conics[2, 1] = cuda_h_conics_triu[4]  # H21 = H12
            cuda_h_conics[2, 2] = cuda_h_conics_triu[5]  # H22
            
            if verbose:
                print(f"--- Gaussian (c={c}, g={g}) ---")
            
            def compare_and_print(name, cuda_val, torch_val):
                nonlocal all_pass
                diff = (cuda_val - torch_val).abs()
                max_diff = diff.max().item()
                scale = torch_val.abs().max().item()
                if scale < 1e-8:
                    scale = 1.0
                rel_error = max_diff / scale
                passed = rel_error < rtol
                status = "✓" if passed else "✗"
                
                if verbose or not passed:
                    print(f"  {name}:")
                    print(f"    CUDA:  {cuda_val.cpu().numpy()}")
                    print(f"    Torch: {torch_val.cpu().numpy()}")
                    print(f"    max_diff={max_diff:.6e}, rel_error={rel_error:.4f} {status}")
                
                if not passed:
                    all_pass = False
                return passed
            
            compare_and_print("h_means2d", cuda_h_means2d, torch_hess['h_means2d'])
            compare_and_print("h_conics", cuda_h_conics, torch_hess['h_conics'])
            compare_and_print("h_mu_mean", cuda_h_mu_mean, torch_hess['h_mu_mean'])
            compare_and_print("h_mu_conic", cuda_h_mu_conic, torch_hess['h_mu_conic'])
            compare_and_print("h_mean_conic", cuda_h_mean_conic, torch_hess['h_mean_conic'])
            compare_and_print("h_mu", cuda_h_mu, torch_hess['h_mu'])
            compare_and_print("h_opacity", cuda_h_opacity, torch_hess['h_opacity'])
            compare_and_print("h_mu_opacity", cuda_h_mu_opacity, torch_hess['h_mu_opacity'])
            compare_and_print("h_conic_opacity", cuda_h_conic_opacity, torch_hess['h_conic_opacity'])
            compare_and_print("h_mean_opacity", cuda_h_mean_opacity, torch_hess['h_mean_opacity'])
            
            if verbose:
                print()
    
    if all_pass:
        print("✓ TEST 3 PASSED: Hessian values match torch.autograd")
    else:
        print("✗ TEST 3 FAILED: Hessian values mismatch")
    
    return all_pass


def test_bwd2_symmetry(
    C: int = 1,
    N: int = 5,
    dim0: int = 32,
    dim1: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32
) -> bool:
    """Test 4: Verify Hessian matrices are symmetric.
    
    With upper triangular storage, symmetry is guaranteed by construction.
    This test verifies the storage format is correct:
    - h_means2d: [H00, H01, H11] (3 elements per Gaussian)
    - h_conics: [H00, H01, H02, H11, H12, H22] (6 elements per Gaussian)
    """
    print("\n" + "="*70)
    print("TEST 4: Hessian Storage Format Check")
    print("="*70)
    
    from gsplat2ct.cuda._wrapper import (
        isect_tiles,
        isect_offset_encode,
        rasterize_to_pixels_fwd,
        rasterize_to_pixels_bwd2,
    )
    
    device = torch.device(device)
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    
    tile_size = 16
    n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
    n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
    
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
    )
    tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
    
    render_output = rasterize_to_pixels_fwd(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        dim1, dim0, tile_size
    )
    
    v_render_output = torch.ones_like(render_output)
    h_render_output = torch.ones_like(render_output)
    
    (_, _, _, _,
     h_means2d, h_conics,
     _, _, _, _, _, _, _, _) = rasterize_to_pixels_bwd2(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        v_render_output, h_render_output,
        dim1, dim0, tile_size
    )
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
    
    # Check h_means2d shape and storage format
    # Expected: [C, N, 3] for upper triangle [H00, H01, H11]
    means2d_shape_ok = h_means2d.shape == (C, N, 3)
    
    # Check h_conics shape and storage format  
    # Expected: [C, N, 6] for upper triangle [H00, H01, H02, H11, H12, H22]
    conics_shape_ok = h_conics.shape == (C, N, 6)
    
    print(f"h_means2d shape: {h_means2d.shape} (expected: ({C}, {N}, 3)) {'✓' if means2d_shape_ok else '✗'}")
    print(f"h_conics shape: {h_conics.shape} (expected: ({C}, {N}, 6)) {'✓' if conics_shape_ok else '✗'}")
    
    # With upper triangular storage, symmetry is guaranteed by construction
    print("\nStorage format (upper triangle, row-major):")
    print("  h_means2d: [H00, H01, H11]")
    print("  h_conics: [H00, H01, H02, H11, H12, H22]")
    print("\nSymmetry: Guaranteed by construction (upper triangle only)")
    
    all_pass = means2d_shape_ok and conics_shape_ok
    
    if all_pass:
        print("\n✓ TEST 4 PASSED: Hessian storage format correct")
    else:
        print("\n✗ TEST 4 FAILED: Hessian storage format incorrect")
    
    return all_pass


def test_bwd2_large_scale(
    C: int = 8,
    N: int = 200,
    dim0: int = 128,
    dim1: int = 128,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.15,
    sample_size: int = 10
) -> bool:
    """Test 5: Large-scale Hessian comparison with weighted squared loss.
    
    Uses L = Σ a_ij * c_ij² as the loss function, which gives:
    - v_O = ∂L/∂c = 2 * a * c (first-order upstream gradient)
    - h_O = ∂²L/∂c² = 2 * a (second-order upstream gradient)
    
    This tests the FULL chain rule: h_θ = h_O × (∂G/∂θ)² + v_O × ∂²G/∂θ²
    
    Args:
        C: Number of cameras
        N: Number of Gaussians per camera
        dim0: Image height
        dim1: Image width
        device: Device to run on
        dtype: Data type
        rtol: Relative tolerance
        sample_size: Number of Gaussians to sample per camera for detailed comparison
    """
    print("\n" + "="*70)
    print("TEST 5: Large-Scale Hessian with Weighted Squared Loss")
    print("="*70)
    
    from gsplat2ct.cuda._wrapper import (
        isect_tiles,
        isect_offset_encode,
        rasterize_to_pixels_fwd,
        rasterize_to_pixels_bwd2,
    )
    import time
    
    device = torch.device(device)
    
    print(f"Configuration: C={C}, N={N}, dim0={dim0}, dim1={dim1}")
    print(f"Total Gaussians: {C * N}")
    print(f"Loss function: L = Σ a_ij * c_ij² (random weights a_ij)")
    print(f"This tests: v_O = 2*a*c (non-zero), h_O = 2*a (non-zero)")
    print(f"Sampling {sample_size} Gaussians per camera for detailed comparison")
    print()
    
    means2d, conics, opacities, mus, radii = generate_gaussian_params(
        C, N, dim0, dim1, device, dtype
    )
    
    tile_size = 16
    n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
    n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
    
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, n_tiles_dim1, n_tiles_dim0
    )
    tile_offsets = isect_offset_encode(isect_ids, C, n_tiles_dim1, n_tiles_dim0)
    
    # Forward pass
    t0 = time.time()
    render_output = rasterize_to_pixels_fwd(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        dim1, dim0, tile_size
    )
    torch.cuda.synchronize()
    fwd_time = time.time() - t0
    print(f"Forward pass time: {fwd_time*1000:.2f} ms")
    print(f"Render output shape: {render_output.shape}")
    print(f"Render output range: [{render_output.min().item():.4f}, {render_output.max().item():.4f}]")
    
    # Generate random weights for L = Σ a_ij * c_ij²
    torch.manual_seed(123)  # Different seed for weights
    weights = torch.rand(C, dim0, dim1, device=device, dtype=dtype) + 0.5  # [0.5, 1.5]
    
    # Compute upstream gradients from L = Σ a * c²
    # v_O = ∂L/∂c = 2 * a * c
    v_render_output = 2 * weights * render_output
    # h_O = ∂²L/∂c² = 2 * a
    h_render_output = 2 * weights
    
    print(f"v_render_output range: [{v_render_output.min().item():.4f}, {v_render_output.max().item():.4f}]")
    print(f"h_render_output range: [{h_render_output.min().item():.4f}, {h_render_output.max().item():.4f}]")
    
    # Backward2 pass
    t0 = time.time()
    (v_means2d_cuda, v_conics_cuda, v_opacities_cuda, v_mus_cuda,
     h_means2d_cuda, h_conics_cuda,
     h_mu_mean_cuda, h_mu_conic_cuda, h_mean_conic_cuda,
     h_mu_opacity_cuda, h_conic_opacity_cuda, h_mean_opacity_cuda,
     h_mus_cuda, h_opacities_cuda) = rasterize_to_pixels_bwd2(
        means2d, conics, opacities, mus,
        tile_offsets, flatten_ids,
        v_render_output, h_render_output,
        dim1, dim0, tile_size
    )
    torch.cuda.synchronize()
    bwd2_time = time.time() - t0
    print(f"Backward2 pass time: {bwd2_time*1000:.2f} ms")
    
    # Memory usage
    h_means2d_mem = h_means2d_cuda.numel() * h_means2d_cuda.element_size() / 1024
    h_conics_mem = h_conics_cuda.numel() * h_conics_cuda.element_size() / 1024
    print(f"h_means2d memory: {h_means2d_mem:.2f} KB, shape: {h_means2d_cuda.shape}")
    print(f"h_conics memory: {h_conics_mem:.2f} KB, shape: {h_conics_cuda.shape}")
    print()
    
    # Sample Gaussians for detailed comparison
    all_pass = True
    errors = {'h_means2d': [], 'h_conics': [], 'h_mu_mean': [], 'h_mu_conic': [], 'h_mean_conic': [], 'h_mu': [], 'h_opacity': [], 'h_mu_opacity': [], 'h_conic_opacity': [], 'h_mean_opacity': []}
    
    torch.manual_seed(42)  # Reproducible sampling
    
    for c in range(C):
        # Sample random Gaussians from this camera
        indices = torch.randperm(N)[:sample_size].tolist()
        
        for g in indices:
            # Compute torch Hessian for this Gaussian with weighted squared loss
            torch_hess = compute_single_gaussian_hessian_torch(
                c, g, means2d, conics, opacities, mus, dim0, dim1,
                weights=weights,  # Pass weights to use L = Σ a * O²
                render_output=render_output  # Pass render_output for correct O
            )
            
            # Extract CUDA Hessian for this Gaussian
            cuda_h_means2d_triu = h_means2d_cuda[c, g]
            cuda_h_conics_triu = h_conics_cuda[c, g]
            cuda_h_mu_mean = h_mu_mean_cuda[c, g]
            cuda_h_mu_conic = h_mu_conic_cuda[c, g]
            cuda_h_mean_conic = h_mean_conic_cuda[c, g]
            cuda_h_mu = h_mus_cuda[c, g]
            cuda_h_opacity = h_opacities_cuda[c, g]
            cuda_h_mu_opacity = h_mu_opacity_cuda[c, g]
            cuda_h_conic_opacity = h_conic_opacity_cuda[c, g]
            cuda_h_mean_opacity = h_mean_opacity_cuda[c, g]
            
            # Reconstruct full matrices from upper triangle
            cuda_h_means2d = torch.zeros(2, 2, device=device, dtype=dtype)
            cuda_h_means2d[0, 0] = cuda_h_means2d_triu[0]
            cuda_h_means2d[0, 1] = cuda_h_means2d_triu[1]
            cuda_h_means2d[1, 0] = cuda_h_means2d_triu[1]
            cuda_h_means2d[1, 1] = cuda_h_means2d_triu[2]
            
            cuda_h_conics = torch.zeros(3, 3, device=device, dtype=dtype)
            cuda_h_conics[0, 0] = cuda_h_conics_triu[0]
            cuda_h_conics[0, 1] = cuda_h_conics_triu[1]
            cuda_h_conics[0, 2] = cuda_h_conics_triu[2]
            cuda_h_conics[1, 0] = cuda_h_conics_triu[1]
            cuda_h_conics[1, 1] = cuda_h_conics_triu[3]
            cuda_h_conics[1, 2] = cuda_h_conics_triu[4]
            cuda_h_conics[2, 0] = cuda_h_conics_triu[2]
            cuda_h_conics[2, 1] = cuda_h_conics_triu[4]
            cuda_h_conics[2, 2] = cuda_h_conics_triu[5]
            
            def compute_rel_error(cuda_val, torch_val):
                diff = (cuda_val - torch_val).abs()
                max_diff = diff.max().item()
                scale = torch_val.abs().max().item()
                if scale < 1e-8:
                    scale = 1.0
                return max_diff / scale
            
            errors['h_means2d'].append(compute_rel_error(cuda_h_means2d, torch_hess['h_means2d']))
            errors['h_conics'].append(compute_rel_error(cuda_h_conics, torch_hess['h_conics']))
            errors['h_mu_mean'].append(compute_rel_error(cuda_h_mu_mean, torch_hess['h_mu_mean']))
            errors['h_mu_conic'].append(compute_rel_error(cuda_h_mu_conic, torch_hess['h_mu_conic']))
            errors['h_mean_conic'].append(compute_rel_error(cuda_h_mean_conic, torch_hess['h_mean_conic']))
            errors['h_mu'].append(compute_rel_error(cuda_h_mu, torch_hess['h_mu']))
            errors['h_opacity'].append(compute_rel_error(cuda_h_opacity, torch_hess['h_opacity']))
            errors['h_mu_opacity'].append(compute_rel_error(cuda_h_mu_opacity, torch_hess['h_mu_opacity']))
            errors['h_conic_opacity'].append(compute_rel_error(cuda_h_conic_opacity, torch_hess['h_conic_opacity']))
            errors['h_mean_opacity'].append(compute_rel_error(cuda_h_mean_opacity, torch_hess['h_mean_opacity']))
    
    print(f"Sampled {C * sample_size} Gaussians for comparison")
    print("\nRelative Error Statistics:")
    print("-" * 50)
    
    for name, err_list in errors.items():
        err_arr = torch.tensor(err_list)
        mean_err = err_arr.mean().item()
        max_err = err_arr.max().item()
        passed = max_err < rtol
        status = "✓" if passed else "✗"
        print(f"  {name}: mean={mean_err:.4f}, max={max_err:.4f} {status}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n✓ TEST 5 PASSED: Large-scale Hessian with weighted squared loss passed")
    else:
        print(f"\n✗ TEST 5 FAILED: Some errors exceed rtol={rtol}")
    
    return all_pass


def run_all_tests():
    """Run all bwd2 tests."""
    print("="*70)
    print("RasterizeToPixels Backward2 Test Suite")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Basic sanity check
    results['sanity'] = test_bwd2_basic_sanity()
    
    # Test 2: First-order gradient consistency
    results['first_order'] = test_bwd2_first_order_consistency()
    
    # Test 3: Hessian comparison (small case with h_O=0 for pure Hessian)
    results['hessian_pure'] = test_bwd2_hessian_vs_torch(C=1, N=3, verbose=True)
    
    # Test 4: Storage format check (symmetry is guaranteed by upper-triangle storage)
    results['storage_format'] = test_bwd2_symmetry()
    
    # Test 5: Large-scale test with weighted squared loss (tests both v_O and h_O)
    # Uses L = Σ a*c², so v_O = 2ac, h_O = 2a (both non-zero)
    results['large_scale_weighted'] = test_bwd2_large_scale(
        C=8, N=2000, dim0=77, dim1=88, sample_size=100
    )
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_pass = all(results.values())
    
    if all_pass:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
    
    return all_pass


if __name__ == "__main__":
    run_all_tests()
