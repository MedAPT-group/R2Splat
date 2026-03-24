"""
Test script for WorldToVolumeMapping module.

Validates the CUDA implementation against the PyTorch autograd
reference implementation by comparing:
1. Forward outputs (means3d_v, conics, radii)
2. Backward outputs (v_means, v_covars)
3. Second-order backward outputs (Hessians and cross-Hessians)

The module transforms:
    means  [N, 3]    (world-space Gaussian centres)
    covars [N, 6]    (upper-tri covariance: C00,C01,C02,C11,C12,C22)
    max_scales [N]   (max activated scale per Gaussian, non-differentiable)
    ranges [3]       (physical extent per axis)
    centers [3]      (physical centre per axis)
    ->  means3d_v [N, 3]  (voxel-space centres)
        conics    [N, 6]  (inverse covariance upper-tri)
        radii     [N, 3]  (integer radii per axis, int32)

Pipeline:
    r[k] = dim[k] / range[k]
    means3d_v[k] = (means[k] - center[k]) * r[k] + dim[k]/2
    Sigma_v[i][j] = r[i]*r[j]*covars[i][j] + eps3d*delta_{ij}
    conics = Sigma_v^{-1}
    radii[k] = ceil(radius_clip * max_scales * r[k])

Usage:
    python tests/test_world_to_volume_mapping.py

Author: Dy164
Date: 2026-02-21
"""

import torch
import sys
import os
import math

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.data import Params, Data, Config, ForwardResult, GradResult, HessResult
from gsplat2ct.ops.world_to_volume_mapping import (
    WorldToVolumeMappingCUDA,
    WorldToVolumeMappingAutograd,
    create_world_to_volume_mapping,
    _compute_world_to_volume_mapping,
)


class ParamWrapper:
    """Simple wrapper to satisfy ParameterLike protocol."""
    def __init__(self, data: torch.Tensor):
        self.data = data


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_random_positive_definite_covars(
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """Generate N random positive-definite 3x3 covariance matrices in upper-tri format.

    Returns:
        covars: (N, 6) upper-tri (C00, C01, C02, C11, C12, C22)
    """
    torch.manual_seed(seed)
    # Random lower-triangular matrices → C = L @ L^T ensures PD
    L = torch.randn(N, 3, 3, device=device, dtype=dtype) * 0.5
    # Make diagonal positive and not too small
    L[:, 0, 0] = L[:, 0, 0].abs() + 0.3
    L[:, 1, 1] = L[:, 1, 1].abs() + 0.3
    L[:, 2, 2] = L[:, 2, 2].abs() + 0.3
    L[:, 0, 1] = 0
    L[:, 0, 2] = 0
    L[:, 1, 2] = 0
    C = L @ L.transpose(-1, -2)  # (N, 3, 3) PD

    covars = torch.stack([
        C[:, 0, 0], C[:, 0, 1], C[:, 0, 2],
        C[:, 1, 1], C[:, 1, 2], C[:, 2, 2],
    ], dim=-1)  # (N, 6)
    return covars


def generate_test_data(
    N: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
    dim2: int = 64,
    dim1: int = 48,
    dim0: int = 32,
):
    """Generate a complete test dataset for WorldToVolumeMapping.

    Returns:
        means: (N, 3)
        covars: (N, 6)
        max_scales: (N,)
        ranges: (3,)
        centers: (3,)
    """
    if device is None:
        device = torch.device("cuda")

    torch.manual_seed(seed)
    # Means: random positions within reasonable bounds
    means = torch.randn(N, 3, device=device, dtype=dtype) * 2.0

    # Covars: positive-definite
    covars = generate_random_positive_definite_covars(N, device, dtype, seed=seed + 10)

    # Max scales: positive values
    torch.manual_seed(seed + 20)
    max_scales = torch.rand(N, device=device, dtype=dtype) * 1.5 + 0.3

    # Physical extents and centres
    ranges = torch.tensor([10.0, 8.0, 6.0], device=device, dtype=dtype)
    centers = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)

    return means, covars, max_scales, ranges, centers


def generate_random_symmetric_matrix(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a random symmetric matrix of size (dim, dim)."""
    M = torch.randn(dim, dim, device=device, dtype=dtype)
    return 0.5 * (M + M.T)


# =============================================================================
# Test 1: Forward Consistency (CUDA vs Autograd)
# =============================================================================

def test_forward_consistency(
    N: int = 100,
    dim2: int = 64,
    dim1: int = 48,
    dim0: int = 32,
    eps3d: float = 1e-4,
    radius_clip: float = 3.0,
    dtype: torch.dtype = torch.float64,
):
    """Test that CUDA and autograd forward passes produce the same results."""
    print(f"\n{'='*60}")
    print(f"Test Forward Consistency: N={N}, dims=({dim2},{dim1},{dim0}), dtype={dtype}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    means, covars, max_scales, ranges, centers = generate_test_data(
        N, device, dtype, dim2=dim2, dim1=dim1, dim0=dim0
    )

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', eps3d)
    config.set('radius_clip', radius_clip)

    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('covars', ParamWrapper(covars))
    params.add_param('max_scales', ParamWrapper(max_scales))
    data = Data()
    data.set('ranges', ranges)
    data.set('centers', centers)

    # CUDA forward
    cuda_op = WorldToVolumeMappingCUDA()
    cuda_op.initialize(params, config)
    cuda_result = cuda_op.forward_only(params, data, config)
    means_v_cuda = cuda_result.outputs['means3d_v']
    conics_cuda = cuda_result.outputs['conics']
    radii_cuda = cuda_result.outputs['radii']

    # Autograd forward
    ag_op = WorldToVolumeMappingAutograd()
    ag_op.initialize(params, config)
    ag_result = ag_op.forward_only(params, data, config)
    means_v_ag = ag_result.outputs['means3d_v']
    conics_ag = ag_result.outputs['conics']
    radii_ag = ag_result.outputs['radii']

    # Compare
    atol = 1e-10 if dtype == torch.float64 else 2e-5

    diff_m = (means_v_cuda - means_v_ag).abs()
    diff_c = (conics_cuda - conics_ag).abs()
    max_diff_m = diff_m.max().item()
    max_diff_c = diff_c.max().item()
    radii_match = torch.equal(radii_cuda, radii_ag)

    print(f"  means3d_v max diff:  {max_diff_m:.2e}")
    print(f"  conics max diff:     {max_diff_c:.2e}")
    print(f"  radii match:         {'PASS' if radii_match else 'FAIL'}")

    for i, name in enumerate(['x', 'y', 'z']):
        print(f"    means_v_{name}: {diff_m[:, i].max().item():.2e}")
    for i, name in enumerate(['P00', 'P01', 'P02', 'P11', 'P12', 'P22']):
        print(f"    conics_{name}: {diff_c[:, i].max().item():.2e}")

    m_ok = max_diff_m < atol
    c_ok = max_diff_c < atol
    ok = m_ok and c_ok and radii_match

    print(f"  Forward match: {'PASS' if ok else 'FAIL'} (atol={atol:.0e})")
    return ok


# =============================================================================
# Test 2: Backward Consistency (CUDA vs Autograd)
# =============================================================================

def test_backward_consistency(
    N: int = 100,
    dim2: int = 64,
    dim1: int = 48,
    dim0: int = 32,
    eps3d: float = 1e-4,
    dtype: torch.dtype = torch.float64,
):
    """Test that CUDA and autograd backward passes produce the same gradients."""
    print(f"\n{'='*60}")
    print(f"Test Backward Consistency: N={N}, dtype={dtype}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    means, covars, max_scales, ranges, centers = generate_test_data(
        N, device, dtype, dim2=dim2, dim1=dim1, dim0=dim0
    )

    # Generate random upstream gradients
    torch.manual_seed(123)
    v_means3d_v = torch.randn(N, 3, device=device, dtype=dtype)
    v_conics = torch.randn(N, 6, device=device, dtype=dtype)

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', eps3d)

    # --- CUDA backward ---
    params_c = Params()
    params_c.add_param('means', ParamWrapper(means.clone()))
    params_c.add_param('covars', ParamWrapper(covars.clone()))
    params_c.add_param('max_scales', ParamWrapper(max_scales.clone()))
    data_c = Data()
    data_c.set('ranges', ranges)
    data_c.set('centers', centers)
    data_c.set('v_means3d_v', v_means3d_v.clone())
    data_c.set('v_conics', v_conics.clone())

    cuda_op = WorldToVolumeMappingCUDA()
    cuda_op.initialize(params_c, config)
    cuda_fwd, cuda_grad = cuda_op.forward_backward1(params_c, data_c, config)

    v_means_cuda = cuda_grad.param_grads['means']
    v_covars_cuda = cuda_grad.param_grads['covars']

    # --- Autograd backward ---
    params_a = Params()
    params_a.add_param('means', ParamWrapper(means.clone()))
    params_a.add_param('covars', ParamWrapper(covars.clone()))
    params_a.add_param('max_scales', ParamWrapper(max_scales.clone()))
    data_a = Data()
    data_a.set('ranges', ranges)
    data_a.set('centers', centers)
    data_a.set('v_means3d_v', v_means3d_v.clone())
    data_a.set('v_conics', v_conics.clone())

    ag_op = WorldToVolumeMappingAutograd()
    ag_op.initialize(params_a, config)
    ag_fwd, ag_grad = ag_op.forward_backward1(params_a, data_a, config)

    v_means_ag = ag_grad.param_grads['means']
    v_covars_ag = ag_grad.param_grads['covars']

    # Compare forward outputs first
    fwd_diff_m = (cuda_fwd.outputs['means3d_v'] - ag_fwd.outputs['means3d_v']).abs().max().item()
    fwd_diff_c = (cuda_fwd.outputs['conics'] - ag_fwd.outputs['conics']).abs().max().item()
    print(f"  Forward means3d_v diff: {fwd_diff_m:.2e}")
    print(f"  Forward conics diff:    {fwd_diff_c:.2e}")

    # Compare gradients
    diff_vm = (v_means_cuda - v_means_ag).abs()
    diff_vc = (v_covars_cuda - v_covars_ag).abs()
    max_diff_vm = diff_vm.max().item()
    max_diff_vc = diff_vc.max().item()

    # f32 has larger rounding due to matrix inversion amplifying error
    atol = 1e-8 if dtype == torch.float64 else 5e-3

    for i, name in enumerate(['x', 'y', 'z']):
        print(f"  v_means_{name}: {diff_vm[:, i].max().item():.2e}")
    for i, name in enumerate(['s00', 's01', 's02', 's11', 's12', 's22']):
        print(f"  v_covars_{name}: {diff_vc[:, i].max().item():.2e}")

    print(f"  v_means max diff:  {max_diff_vm:.2e}")
    print(f"  v_covars max diff: {max_diff_vc:.2e}")

    m_ok = max_diff_vm < atol
    c_ok = max_diff_vc < atol
    print(f"  Backward means match:  {'PASS' if m_ok else 'FAIL'} (atol={atol:.0e})")
    print(f"  Backward covars match: {'PASS' if c_ok else 'FAIL'} (atol={atol:.0e})")

    return m_ok and c_ok


# =============================================================================
# Test 3: Backward vs Numerical (Finite Differences)
# =============================================================================

def test_backward_vs_numerical(
    N: int = 8,
    dim2: int = 32,
    dim1: int = 24,
    dim0: int = 16,
    eps3d: float = 1e-4,
    dtype: torch.dtype = torch.float64,
    fd_eps: float = 1e-6,
):
    """Validate backward using finite differences (gold standard)."""
    print(f"\n{'='*60}")
    print(f"Test Backward vs Numerical (FD): N={N}, fd_eps={fd_eps}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    means, covars, max_scales, ranges, centers = generate_test_data(
        N, device, dtype, seed=99, dim2=dim2, dim1=dim1, dim0=dim0
    )

    from gsplat2ct.cuda._wrapper import (
        world_to_volume_mapping_fwd,
        world_to_volume_mapping_bwd,
    )

    radius_clip = 3.0

    # Forward
    means3d_v, conics, radii = world_to_volume_mapping_fwd(
        means, covars, max_scales, ranges, centers,
        dim2, dim1, dim0, eps3d, radius_clip,
    )

    # Random upstream gradient
    torch.manual_seed(77)
    v_means3d_v = torch.randn_like(means3d_v)
    v_conics = torch.randn_like(conics)

    # CUDA backward
    v_means_cuda, v_covars_cuda = world_to_volume_mapping_bwd(
        means, covars, ranges, centers,
        dim2, dim1, dim0, eps3d,
        v_means3d_v, v_conics,
    )

    # Helper: scalar product of upstream gradients with forward outputs
    def compute_scalar(m, c):
        mv, cn, _ = world_to_volume_mapping_fwd(
            m, c, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d, radius_clip,
        )
        return (mv * v_means3d_v).sum() + (cn * v_conics).sum()

    # Finite-difference gradient for means
    v_means_fd = torch.zeros_like(means)
    for n in range(N):
        for k in range(3):
            m_p = means.clone(); m_p[n, k] += fd_eps
            m_m = means.clone(); m_m[n, k] -= fd_eps
            v_means_fd[n, k] = (
                compute_scalar(m_p, covars).item()
                - compute_scalar(m_m, covars).item()
            ) / (2 * fd_eps)

    # Finite-difference gradient for covars
    v_covars_fd = torch.zeros_like(covars)
    for n in range(N):
        for k in range(6):
            c_p = covars.clone(); c_p[n, k] += fd_eps
            c_m = covars.clone(); c_m[n, k] -= fd_eps
            v_covars_fd[n, k] = (
                compute_scalar(means, c_p).item()
                - compute_scalar(means, c_m).item()
            ) / (2 * fd_eps)

    # Compare
    atol = 1e-5

    diff_m = (v_means_cuda - v_means_fd).abs()
    diff_c = (v_covars_cuda - v_covars_fd).abs()
    max_diff_m = diff_m.max().item()
    max_diff_c = diff_c.max().item()

    print(f"  v_means max diff vs FD:  {max_diff_m:.2e}")
    print(f"  v_covars max diff vs FD: {max_diff_c:.2e}")

    for i, name in enumerate(['x', 'y', 'z']):
        print(f"    v_means_{name}: {diff_m[:, i].max().item():.2e}")
    for i, name in enumerate(['s00', 's01', 's02', 's11', 's12', 's22']):
        print(f"    v_covars_{name}: {diff_c[:, i].max().item():.2e}")

    m_ok = max_diff_m < atol
    c_ok = max_diff_c < atol
    print(f"  Numerical means match:  {'PASS' if m_ok else 'FAIL'} (atol={atol:.0e})")
    print(f"  Numerical covars match: {'PASS' if c_ok else 'FAIL'} (atol={atol:.0e})")

    return m_ok and c_ok


# =============================================================================
# Test 4: Bwd2 — Downstream Quadratic Loss Construction
# =============================================================================

def compute_quadratic_downstream(
    means3d_v: torch.Tensor,    # [N, 3] detached
    conics: torch.Tensor,       # [N, 6] detached
    opacity: torch.Tensor,      # [N] detached
    W_mv: torch.Tensor,         # [3, 3] symmetric — d^2L/d(means3d_v)d(means3d_v)
    W_cn: torch.Tensor,         # [6, 6] symmetric — d^2L/d(conics)d(conics)
    A_mv_cn: torch.Tensor,      # [3, 6] — d^2L/d(means3d_v)d(conics)
    A_mv_opa: torch.Tensor,     # [3, 1] — d^2L/d(means3d_v)d(opacity)
    A_cn_opa: torch.Tensor,     # [6, 1] — d^2L/d(conics)d(opacity)
):
    """Analytically compute downstream quantities for bwd2 kernel.

    For the quadratic loss:
        L = 0.5 * mv^T W_mv mv + 0.5 * cn^T W_cn cn
            + mv^T A_mv_cn cn + mv^T A_mv_opa o + cn^T A_cn_opa o

    Downstream quantities:
        v_means3d_v = W_mv @ mv + A_mv_cn @ cn + A_mv_opa * o
        v_conics = W_cn @ cn + A_mv_cn^T @ mv + A_cn_opa * o
        h_means3d_v = W_mv upper-tri (N, 6) of 3x3
        h_conics = W_cn upper-tri (N, 21) of 6x6
        h_means3d_v_conics = A_mv_cn (N, 18)  3x6 row-major
        h_means3d_v_opacity = A_mv_opa (N, 3)
        h_conics_opacity = A_cn_opa (N, 6)
    """
    N = means3d_v.shape[0]
    device = means3d_v.device
    dtype = means3d_v.dtype

    # First-order gradients
    v_means3d_v = (
        means3d_v @ W_mv
        + conics @ A_mv_cn.T
        + opacity.unsqueeze(-1) * A_mv_opa.squeeze(-1).unsqueeze(0)
    )
    v_conics = (
        conics @ W_cn
        + means3d_v @ A_mv_cn
        + opacity.unsqueeze(-1) * A_cn_opa.squeeze(-1).unsqueeze(0)
    )

    # h_means3d_v: upper-tri of W_mv [3x3] → 6 elements
    h_means3d_v = torch.zeros(N, 6, device=device, dtype=dtype)
    idx = 0
    for i in range(3):
        for j in range(i, 3):
            h_means3d_v[:, idx] = W_mv[i, j]
            idx += 1

    # h_conics: upper-tri of W_cn [6x6] → 21 elements
    h_conics = torch.zeros(N, 21, device=device, dtype=dtype)
    idx = 0
    for i in range(6):
        for j in range(i, 6):
            h_conics[:, idx] = W_cn[i, j]
            idx += 1

    # h_means3d_v_conics: A_mv_cn [3, 6] row-major → [N, 18]
    h_means3d_v_conics = A_mv_cn.reshape(1, 18).expand(N, -1).contiguous()

    # h_means3d_v_opacity: A_mv_opa [3, 1] → [N, 3]
    h_means3d_v_opacity = A_mv_opa.squeeze(-1).unsqueeze(0).expand(N, -1).contiguous()

    # h_conics_opacity: A_cn_opa [6, 1] → [N, 6]
    h_conics_opacity = A_cn_opa.squeeze(-1).unsqueeze(0).expand(N, -1).contiguous()

    return {
        'v_means3d_v': v_means3d_v,
        'v_conics': v_conics,
        'h_means3d_v': h_means3d_v,
        'h_conics': h_conics,
        'h_means3d_v_conics': h_means3d_v_conics,
        'h_means3d_v_opacity': h_means3d_v_opacity,
        'h_conics_opacity': h_conics_opacity,
    }


def compute_ground_truth_autograd(
    means: torch.Tensor,        # [N, 3] detached
    covars: torch.Tensor,       # [N, 6] detached
    max_scales: torch.Tensor,   # [N] detached
    ranges: torch.Tensor,       # [3] detached
    centers: torch.Tensor,      # [3] detached
    opacity: torch.Tensor,      # [N] detached
    dim2: int,
    dim1: int,
    dim0: int,
    eps3d: float,
    radius_clip: float,
    W_mv: torch.Tensor,
    W_cn: torch.Tensor,
    A_mv_cn: torch.Tensor,
    A_mv_opa: torch.Tensor,
    A_cn_opa: torch.Tensor,
):
    """Compute ground-truth Hessians using torch.autograd double-backward.

    Constructs a quadratic loss in (means3d_v, conics, opacity)
    then chain-rules through the world-to-volume mapping to get
    Hessian blocks w.r.t. (means, covars).
    """
    N = means.shape[0]
    device = means.device
    dtype = means.dtype

    # Create leaf tensors
    m = means.detach().clone().requires_grad_(True)
    c = covars.detach().clone().requires_grad_(True)
    o = opacity.detach().clone().requires_grad_(True)

    # Forward
    means3d_v, conics, _ = _compute_world_to_volume_mapping(
        m, c, max_scales, ranges, centers,
        dim2, dim1, dim0, eps3d, radius_clip
    )

    # Quadratic loss (summed over all Gaussians)
    # 0.5 * mv^T W_mv mv
    Wmv = means3d_v @ W_mv.T
    term1 = 0.5 * (means3d_v * Wmv).sum()
    # 0.5 * cn^T W_cn cn
    Wcn = conics @ W_cn.T
    term2 = 0.5 * (conics * Wcn).sum()
    # mv^T A_mv_cn cn
    term3 = ((means3d_v @ A_mv_cn) * conics).sum()
    # mv^T A_mv_opa o
    term4 = ((means3d_v @ A_mv_opa.squeeze(-1)) * o).sum()
    # cn^T A_cn_opa o
    term5 = ((conics @ A_cn_opa.squeeze(-1)) * o).sum()
    L_total = term1 + term2 + term3 + term4 + term5

    # First-order gradients
    v_m, v_c, v_o = torch.autograd.grad(
        L_total, [m, c, o], create_graph=True
    )

    # Helper function for second-order
    def hess_block(v_source, source_var, target_var, source_dim, target_dim):
        """Compute d(v_source)/d(target_var) block."""
        H = torch.zeros(N, source_dim, target_dim, device=device, dtype=dtype)
        for a in range(source_dim):
            selector = torch.zeros_like(v_source)
            selector[:, a] = 1.0
            grads = torch.autograd.grad(
                v_source, target_var, grad_outputs=selector, retain_graph=True,
                allow_unused=True,
            )[0]
            if grads is not None:
                if grads.dim() == 2:
                    H[:, a, :] = grads
                else:
                    H[:, a, 0] = grads
        return H

    # H_{m,m} [N, 3, 3]
    h_mm = hess_block(v_m, m, m, 3, 3)
    # H_{c,c} [N, 6, 6]
    h_cc = hess_block(v_c, c, c, 6, 6)
    # H_{m,c} [N, 3, 6]
    h_mc = hess_block(v_m, m, c, 3, 6)
    # H_{m,o} [N, 3, 1] → [N, 3]
    h_mo_full = torch.zeros(N, 3, device=device, dtype=dtype)
    for a in range(3):
        selector = torch.zeros_like(v_m)
        selector[:, a] = 1.0
        g = torch.autograd.grad(
            v_m, o, grad_outputs=selector, retain_graph=True, allow_unused=True
        )[0]
        if g is not None:
            h_mo_full[:, a] = g
    # H_{c,o} [N, 6]
    h_co_full = torch.zeros(N, 6, device=device, dtype=dtype)
    for a in range(6):
        selector = torch.zeros_like(v_c)
        selector[:, a] = 1.0
        g = torch.autograd.grad(
            v_c, o, grad_outputs=selector, retain_graph=True, allow_unused=True
        )[0]
        if g is not None:
            h_co_full[:, a] = g

    return {
        'v_m': v_m.detach(),
        'v_c': v_c.detach(),
        'h_mm': h_mm.detach(),
        'h_cc': h_cc.detach(),
        'h_mc': h_mc.detach(),
        'h_mo': h_mo_full.detach(),
        'h_co': h_co_full.detach(),
    }


# =============================================================================
# Test 5: Bwd2 CUDA vs Autograd Ground Truth
# =============================================================================

def test_bwd2_cuda_vs_autograd_ground_truth(
    N: int = 16,
    dim2: int = 32,
    dim1: int = 24,
    dim0: int = 16,
    eps3d: float = 1e-4,
    radius_clip: float = 3.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test CUDA bwd2 against autograd ground truth.

    Uses a random quadratic loss function to provide nonzero downstream
    Hessians, then compares all output blocks.
    """
    print(f"\n{'='*70}")
    print(f"Test bwd2 CUDA vs Autograd Ground Truth")
    print(f"  N={N}, dims=({dim2},{dim1},{dim0}), dtype={dtype}, seed={seed}")
    print(f"{'='*70}")

    device = torch.device("cuda")

    means, covars, max_scales, ranges, centers = generate_test_data(
        N, device, dtype, seed=seed, dim2=dim2, dim1=dim1, dim0=dim0
    )

    torch.manual_seed(seed + 100)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1

    # Random quadratic coefficients
    torch.manual_seed(seed + 200)
    W_mv = generate_random_symmetric_matrix(3, device, dtype)
    W_cn = generate_random_symmetric_matrix(6, device, dtype)
    A_mv_cn = torch.randn(3, 6, device=device, dtype=dtype)
    A_mv_opa = torch.randn(3, 1, device=device, dtype=dtype)
    A_cn_opa = torch.randn(6, 1, device=device, dtype=dtype)

    # --- Ground truth via autograd double-backward ---
    gt = compute_ground_truth_autograd(
        means, covars, max_scales, ranges, centers, opacity,
        dim2, dim1, dim0, eps3d, radius_clip,
        W_mv, W_cn, A_mv_cn, A_mv_opa, A_cn_opa,
    )

    # --- CUDA bwd2 via ops interface ---
    from gsplat2ct.cuda._wrapper import world_to_volume_mapping_fwd
    means3d_v, conics, radii = world_to_volume_mapping_fwd(
        means, covars, max_scales, ranges, centers,
        dim2, dim1, dim0, eps3d, radius_clip,
    )

    downstream = compute_quadratic_downstream(
        means3d_v, conics, opacity,
        W_mv, W_cn, A_mv_cn, A_mv_opa, A_cn_opa,
    )

    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('covars', ParamWrapper(covars))
    params.add_param('max_scales', ParamWrapper(max_scales))
    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', eps3d)
    config.set('radius_clip', radius_clip)
    data_obj = Data()
    data_obj.set('ranges', ranges)
    data_obj.set('centers', centers)
    for key, val in downstream.items():
        data_obj.set(key, val)

    cuda_op = WorldToVolumeMappingCUDA()
    cuda_op.initialize(params, config)
    _, grad_c, hess_c = cuda_op.forward_backward2(params, data_obj, config)

    # --- Compare ---
    all_pass = True
    atol = 1e-7 if dtype == torch.float64 else 1e-3
    rtol = 1e-5 if dtype == torch.float64 else 1e-2

    def compare(name, cuda_val, gt_val, atol=atol, rtol=rtol):
        nonlocal all_pass
        diff = (cuda_val - gt_val).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        gt_max = gt_val.abs().max().item()

        rel_diff = diff / (gt_val.abs() + 1e-15)
        max_rel = rel_diff.max().item()

        ok = max_diff < atol or max_rel < rtol
        status = "PASS" if ok else "FAIL"

        print(f"  {name:35s}: max_abs={max_diff:.2e}, mean_abs={mean_diff:.2e}, "
              f"max_rel={max_rel:.2e}, gt_max={gt_max:.2e} [{status}]")

        if not ok:
            all_pass = False
            worst_idx = diff.argmax()
            flat_cuda = cuda_val.reshape(-1)
            flat_gt = gt_val.reshape(-1)
            print(f"    worst: cuda={flat_cuda[worst_idx].item():.10e}, "
                  f"gt={flat_gt[worst_idx].item():.10e}")
        return ok

    # First-order gradients
    compare("v_means (grad)", grad_c.param_grads['means'], gt['v_m'])
    compare("v_covars (grad)", grad_c.param_grads['covars'], gt['v_c'])

    # H_{means, means}: upper-tri 3x3 → 6
    h_mm_utri = torch.zeros(N, 6, device=device, dtype=dtype)
    idx = 0
    for i in range(3):
        for j in range(i, 3):
            h_mm_utri[:, idx] = gt['h_mm'][:, i, j]
            idx += 1
    compare("h_means (upper-tri 3x3)", hess_c.param_hess['means'], h_mm_utri)

    # H_{covars, covars}: upper-tri 6x6 → 21
    h_cc_utri = torch.zeros(N, 21, device=device, dtype=dtype)
    idx = 0
    for i in range(6):
        for j in range(i, 6):
            h_cc_utri[:, idx] = gt['h_cc'][:, i, j]
            idx += 1
    compare("h_covars (upper-tri 6x6)", hess_c.param_hess['covars'], h_cc_utri)

    # H_{means, covars}: 3x6 row-major → 18
    h_mc_flat = gt['h_mc'].reshape(N, 18)
    compare("h_means_covars (3x6)", hess_c.param_hess['means_covars'], h_mc_flat)

    # Cross-Hessians with opacity
    compare("h_means_opacity (3)", hess_c.param_hess['means_opacity'], gt['h_mo'])
    compare("h_covars_opacity (6)", hess_c.param_hess['covars_opacity'], gt['h_co'])

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# =============================================================================
# Test 6: Bwd2 Autograd Op vs Ground Truth
# =============================================================================

def test_bwd2_autograd_op_vs_ground_truth(
    N: int = 16,
    dim2: int = 32,
    dim1: int = 24,
    dim0: int = 16,
    eps3d: float = 1e-4,
    radius_clip: float = 3.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test WorldToVolumeMappingAutograd.forward_backward2 against full autograd ground truth."""
    print(f"\n{'='*70}")
    print(f"Test bwd2 Autograd Op vs Ground Truth")
    print(f"  N={N}, dims=({dim2},{dim1},{dim0}), dtype={dtype}")
    print(f"{'='*70}")

    device = torch.device("cuda")

    means, covars, max_scales, ranges, centers = generate_test_data(
        N, device, dtype, seed=seed, dim2=dim2, dim1=dim1, dim0=dim0
    )

    torch.manual_seed(seed + 100)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1

    torch.manual_seed(seed + 200)
    W_mv = generate_random_symmetric_matrix(3, device, dtype)
    W_cn = generate_random_symmetric_matrix(6, device, dtype)
    A_mv_cn = torch.randn(3, 6, device=device, dtype=dtype)
    A_mv_opa = torch.randn(3, 1, device=device, dtype=dtype)
    A_cn_opa = torch.randn(6, 1, device=device, dtype=dtype)

    # Ground truth
    gt = compute_ground_truth_autograd(
        means, covars, max_scales, ranges, centers, opacity,
        dim2, dim1, dim0, eps3d, radius_clip,
        W_mv, W_cn, A_mv_cn, A_mv_opa, A_cn_opa,
    )

    # Autograd Op
    with torch.no_grad():
        means3d_v, conics, radii = _compute_world_to_volume_mapping(
            means, covars, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d, radius_clip
        )

    downstream = compute_quadratic_downstream(
        means3d_v, conics, opacity,
        W_mv, W_cn, A_mv_cn, A_mv_opa, A_cn_opa,
    )

    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('covars', ParamWrapper(covars))
    params.add_param('max_scales', ParamWrapper(max_scales))
    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', eps3d)
    config.set('radius_clip', radius_clip)
    data_obj = Data()
    data_obj.set('ranges', ranges)
    data_obj.set('centers', centers)
    for key, val in downstream.items():
        data_obj.set(key, val)

    ag_op = WorldToVolumeMappingAutograd()
    ag_op.initialize(params, config)
    _, grad_a, hess_a = ag_op.forward_backward2(params, data_obj, config)

    all_pass = True
    atol = 1e-8 if dtype == torch.float64 else 1e-4
    rtol = 1e-5 if dtype == torch.float64 else 1e-3

    def compare(name, val, gt_val, atol=atol, rtol=rtol):
        nonlocal all_pass
        diff = (val - gt_val).abs()
        max_diff = diff.max().item()
        gt_max = gt_val.abs().max().item()
        rel_diff = diff / (gt_val.abs() + 1e-15)
        max_rel = rel_diff.max().item()

        ok = max_diff < atol or max_rel < rtol
        status = "PASS" if ok else "FAIL"
        print(f"  {name:35s}: max_abs={max_diff:.2e}, max_rel={max_rel:.2e}, "
              f"gt_max={gt_max:.2e} [{status}]")
        if not ok:
            all_pass = False
        return ok

    compare("v_means (grad)", grad_a.param_grads['means'], gt['v_m'])
    compare("v_covars (grad)", grad_a.param_grads['covars'], gt['v_c'])

    # h_means: upper-tri 3x3
    h_mm_utri = torch.zeros(N, 6, device=device, dtype=dtype)
    idx = 0
    for i in range(3):
        for j in range(i, 3):
            h_mm_utri[:, idx] = gt['h_mm'][:, i, j]
            idx += 1
    compare("h_means (upper-tri)", hess_a.param_hess['means'], h_mm_utri)

    # h_covars: upper-tri 6x6
    h_cc_utri = torch.zeros(N, 21, device=device, dtype=dtype)
    idx = 0
    for i in range(6):
        for j in range(i, 6):
            h_cc_utri[:, idx] = gt['h_cc'][:, i, j]
            idx += 1
    compare("h_covars (upper-tri)", hess_a.param_hess['covars'], h_cc_utri)

    compare("h_means_covars", hess_a.param_hess['means_covars'], gt['h_mc'].reshape(N, 18))
    compare("h_means_opacity", hess_a.param_hess['means_opacity'], gt['h_mo'])
    compare("h_covars_opacity", hess_a.param_hess['covars_opacity'], gt['h_co'])

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# =============================================================================
# Test 7: Bwd2 CUDA vs Autograd Op (Direct)
# =============================================================================

def test_bwd2_cuda_vs_autograd_op(
    N: int = 16,
    dim2: int = 32,
    dim1: int = 24,
    dim0: int = 16,
    eps3d: float = 1e-4,
    radius_clip: float = 3.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Direct comparison: CUDA forward_backward2 vs Autograd forward_backward2."""
    print(f"\n{'='*70}")
    print(f"Test bwd2 CUDA vs Autograd Op (direct)")
    print(f"  N={N}, dims=({dim2},{dim1},{dim0}), dtype={dtype}")
    print(f"{'='*70}")

    device = torch.device("cuda")

    means, covars, max_scales, ranges, centers = generate_test_data(
        N, device, dtype, seed=seed, dim2=dim2, dim1=dim1, dim0=dim0
    )

    torch.manual_seed(seed + 100)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1

    torch.manual_seed(seed + 200)
    W_mv = generate_random_symmetric_matrix(3, device, dtype)
    W_cn = generate_random_symmetric_matrix(6, device, dtype)
    A_mv_cn = torch.randn(3, 6, device=device, dtype=dtype)
    A_mv_opa = torch.randn(3, 1, device=device, dtype=dtype)
    A_cn_opa = torch.randn(6, 1, device=device, dtype=dtype)

    # Forward to get downstream inputs
    with torch.no_grad():
        means3d_v, conics, _ = _compute_world_to_volume_mapping(
            means, covars, max_scales, ranges, centers,
            dim2, dim1, dim0, eps3d, radius_clip
        )

    downstream = compute_quadratic_downstream(
        means3d_v, conics, opacity,
        W_mv, W_cn, A_mv_cn, A_mv_opa, A_cn_opa,
    )

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', eps3d)
    config.set('radius_clip', radius_clip)

    # --- CUDA ---
    params_c = Params()
    params_c.add_param('means', ParamWrapper(means.clone()))
    params_c.add_param('covars', ParamWrapper(covars.clone()))
    params_c.add_param('max_scales', ParamWrapper(max_scales.clone()))
    data_c = Data()
    data_c.set('ranges', ranges)
    data_c.set('centers', centers)
    for key, val in downstream.items():
        data_c.set(key, val.clone())

    cuda_op = WorldToVolumeMappingCUDA()
    cuda_op.initialize(params_c, config)
    _, grad_c, hess_c = cuda_op.forward_backward2(params_c, data_c, config)

    # --- Autograd ---
    params_a = Params()
    params_a.add_param('means', ParamWrapper(means.clone()))
    params_a.add_param('covars', ParamWrapper(covars.clone()))
    params_a.add_param('max_scales', ParamWrapper(max_scales.clone()))
    data_a = Data()
    data_a.set('ranges', ranges)
    data_a.set('centers', centers)
    for key, val in downstream.items():
        data_a.set(key, val.clone())

    ag_op = WorldToVolumeMappingAutograd()
    ag_op.initialize(params_a, config)
    _, grad_a, hess_a = ag_op.forward_backward2(params_a, data_a, config)

    all_pass = True
    atol = 1e-8 if dtype == torch.float64 else 1e-4
    rtol = 1e-5 if dtype == torch.float64 else 1e-3

    def compare(name, cuda_val, ag_val, atol=atol, rtol=rtol):
        nonlocal all_pass
        diff = (cuda_val - ag_val).abs()
        max_diff = diff.max().item()
        ref_max = ag_val.abs().max().item()
        rel_diff = diff / (ag_val.abs() + 1e-15)
        max_rel = rel_diff.max().item()

        ok = max_diff < atol or max_rel < rtol
        status = "PASS" if ok else "FAIL"
        print(f"  {name:35s}: max_abs={max_diff:.2e}, max_rel={max_rel:.2e}, "
              f"ref_max={ref_max:.2e} [{status}]")
        if not ok:
            all_pass = False
        return ok

    compare("v_means", grad_c.param_grads['means'], grad_a.param_grads['means'])
    compare("v_covars", grad_c.param_grads['covars'], grad_a.param_grads['covars'])
    compare("h_means", hess_c.param_hess['means'], hess_a.param_hess['means'])
    compare("h_covars", hess_c.param_hess['covars'], hess_a.param_hess['covars'])
    compare("h_means_covars", hess_c.param_hess['means_covars'], hess_a.param_hess['means_covars'])
    compare("h_means_opacity", hess_c.param_hess['means_opacity'], hess_a.param_hess['means_opacity'])
    compare("h_covars_opacity", hess_c.param_hess['covars_opacity'], hess_a.param_hess['covars_opacity'])

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# =============================================================================
# Test 8: Hessian Symmetry
# =============================================================================

def test_bwd2_hessian_symmetry(
    N: int = 8,
    dim2: int = 32,
    dim1: int = 24,
    dim0: int = 16,
    eps3d: float = 1e-4,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test that H_{m,m} and H_{c,c} are symmetric."""
    print(f"\n{'='*70}")
    print(f"Test bwd2 Hessian Symmetry")
    print(f"  N={N}, dtype={dtype}")
    print(f"{'='*70}")

    device = torch.device("cuda")
    means, covars, max_scales, ranges, centers = generate_test_data(
        N, device, dtype, seed=seed, dim2=dim2, dim1=dim1, dim0=dim0
    )

    torch.manual_seed(seed + 100)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1

    torch.manual_seed(seed + 200)
    W_mv = generate_random_symmetric_matrix(3, device, dtype)
    W_cn = generate_random_symmetric_matrix(6, device, dtype)
    A_mv_cn = torch.randn(3, 6, device=device, dtype=dtype)
    A_mv_opa = torch.randn(3, 1, device=device, dtype=dtype)
    A_cn_opa = torch.randn(6, 1, device=device, dtype=dtype)

    gt = compute_ground_truth_autograd(
        means, covars, max_scales, ranges, centers, opacity,
        dim2, dim1, dim0, eps3d, 3.0,
        W_mv, W_cn, A_mv_cn, A_mv_opa, A_cn_opa,
    )

    # Check h_mm symmetry
    h_mm = gt['h_mm']  # [N, 3, 3]
    sym_diff_mm = (h_mm - h_mm.transpose(-1, -2)).abs().max().item()

    # Check h_cc symmetry
    h_cc = gt['h_cc']  # [N, 6, 6]
    sym_diff_cc = (h_cc - h_cc.transpose(-1, -2)).abs().max().item()

    # H_cc involves double backward through matrix inverse, so numerical
    # symmetry is limited by floating-point precision in autograd
    atol_mm = 1e-10
    atol_cc = 1e-8  # relax for 6x6 Hessian through inverse
    ok_mm = sym_diff_mm < atol_mm
    ok_cc = sym_diff_cc < atol_cc

    print(f"  H_mm symmetry max diff: {sym_diff_mm:.2e} [{'PASS' if ok_mm else 'FAIL'}] (atol={atol_mm:.0e})")
    print(f"  H_cc symmetry max diff: {sym_diff_cc:.2e} [{'PASS' if ok_cc else 'FAIL'}] (atol={atol_cc:.0e})")

    return ok_mm and ok_cc


# =============================================================================
# Test 9: Edge Cases
# =============================================================================

def test_edge_cases(
    dim2: int = 32,
    dim1: int = 24,
    dim0: int = 16,
    eps3d: float = 1e-4,
    dtype: torch.dtype = torch.float64,
):
    """Test edge cases: diagonal covariance, N=0, N=1."""
    print(f"\n{'='*60}")
    print(f"Test Edge Cases")
    print(f"{'='*60}")

    device = torch.device("cuda")
    all_ok = True

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', eps3d)
    config.set('radius_clip', 3.0)

    ranges = torch.tensor([10.0, 8.0, 6.0], device=device, dtype=dtype)
    centers = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)

    # --- Diagonal covariance (identity-like) ---
    print("  Diagonal covariance (identity):")
    m = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
    c = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]], device=device, dtype=dtype)
    ms = torch.tensor([0.5], device=device, dtype=dtype)

    params = Params()
    params.add_param('means', ParamWrapper(m))
    params.add_param('covars', ParamWrapper(c))
    params.add_param('max_scales', ParamWrapper(ms))
    data = Data()
    data.set('ranges', ranges)
    data.set('centers', centers)

    cuda_op = WorldToVolumeMappingCUDA()
    cuda_op.initialize(params, config)
    result = cuda_op.forward_only(params, data, config)

    ag_op = WorldToVolumeMappingAutograd()
    ag_op.initialize(params, config)
    result_ag = ag_op.forward_only(params, data, config)

    diff_mv = (result.outputs['means3d_v'] - result_ag.outputs['means3d_v']).abs().max().item()
    diff_cn = (result.outputs['conics'] - result_ag.outputs['conics']).abs().max().item()
    radii_match = torch.equal(result.outputs['radii'], result_ag.outputs['radii'])

    ok = diff_mv < 1e-12 and diff_cn < 1e-10 and radii_match
    print(f"    means3d_v diff: {diff_mv:.2e}, conics diff: {diff_cn:.2e}, "
          f"radii match: {radii_match} {'PASS' if ok else 'FAIL'}")
    all_ok &= ok

    # Verify the output is correct analytically
    r = torch.tensor([dim2/10.0, dim1/8.0, dim0/6.0], device=device, dtype=dtype)
    expected_mv = (m[0] - centers) * r + torch.tensor([dim2/2.0, dim1/2.0, dim0/2.0],
                                                       device=device, dtype=dtype)
    ok2 = (result.outputs['means3d_v'][0] - expected_mv).abs().max().item() < 1e-10
    print(f"    Analytical means3d_v check: {'PASS' if ok2 else 'FAIL'}")
    all_ok &= ok2

    # --- N=0 ---
    print("  N=0:")
    m0 = torch.empty(0, 3, device=device, dtype=dtype)
    c0 = torch.empty(0, 6, device=device, dtype=dtype)
    ms0 = torch.empty(0, device=device, dtype=dtype)
    params0 = Params()
    params0.add_param('means', ParamWrapper(m0))
    params0.add_param('covars', ParamWrapper(c0))
    params0.add_param('max_scales', ParamWrapper(ms0))

    cuda_op0 = WorldToVolumeMappingCUDA()
    cuda_op0.initialize(params0, config)
    result0 = cuda_op0.forward_only(params0, data, config)
    ok_n0 = (result0.outputs['means3d_v'].shape == (0, 3)
             and result0.outputs['conics'].shape == (0, 6)
             and result0.outputs['radii'].shape == (0, 3))
    print(f"    shapes: means3d_v={result0.outputs['means3d_v'].shape}, "
          f"conics={result0.outputs['conics'].shape}, "
          f"radii={result0.outputs['radii'].shape} "
          f"{'PASS' if ok_n0 else 'FAIL'}")
    all_ok &= ok_n0

    # --- N=1 backward ---
    print("  N=1 backward:")
    m1 = torch.tensor([[0.5, -1.0, 2.0]], device=device, dtype=dtype)
    c1 = torch.tensor([[0.5, 0.1, -0.05, 0.8, 0.2, 0.6]], device=device, dtype=dtype)
    ms1 = torch.tensor([0.7], device=device, dtype=dtype)
    v_mv1 = torch.ones(1, 3, device=device, dtype=dtype)
    v_cn1 = torch.ones(1, 6, device=device, dtype=dtype)

    params1 = Params()
    params1.add_param('means', ParamWrapper(m1))
    params1.add_param('covars', ParamWrapper(c1))
    params1.add_param('max_scales', ParamWrapper(ms1))
    data1 = Data()
    data1.set('ranges', ranges)
    data1.set('centers', centers)
    data1.set('v_means3d_v', v_mv1)
    data1.set('v_conics', v_cn1)

    cuda_op1 = WorldToVolumeMappingCUDA()
    cuda_op1.initialize(params1, config)
    _, grad1 = cuda_op1.forward_backward1(params1, data1, config)

    vm = grad1.param_grads['means']
    vc = grad1.param_grads['covars']
    ok_shape = vm.shape == (1, 3) and vc.shape == (1, 6)
    ok_finite = torch.isfinite(vm).all().item() and torch.isfinite(vc).all().item()
    print(f"    shapes: vm={vm.shape}, vc={vc.shape} "
          f"{'PASS' if ok_shape else 'FAIL'}")
    print(f"    all finite: {'PASS' if ok_finite else 'FAIL'}")
    all_ok &= ok_shape and ok_finite

    print(f"  Edge cases: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


# =============================================================================
# Test 10: Factory Function
# =============================================================================

def test_factory_function():
    """Test the create_world_to_volume_mapping factory function."""
    print(f"\n{'='*60}")
    print(f"Test Factory Function")
    print(f"{'='*60}")

    try:
        cuda_op = create_world_to_volume_mapping("cuda")
        assert isinstance(cuda_op, WorldToVolumeMappingCUDA)
        print("  create_world_to_volume_mapping('cuda'): PASS")
    except Exception as e:
        print(f"  create_world_to_volume_mapping('cuda'): FAIL ({e})")
        return False

    try:
        ag_op = create_world_to_volume_mapping("autograd")
        assert isinstance(ag_op, WorldToVolumeMappingAutograd)
        print("  create_world_to_volume_mapping('autograd'): PASS")
    except Exception as e:
        print(f"  create_world_to_volume_mapping('autograd'): FAIL ({e})")
        return False

    try:
        create_world_to_volume_mapping("invalid")
        print("  create_world_to_volume_mapping('invalid') should raise: FAIL")
        return False
    except ValueError:
        print("  create_world_to_volume_mapping('invalid') raises ValueError: PASS")

    return True


# =============================================================================
# Test 11: Multiple Configurations
# =============================================================================

def test_bwd2_multiple_configs():
    """Run bwd2 tests with various configurations."""
    print(f"\n{'#'*70}")
    print(f"# Running bwd2 tests with multiple configurations")
    print(f"{'#'*70}")

    all_pass = True

    # Test with different sizes
    for N in [1, 8, 32]:
        ok = test_bwd2_cuda_vs_autograd_ground_truth(
            N=N, dim2=32, dim1=24, dim0=16,
            dtype=torch.float64, seed=42
        )
        all_pass = all_pass and ok

    # Test with different grid dimensions
    for d2, d1, d0 in [(16, 16, 16), (64, 32, 32), (128, 64, 48)]:
        ok = test_bwd2_cuda_vs_autograd_ground_truth(
            N=8, dim2=d2, dim1=d1, dim0=d0,
            dtype=torch.float64, seed=42
        )
        all_pass = all_pass and ok

    # Test with different eps3d values
    for eps in [1e-6, 1e-4, 1e-2]:
        ok = test_bwd2_cuda_vs_autograd_ground_truth(
            N=8, eps3d=eps,
            dtype=torch.float64, seed=42
        )
        all_pass = all_pass and ok

    return all_pass


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        sys.exit(1)

    print("=" * 70)
    print("  WorldToVolumeMapping Module Test Suite")
    print("=" * 70)

    results = []

    # Test 1: Forward consistency
    results.append(("Forward f64", test_forward_consistency(
        N=100, dtype=torch.float64)))
    results.append(("Forward f32", test_forward_consistency(
        N=100, dtype=torch.float32)))

    # Test 2: Backward consistency
    results.append(("Backward f64", test_backward_consistency(
        N=100, dtype=torch.float64)))
    results.append(("Backward f32", test_backward_consistency(
        N=100, dtype=torch.float32)))

    # Test 3: Backward vs numerical
    results.append(("Backward vs FD", test_backward_vs_numerical()))

    # Test 4: Hessian symmetry
    results.append(("Hessian Symmetry", test_bwd2_hessian_symmetry(N=100)))

    # Test 5: Autograd Op vs Ground Truth
    results.append(("Autograd Op vs GT", test_bwd2_autograd_op_vs_ground_truth(N=100)))

    # Test 6: CUDA vs Ground Truth
    results.append(("CUDA vs GT", test_bwd2_cuda_vs_autograd_ground_truth(N=100)))

    # Test 7: CUDA vs Autograd Op
    results.append(("CUDA vs Autograd Op", test_bwd2_cuda_vs_autograd_op(N=100)))

    # Test 8: Edge cases
    results.append(("Edge Cases", test_edge_cases()))

    # Test 9: Factory function
    results.append(("Factory Function", test_factory_function()))

    # Test 10: Multiple configurations
    results.append(("Multiple Configs", test_bwd2_multiple_configs()))

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for name, ok in results:
        print(f"  {name:30s}: {'PASS' if ok else 'FAIL'}")

    all_pass = all(ok for _, ok in results)
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"{'='*70}")

    sys.exit(0 if all_pass else 1)
