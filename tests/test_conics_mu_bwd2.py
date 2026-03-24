"""
Test script for ConicsMu second-order backward pass (bwd2).

Validates the CUDA bwd2 implementation against the PyTorch autograd
double-backward reference by constructing a random quadratic loss function:

    L = 0.5 * phi^T W phi + phi^T (A psi)

where:
    phi = (u11, u12, u22, mu) are the 4 ConicsMu outputs
    psi = (means2d_x, means2d_y, opacity) are 3 pass-through variables
    W is a random 4x4 symmetric matrix (ensures H_{phi,phi} = W != 0)
    A is a random 4x3 matrix (ensures phi-psi cross coupling)

This loss ensures:
    - H_{phi,phi} = W (nonzero, tests term1: J^T W J)
    - v_phi = W phi + A psi (nonzero, tests term2: sum_k v_k Hess_k)
    - H_{phi,means2d} = A[:, :2] (tests cross-Hessian propagation)
    - H_{phi,opacity} = A[:, 2] (tests cross-Hessian propagation)

Usage:
    python tests/test_conics_mu_bwd2.py

Author: Dy164
Date: 2026-02-11
"""

import torch
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.data import Params, Data, Config
from gsplat2ct.ops.conics_mu import ConicsMuCUDA, ConicsMuAutograd


class ParamWrapper:
    """Simple wrapper to satisfy ParameterLike protocol."""
    def __init__(self, data: torch.Tensor):
        self.data = data


def generate_random_spd_cov2d_3x3(
    C: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """Generate random SPD 3x3 covariance matrices.
    
    Uses Cholesky: Sigma = L @ L^T to guarantee SPD.
    Returns upper-triangular storage: (a, b, c, d, e, f).
    """
    torch.manual_seed(seed)
    
    l00 = torch.rand(C, N, device=device, dtype=dtype) * 3.0 + 1.0
    l10 = torch.randn(C, N, device=device, dtype=dtype) * 1.0
    l11 = torch.rand(C, N, device=device, dtype=dtype) * 3.0 + 1.0
    l20 = torch.randn(C, N, device=device, dtype=dtype) * 1.0
    l21 = torch.randn(C, N, device=device, dtype=dtype) * 1.0
    l22 = torch.rand(C, N, device=device, dtype=dtype) * 3.0 + 1.0
    
    a = l00 * l00
    b = l00 * l10
    c = l00 * l20
    d = l10 * l10 + l11 * l11
    e = l10 * l20 + l11 * l21
    f = l20 * l20 + l21 * l21 + l22 * l22
    
    return torch.stack([a, b, c, d, e, f], dim=-1)


def generate_random_symmetric_matrix(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a random symmetric matrix of size (dim, dim)."""
    M = torch.randn(dim, dim, device=device, dtype=dtype)
    return 0.5 * (M + M.T)


def compute_quadratic_loss_components(
    conics: torch.Tensor,       # (C, N, 3)
    mus: torch.Tensor,          # (C, N)
    means2d: torch.Tensor,      # (C, N, 2)
    opacity: torch.Tensor,      # (C, N)
    W: torch.Tensor,            # (4, 4) symmetric
    A: torch.Tensor,            # (4, 3)
) -> torch.Tensor:
    """Compute the quadratic loss: L = 0.5 * phi^T W phi + phi^T (A psi).
    
    Computes per-element (C, N) scalar loss.
    
    phi = (u11, u12, u22, mu)  shape (C, N, 4)
    psi = (means2d_x, means2d_y, opacity)  shape (C, N, 3)
    """
    # Stack phi: (C, N, 4)
    phi = torch.cat([conics, mus.unsqueeze(-1)], dim=-1)
    
    # Stack psi: (C, N, 3)
    psi = torch.cat([means2d, opacity.unsqueeze(-1)], dim=-1)
    
    # L = 0.5 * phi^T W phi + phi^T A psi
    # Per-element: L[c,n] = 0.5 * phi[c,n] @ W @ phi[c,n] + phi[c,n] @ A @ psi[c,n]
    Wphi = torch.einsum('ij,...j->...i', W, phi)          # (C, N, 4)
    term1 = 0.5 * (phi * Wphi).sum(dim=-1)                # (C, N)
    
    Apsi = torch.einsum('ij,...j->...i', A, psi)           # (C, N, 4)
    term2 = (phi * Apsi).sum(dim=-1)                       # (C, N)
    
    return term1 + term2


def extract_downstream_from_quadratic(
    conics: torch.Tensor,       # (C, N, 3) detached
    mus: torch.Tensor,          # (C, N) detached
    means2d: torch.Tensor,      # (C, N, 2) detached
    opacity: torch.Tensor,      # (C, N) detached
    W: torch.Tensor,            # (4, 4) symmetric
    A: torch.Tensor,            # (4, 3)
):
    """Analytically compute downstream gradients and Hessians for the quadratic loss.
    
    dL/dphi = W phi + A psi      (first-order gradient, shape 4)
    d^2L/dphi^2 = W              (phi-phi Hessian, shape 4x4)
    d^2L/dphi dpsi = A           (phi-psi cross-Hessian, shape 4x3)
    
    Returns all downstream quantities needed by bwd2 kernel.
    """
    C, N = mus.shape
    device = mus.device
    dtype = mus.dtype
    
    # phi and psi
    phi = torch.cat([conics, mus.unsqueeze(-1)], dim=-1)  # (C, N, 4)
    psi = torch.cat([means2d, opacity.unsqueeze(-1)], dim=-1)  # (C, N, 3)
    
    # v_phi = dL/dphi = W @ phi + A @ psi
    v_phi = torch.einsum('ij,...j->...i', W, phi) + torch.einsum('ij,...j->...i', A, psi)
    # (C, N, 4)
    
    v_conics = v_phi[..., :3].contiguous()    # (C, N, 3)
    v_mus = v_phi[..., 3].contiguous()          # (C, N)
    
    # H_{phi,phi} = W (constant, broadcast over C, N)
    # Upper-triangular of 4x4 symmetric → 10 elements for full,
    # but our kernel expects:
    #   h_conics: upper-tri of W[:3,:3] → 6 elements (00,01,02,11,12,22)
    #   h_conics_mu: W[:3, 3] → 3 elements
    #   h_mu_mu: W[3, 3] → 1 element
    h_conics = torch.zeros(C, N, 6, device=device, dtype=dtype)
    h_conics[..., 0] = W[0, 0]  # (0,0)
    h_conics[..., 1] = W[0, 1]  # (0,1)
    h_conics[..., 2] = W[0, 2]  # (0,2)
    h_conics[..., 3] = W[1, 1]  # (1,1)
    h_conics[..., 4] = W[1, 2]  # (1,2)
    h_conics[..., 5] = W[2, 2]  # (2,2)
    
    h_conics_mu = torch.zeros(C, N, 3, device=device, dtype=dtype)
    h_conics_mu[..., 0] = W[0, 3]
    h_conics_mu[..., 1] = W[1, 3]
    h_conics_mu[..., 2] = W[2, 3]
    
    h_mu_mu = torch.full((C, N), W[3, 3].item(), device=device, dtype=dtype)
    
    # H_{phi,psi} = A (constant, broadcast)
    # A is (4, 3), split into:
    #   h_means2d_conics: A[:3, :2]^T → 2x3 row-major → (C, N, 6)
    #     Layout: [H_{mx,u11}, H_{mx,u12}, H_{mx,u22}, H_{my,u11}, H_{my,u12}, H_{my,u22}]
    #   h_conics_opacity: A[:3, 2]  → (C, N, 3)
    #   h_mu_means2d: A[3, :2]     → (C, N, 2)
    #   h_mu_opacity: A[3, 2]      → (C, N)
    h_means2d_conics = torch.zeros(C, N, 6, device=device, dtype=dtype)
    for i in range(2):  # mean dim (mx, my)
        for j in range(3):  # conic dim (u11, u12, u22)
            h_means2d_conics[..., i * 3 + j] = A[j, i]  # H_{mean_i, conic_j} = A[j, i]
    
    h_conics_opacity = torch.zeros(C, N, 3, device=device, dtype=dtype)
    for i in range(3):
        h_conics_opacity[..., i] = A[i, 2]
    
    h_mu_means2d = torch.zeros(C, N, 2, device=device, dtype=dtype)
    h_mu_means2d[..., 0] = A[3, 0]
    h_mu_means2d[..., 1] = A[3, 1]
    
    h_mu_opacity = torch.full((C, N), A[3, 2].item(), device=device, dtype=dtype)
    
    return {
        'v_conics': v_conics,
        'v_mus': v_mus,
        'h_conics': h_conics,
        'h_conics_mu': h_conics_mu,
        'h_mu_mu': h_mu_mu,
        'h_means2d_conics': h_means2d_conics,
        'h_conics_opacity': h_conics_opacity,
        'h_mu_means2d': h_mu_means2d,
        'h_mu_opacity': h_mu_opacity,
    }


def compute_ground_truth_autograd(
    cov2d_3x3: torch.Tensor,   # (C, N, 6)
    means2d: torch.Tensor,      # (C, N, 2)
    opacity: torch.Tensor,      # (C, N)
    eps: float,
    W: torch.Tensor,            # (4, 4)
    A: torch.Tensor,            # (4, 3)
):
    """Compute ground-truth Hessians using torch.autograd double-backward.
    
    Constructs the full loss L(sigma, means2d, opacity) = 
        quadratic(conics_mu(sigma), means2d, opacity)
    and differentiates twice to get the Hessian blocks.
    
    Returns:
        v_cov: (C, N, 6) first-order gradient
        h_cov_cov: (C, N, 21) upper-tri Hessian
        h_cov_means2d: (C, N, 12) cross-Hessian 6x2 row-major
        h_cov_opacity: (C, N, 6) cross-Hessian
    """
    from gsplat2ct.ops.conics_mu import _compute_conics_mu
    
    C, N = cov2d_3x3.shape[:2]
    device = cov2d_3x3.device
    dtype = cov2d_3x3.dtype
    
    # Create leaf tensors with requires_grad
    sigma = cov2d_3x3.detach().clone().requires_grad_(True)
    m2d = means2d.detach().clone().requires_grad_(True)
    opa = opacity.detach().clone().requires_grad_(True)
    
    # Forward
    conics, mus = _compute_conics_mu(sigma, eps)
    
    # Loss
    L = compute_quadratic_loss_components(conics, mus, m2d, opa, W, A)
    L_total = L.sum()
    
    # First-order gradients
    v_sigma, v_m2d, v_opa = torch.autograd.grad(
        L_total, [sigma, m2d, opa], create_graph=True
    )
    
    # --- H_{sigma,sigma} via second backward ---
    h_cov_cov = torch.zeros(C, N, 21, device=device, dtype=dtype)
    for i in range(6):
        selector = torch.zeros_like(v_sigma)
        selector[..., i] = 1.0
        
        grads = torch.autograd.grad(
            v_sigma, sigma, grad_outputs=selector,
            retain_graph=True
        )
        hess_row_i = grads[0]  # (C, N, 6)
        
        for j in range(i, 6):
            flat_idx = i * (11 - i) // 2 + j
            h_cov_cov[..., flat_idx] = hess_row_i[..., j]
    
    # --- H_{sigma,means2d} via second backward ---
    h_cov_means2d = torch.zeros(C, N, 12, device=device, dtype=dtype)
    for i in range(6):
        selector = torch.zeros_like(v_sigma)
        selector[..., i] = 1.0
        
        grads = torch.autograd.grad(
            v_sigma, m2d, grad_outputs=selector,
            retain_graph=True
        )
        hess_row_i = grads[0]  # (C, N, 2)
        
        for j in range(2):
            h_cov_means2d[..., i * 2 + j] = hess_row_i[..., j]
    
    # --- H_{sigma,opacity} via second backward ---
    h_cov_opacity = torch.zeros(C, N, 6, device=device, dtype=dtype)
    for i in range(6):
        selector = torch.zeros_like(v_sigma)
        selector[..., i] = 1.0
        
        grads = torch.autograd.grad(
            v_sigma, opa, grad_outputs=selector,
            retain_graph=True
        )
        h_cov_opacity[..., i] = grads[0]  # (C, N)
    
    return {
        'v_cov': v_sigma.detach(),
        'h_cov_cov': h_cov_cov.detach(),
        'h_cov_means2d': h_cov_means2d.detach(),
        'h_cov_opacity': h_cov_opacity.detach(),
    }


def test_bwd2_cuda_vs_autograd_ground_truth(
    C: int = 2,
    N: int = 16,
    eps: float = 0.3,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test CUDA bwd2 against autograd ground truth.
    
    Uses a random quadratic loss function to provide nonzero downstream
    Hessians, then compares all output blocks.
    """
    print(f"\n{'='*70}")
    print(f"Test bwd2 CUDA vs Autograd Ground Truth")
    print(f"  C={C}, N={N}, eps={eps}, dtype={dtype}, seed={seed}")
    print(f"{'='*70}")
    
    device = torch.device("cuda")
    
    # Generate inputs
    cov2d_3x3 = generate_random_spd_cov2d_3x3(C, N, device, dtype, seed=seed)
    
    torch.manual_seed(seed + 100)
    means2d = torch.randn(C, N, 2, device=device, dtype=dtype)
    opacity = torch.rand(C, N, device=device, dtype=dtype) * 0.8 + 0.1
    
    # Generate random quadratic coefficients
    torch.manual_seed(seed + 200)
    W = generate_random_symmetric_matrix(4, device, dtype)
    A = torch.randn(4, 3, device=device, dtype=dtype)
    
    # --- Ground truth via full autograd double-backward ---
    gt = compute_ground_truth_autograd(cov2d_3x3, means2d, opacity, eps, W, A)
    
    # --- CUDA bwd2 ---
    # First compute forward to get conics, mus
    from gsplat2ct.cuda._wrapper import conics_mu_fwd
    conics, mus = conics_mu_fwd(cov2d_3x3, eps)
    
    # Compute downstream quantities analytically from the quadratic loss
    downstream = extract_downstream_from_quadratic(
        conics, mus, means2d, opacity, W, A
    )
    
    # Build params and data for ConicsMuCUDA
    params = Params()
    params.add_param('cov2d_3x3', ParamWrapper(cov2d_3x3))
    config = Config()
    config.set('eps', eps)
    data = Data()
    for key, val in downstream.items():
        data.set(key, val)
    
    cuda_op = ConicsMuCUDA()
    cuda_op.initialize(params, config)
    fwd_result, grad_result, hess_result = cuda_op.forward_backward2(
        params, data, config
    )
    
    v_cov_cuda = grad_result.param_grads['cov2d_3x3']
    h_cov_cov_cuda = hess_result.param_hess['cov_cov']
    h_cov_means2d_cuda = hess_result.param_hess['cov_means2d']
    h_cov_opacity_cuda = hess_result.param_hess['cov_opacity']
    
    # --- Compare ---
    all_pass = True
    
    # Tolerances
    atol = 1e-8 if dtype == torch.float64 else 1e-4
    rtol = 1e-5 if dtype == torch.float64 else 1e-3
    
    def compare(name, cuda_val, gt_val, atol, rtol):
        nonlocal all_pass
        diff = (cuda_val - gt_val).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        gt_abs = gt_val.abs()
        gt_max = gt_abs.max().item()
        
        # Check with both absolute and relative tolerance
        rel_diff = diff / (gt_abs + 1e-15)
        max_rel = rel_diff.max().item()
        
        ok = max_diff < atol or max_rel < rtol
        status = "PASS" if ok else "FAIL"
        
        print(f"  {name:25s}: max_abs={max_diff:.2e}, mean_abs={mean_diff:.2e}, "
              f"max_rel={max_rel:.2e}, gt_max={gt_max:.2e} [{status}]")
        
        if not ok:
            all_pass = False
            # Print worst elements for debugging
            worst_idx = diff.argmax()
            flat_cuda = cuda_val.reshape(-1)
            flat_gt = gt_val.reshape(-1)
            print(f"    worst element: cuda={flat_cuda[worst_idx].item():.10e}, "
                  f"gt={flat_gt[worst_idx].item():.10e}")
        
        return ok
    
    compare("v_cov2d_3x3 (grad)", v_cov_cuda, gt['v_cov'], atol, rtol)
    compare("h_cov_cov (21 Hessian)", h_cov_cov_cuda, gt['h_cov_cov'], atol, rtol)
    compare("h_cov_means2d (12 cross)", h_cov_means2d_cuda, gt['h_cov_means2d'], atol, rtol)
    compare("h_cov_opacity (6 cross)", h_cov_opacity_cuda, gt['h_cov_opacity'], atol, rtol)
    
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def test_bwd2_autograd_op_vs_ground_truth(
    C: int = 2,
    N: int = 16,
    eps: float = 0.3,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test ConicsMuAutograd.forward_backward2 against full autograd ground truth.
    
    This validates the autograd ops integration independently of CUDA.
    """
    print(f"\n{'='*70}")
    print(f"Test bwd2 Autograd Op vs Ground Truth")
    print(f"  C={C}, N={N}, eps={eps}, dtype={dtype}, seed={seed}")
    print(f"{'='*70}")
    
    device = torch.device("cuda")
    
    # Generate inputs
    cov2d_3x3 = generate_random_spd_cov2d_3x3(C, N, device, dtype, seed=seed)
    
    torch.manual_seed(seed + 100)
    means2d = torch.randn(C, N, 2, device=device, dtype=dtype)
    opacity = torch.rand(C, N, device=device, dtype=dtype) * 0.8 + 0.1
    
    torch.manual_seed(seed + 200)
    W = generate_random_symmetric_matrix(4, device, dtype)
    A = torch.randn(4, 3, device=device, dtype=dtype)
    
    # Ground truth
    gt = compute_ground_truth_autograd(cov2d_3x3, means2d, opacity, eps, W, A)
    
    # ConicsMuAutograd.forward_backward2
    from gsplat2ct.ops.conics_mu import _compute_conics_mu
    with torch.no_grad():
        conics, mus = _compute_conics_mu(cov2d_3x3, eps)
    
    downstream = extract_downstream_from_quadratic(
        conics, mus, means2d, opacity, W, A
    )
    
    params = Params()
    params.add_param('cov2d_3x3', ParamWrapper(cov2d_3x3))
    config = Config()
    config.set('eps', eps)
    data = Data()
    for key, val in downstream.items():
        data.set(key, val)
    
    ag_op = ConicsMuAutograd()
    ag_op.initialize(params, config)
    fwd_result, grad_result, hess_result = ag_op.forward_backward2(
        params, data, config
    )
    
    v_cov_ag = grad_result.param_grads['cov2d_3x3']
    h_cov_cov_ag = hess_result.param_hess['cov_cov']
    h_cov_means2d_ag = hess_result.param_hess['cov_means2d']
    h_cov_opacity_ag = hess_result.param_hess['cov_opacity']
    
    all_pass = True
    atol = 1e-8 if dtype == torch.float64 else 1e-4
    rtol = 1e-5 if dtype == torch.float64 else 1e-3
    
    def compare(name, val, gt_val, atol, rtol):
        nonlocal all_pass
        diff = (val - gt_val).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        gt_max = gt_val.abs().max().item()
        rel_diff = diff / (gt_val.abs() + 1e-15)
        max_rel = rel_diff.max().item()
        
        ok = max_diff < atol or max_rel < rtol
        status = "PASS" if ok else "FAIL"
        
        print(f"  {name:25s}: max_abs={max_diff:.2e}, mean_abs={mean_diff:.2e}, "
              f"max_rel={max_rel:.2e}, gt_max={gt_max:.2e} [{status}]")
        if not ok:
            all_pass = False
        return ok
    
    compare("v_cov2d_3x3 (grad)", v_cov_ag, gt['v_cov'], atol, rtol)
    compare("h_cov_cov (21 Hessian)", h_cov_cov_ag, gt['h_cov_cov'], atol, rtol)
    compare("h_cov_means2d (12 cross)", h_cov_means2d_ag, gt['h_cov_means2d'], atol, rtol)
    compare("h_cov_opacity (6 cross)", h_cov_opacity_ag, gt['h_cov_opacity'], atol, rtol)
    
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def test_bwd2_cuda_vs_autograd_op(
    C: int = 2,
    N: int = 16,
    eps: float = 0.3,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Direct comparison: ConicsMuCUDA.forward_backward2 vs ConicsMuAutograd.forward_backward2.
    
    Uses same downstream inputs, compares all output blocks.
    """
    print(f"\n{'='*70}")
    print(f"Test bwd2 CUDA Op vs Autograd Op (direct)")
    print(f"  C={C}, N={N}, eps={eps}, dtype={dtype}, seed={seed}")
    print(f"{'='*70}")
    
    device = torch.device("cuda")
    
    cov2d_3x3 = generate_random_spd_cov2d_3x3(C, N, device, dtype, seed=seed)
    
    torch.manual_seed(seed + 100)
    means2d = torch.randn(C, N, 2, device=device, dtype=dtype)
    opacity = torch.rand(C, N, device=device, dtype=dtype) * 0.8 + 0.1
    
    torch.manual_seed(seed + 200)
    W = generate_random_symmetric_matrix(4, device, dtype)
    A = torch.randn(4, 3, device=device, dtype=dtype)
    
    # Compute forward outputs (shared)
    from gsplat2ct.ops.conics_mu import _compute_conics_mu
    with torch.no_grad():
        conics, mus = _compute_conics_mu(cov2d_3x3, eps)
    
    downstream = extract_downstream_from_quadratic(
        conics, mus, means2d, opacity, W, A
    )
    
    config = Config()
    config.set('eps', eps)
    
    # --- CUDA ---
    params_c = Params()
    params_c.add_param('cov2d_3x3', ParamWrapper(cov2d_3x3.clone()))
    data_c = Data()
    for key, val in downstream.items():
        data_c.set(key, val.clone())
    
    cuda_op = ConicsMuCUDA()
    cuda_op.initialize(params_c, config)
    _, grad_c, hess_c = cuda_op.forward_backward2(params_c, data_c, config)
    
    # --- Autograd ---
    params_a = Params()
    params_a.add_param('cov2d_3x3', ParamWrapper(cov2d_3x3.clone()))
    data_a = Data()
    for key, val in downstream.items():
        data_a.set(key, val.clone())
    
    ag_op = ConicsMuAutograd()
    ag_op.initialize(params_a, config)
    _, grad_a, hess_a = ag_op.forward_backward2(params_a, data_a, config)
    
    all_pass = True
    atol = 1e-8 if dtype == torch.float64 else 1e-4
    rtol = 1e-5 if dtype == torch.float64 else 1e-3
    
    def compare(name, cuda_val, ag_val, atol, rtol):
        nonlocal all_pass
        diff = (cuda_val - ag_val).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        ref_max = ag_val.abs().max().item()
        rel_diff = diff / (ag_val.abs() + 1e-15)
        max_rel = rel_diff.max().item()
        
        ok = max_diff < atol or max_rel < rtol
        status = "PASS" if ok else "FAIL"
        
        print(f"  {name:25s}: max_abs={max_diff:.2e}, mean_abs={mean_diff:.2e}, "
              f"max_rel={max_rel:.2e}, ref_max={ref_max:.2e} [{status}]")
        if not ok:
            all_pass = False
        return ok
    
    compare("v_cov2d_3x3 (grad)",
            grad_c.param_grads['cov2d_3x3'], grad_a.param_grads['cov2d_3x3'],
            atol, rtol)
    compare("h_cov_cov (21)",
            hess_c.param_hess['cov_cov'], hess_a.param_hess['cov_cov'],
            atol, rtol)
    compare("h_cov_means2d (12)",
            hess_c.param_hess['cov_means2d'], hess_a.param_hess['cov_means2d'],
            atol, rtol)
    compare("h_cov_opacity (6)",
            hess_c.param_hess['cov_opacity'], hess_a.param_hess['cov_opacity'],
            atol, rtol)
    
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def test_bwd2_hessian_symmetry(
    C: int = 1,
    N: int = 8,
    eps: float = 0.3,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test that H_{sigma,sigma} is symmetric by checking upper-tri vs lower-tri.
    
    The Hessian of a twice-differentiable function must be symmetric.
    We verify this by computing element (i,j) and (j,i) independently
    via the autograd ground truth and checking they match.
    """
    print(f"\n{'='*70}")
    print(f"Test bwd2 Hessian Symmetry")
    print(f"  C={C}, N={N}, eps={eps}, dtype={dtype}")
    print(f"{'='*70}")
    
    device = torch.device("cuda")
    cov2d_3x3 = generate_random_spd_cov2d_3x3(C, N, device, dtype, seed=seed)
    
    torch.manual_seed(seed + 100)
    means2d = torch.randn(C, N, 2, device=device, dtype=dtype)
    opacity = torch.rand(C, N, device=device, dtype=dtype) * 0.8 + 0.1
    
    torch.manual_seed(seed + 200)
    W = generate_random_symmetric_matrix(4, device, dtype)
    A = torch.randn(4, 3, device=device, dtype=dtype)
    
    # Ground truth gives upper-triangular → reconstruct full and check symmetry
    gt = compute_ground_truth_autograd(cov2d_3x3, means2d, opacity, eps, W, A)
    h_cov_cov = gt['h_cov_cov']  # (C, N, 21)
    
    # Reconstruct full 6x6 matrix
    full_H = torch.zeros(C, N, 6, 6, device=device, dtype=dtype)
    idx = 0
    for i in range(6):
        for j in range(i, 6):
            full_H[..., i, j] = h_cov_cov[..., idx]
            full_H[..., j, i] = h_cov_cov[..., idx]
            idx += 1
    
    # Check symmetry
    sym_diff = (full_H - full_H.transpose(-1, -2)).abs().max().item()
    ok = sym_diff < 1e-10
    print(f"  Symmetry max diff: {sym_diff:.2e} [{'PASS' if ok else 'FAIL'}]")
    
    return ok


def test_bwd2_multiple_configs():
    """Run bwd2 tests with various configurations."""
    print(f"\n{'#'*70}")
    print(f"# Running bwd2 tests with multiple configurations")
    print(f"{'#'*70}")
    
    all_pass = True
    
    # Test with different sizes
    for C, N in [(1, 8), (2, 16), (1, 32)]:
        ok = test_bwd2_cuda_vs_autograd_ground_truth(
            C=C, N=N, eps=0.3, dtype=torch.float64, seed=42
        )
        all_pass = all_pass and ok
    
    # Test with different eps values
    for eps in [0.1, 0.3, 1.0]:
        ok = test_bwd2_cuda_vs_autograd_ground_truth(
            C=1, N=16, eps=eps, dtype=torch.float64, seed=42
        )
        all_pass = all_pass and ok
    
    return all_pass


if __name__ == "__main__":
    print("=" * 70)
    print("ConicsMu bwd2 (Second-Order Backward) Tests")
    print("=" * 70)
    
    results = []
    
    # Test 1: Hessian symmetry (quick sanity check)
    results.append(("Hessian Symmetry", test_bwd2_hessian_symmetry(C=3, N=100)))
    
    # Test 2: Autograd op vs ground truth (validates autograd path)
    results.append(("Autograd Op vs GT", test_bwd2_autograd_op_vs_ground_truth(C=3, N=100)))
    
    # Test 3: CUDA vs ground truth (the main validation)
    results.append(("CUDA vs GT", test_bwd2_cuda_vs_autograd_ground_truth(C=3, N=100)))
    
    # Test 4: CUDA vs Autograd op (direct comparison)
    results.append(("CUDA vs Autograd Op", test_bwd2_cuda_vs_autograd_op(C=3, N=100)))
    
    # Test 5: Multiple configurations
    results.append(("Multiple Configs", test_bwd2_multiple_configs()))
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    for name, ok in results:
        print(f"  {name:30s}: {'PASS' if ok else 'FAIL'}")
    
    all_pass = all(ok for _, ok in results)
    print(f"\n  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    
    sys.exit(0 if all_pass else 1)
