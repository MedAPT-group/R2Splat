"""
Test script for ParameterTransform module.

Validates the CUDA implementation against the PyTorch autograd
reference implementation by comparing:
1. Forward outputs (covars)
2. Backward outputs (v_quats, v_scale_ori)
3. Second-order backward outputs (Hessians and cross-Hessians)

The module transforms:
    quats [N, 4]     (unit quaternion, w,x,y,z)
    scale_ori [N, 3] (raw scale before sigmoid activation)
    ->  covars [N, 6] (upper-triangular covariance C00,C01,C02,C11,C12,C22)

Pipeline:
    sig_i = sigmoid(so_i)
    s_i = sig_i * (scale_max - scale_min) + scale_min
    R = rotation_matrix(quat)
    C = R * diag(s^2) * R^T

Usage:
    python tests/test_parameter_transform.py

Author: Dy164
Date: 2026-02-20
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.data import Params, Data, Config, ForwardResult, GradResult, HessResult
from gsplat2ct.ops.parameter_transform import (
    ParameterTransformCUDA,
    ParameterTransformAutograd,
    create_parameter_transform,
    _compute_covars,
)


class ParamWrapper:
    """Simple wrapper to satisfy ParameterLike protocol."""
    def __init__(self, data: torch.Tensor):
        self.data = data


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_random_unit_quaternions(
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """Generate N random unit quaternions."""
    torch.manual_seed(seed)
    q = torch.randn(N, 4, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)  # normalize
    return q


def generate_random_scale_ori(
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> torch.Tensor:
    """Generate N random scale_ori vectors (pre-sigmoid).

    Values in [-3, 3] so sigmoid is not saturated.
    """
    torch.manual_seed(seed + 1000)
    return torch.randn(N, 3, device=device, dtype=dtype) * 1.5


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
    N: int = 1000,
    scale_min: float = 0.01,
    scale_max: float = 2.0,
    dtype: torch.dtype = torch.float64,
):
    """Test that CUDA and autograd forward passes produce the same results."""
    print(f"\n{'='*60}")
    print(f"Test Forward Consistency: N={N}, dtype={dtype}")
    print(f"  scale_min={scale_min}, scale_max={scale_max}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    quats = generate_random_unit_quaternions(N, device, dtype)
    scale_ori = generate_random_scale_ori(N, device, dtype)

    config = Config()
    config.set('scale_min', scale_min)
    config.set('scale_max', scale_max)

    params = Params()
    params.add_param('quats', ParamWrapper(quats))
    params.add_param('scale_ori', ParamWrapper(scale_ori))
    data = Data()

    # CUDA forward
    cuda_op = ParameterTransformCUDA()
    cuda_op.initialize(params, config)
    cuda_result = cuda_op.forward_only(params, data, config)
    covars_cuda = cuda_result.outputs['covars']

    # Autograd forward
    ag_op = ParameterTransformAutograd()
    ag_op.initialize(params, config)
    ag_result = ag_op.forward_only(params, data, config)
    covars_ag = ag_result.outputs['covars']

    # Compare
    diff = (covars_cuda - covars_ag).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    atol = 1e-10 if dtype == torch.float64 else 1e-5

    print(f"  max abs diff:  {max_diff:.2e}")
    print(f"  mean abs diff: {mean_diff:.2e}")

    for i, name in enumerate(['C00', 'C01', 'C02', 'C11', 'C12', 'C22']):
        elem_diff = diff[:, i].max().item()
        print(f"    {name}: {elem_diff:.2e}")

    ok = max_diff < atol
    print(f"  Forward match: {'PASS' if ok else 'FAIL'} (atol={atol:.0e})")
    return ok


# =============================================================================
# Test 2: Backward Consistency (CUDA vs Autograd)
# =============================================================================

def test_backward_consistency(
    N: int = 1000,
    scale_min: float = 0.01,
    scale_max: float = 2.0,
    dtype: torch.dtype = torch.float64,
):
    """Test that CUDA and autograd backward passes produce the same gradients."""
    print(f"\n{'='*60}")
    print(f"Test Backward Consistency: N={N}, dtype={dtype}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    quats = generate_random_unit_quaternions(N, device, dtype)
    scale_ori = generate_random_scale_ori(N, device, dtype)

    # Generate random upstream gradients
    torch.manual_seed(123)
    v_covars = torch.randn(N, 6, device=device, dtype=dtype)

    config = Config()
    config.set('scale_min', scale_min)
    config.set('scale_max', scale_max)

    # --- CUDA backward ---
    params_c = Params()
    params_c.add_param('quats', ParamWrapper(quats.clone()))
    params_c.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    data_c = Data()
    data_c.set('v_covars', v_covars.clone())

    cuda_op = ParameterTransformCUDA()
    cuda_op.initialize(params_c, config)
    cuda_fwd, cuda_grad = cuda_op.forward_backward1(params_c, data_c, config)

    v_quats_cuda = cuda_grad.param_grads['quats']
    v_so_cuda = cuda_grad.param_grads['scale_ori']

    # --- Autograd backward ---
    params_a = Params()
    params_a.add_param('quats', ParamWrapper(quats.clone()))
    params_a.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    data_a = Data()
    data_a.set('v_covars', v_covars.clone())

    ag_op = ParameterTransformAutograd()
    ag_op.initialize(params_a, config)
    ag_fwd, ag_grad = ag_op.forward_backward1(params_a, data_a, config)

    v_quats_ag = ag_grad.param_grads['quats']
    v_so_ag = ag_grad.param_grads['scale_ori']

    # Compare forward outputs first
    covars_diff = (cuda_fwd.outputs['covars'] - ag_fwd.outputs['covars']).abs().max().item()
    print(f"  Forward covars max diff: {covars_diff:.2e}")

    # Compare gradients
    diff_q = (v_quats_cuda - v_quats_ag).abs()
    diff_s = (v_so_cuda - v_so_ag).abs()

    max_diff_q = diff_q.max().item()
    max_diff_s = diff_s.max().item()
    mean_diff_q = diff_q.mean().item()
    mean_diff_s = diff_s.mean().item()

    # Per-element max diff
    for i, name in enumerate(['w', 'x', 'y', 'z']):
        elem_diff = diff_q[:, i].max().item()
        print(f"  v_{name} max abs diff: {elem_diff:.2e}")

    for i, name in enumerate(['so0', 'so1', 'so2']):
        elem_diff = diff_s[:, i].max().item()
        print(f"  v_{name} max abs diff: {elem_diff:.2e}")

    atol = 1e-8 if dtype == torch.float64 else 1e-4

    print(f"  Overall v_quats max abs diff:     {max_diff_q:.2e}")
    print(f"  Overall v_quats mean abs diff:    {mean_diff_q:.2e}")
    print(f"  Overall v_scale_ori max abs diff: {max_diff_s:.2e}")
    print(f"  Overall v_scale_ori mean abs diff:{mean_diff_s:.2e}")

    q_ok = max_diff_q < atol
    s_ok = max_diff_s < atol
    print(f"  Backward quats match:     {'PASS' if q_ok else 'FAIL'} (atol={atol:.0e})")
    print(f"  Backward scale_ori match: {'PASS' if s_ok else 'FAIL'} (atol={atol:.0e})")

    return q_ok and s_ok


# =============================================================================
# Test 3: Backward vs Numerical (Finite Differences)
# =============================================================================

def test_backward_vs_numerical(
    N: int = 10,
    scale_min: float = 0.01,
    scale_max: float = 2.0,
    dtype: torch.dtype = torch.float64,
    fd_eps: float = 1e-6,
):
    """Validate backward using finite differences (gold standard)."""
    print(f"\n{'='*60}")
    print(f"Test Backward vs Numerical (Finite Diff): N={N}, fd_eps={fd_eps}")
    print(f"{'='*60}")

    device = torch.device("cuda")
    quats = generate_random_unit_quaternions(N, device, dtype, seed=99)
    scale_ori = generate_random_scale_ori(N, device, dtype, seed=99)

    from gsplat2ct.cuda._wrapper import parameter_transform_fwd, parameter_transform_bwd

    # Forward
    covars, _ = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)

    # Random upstream gradient
    torch.manual_seed(77)
    v_covars = torch.randn_like(covars)

    # CUDA backward
    v_quats_cuda, v_so_cuda = parameter_transform_bwd(
        quats, scale_ori, scale_min, scale_max, v_covars
    )

    # Finite-difference gradient for quats
    v_quats_fd = torch.zeros_like(quats)
    for n in range(N):
        for c in range(4):
            q_p = quats.clone(); q_p[n, c] += fd_eps
            q_m = quats.clone(); q_m[n, c] -= fd_eps

            c_p, _ = parameter_transform_fwd(q_p, scale_ori, scale_min, scale_max)
            c_m, _ = parameter_transform_fwd(q_m, scale_ori, scale_min, scale_max)

            dc = (c_p - c_m) / (2 * fd_eps)
            v_quats_fd[n, c] = (dc[n] * v_covars[n]).sum()

    # Finite-difference gradient for scale_ori
    v_so_fd = torch.zeros_like(scale_ori)
    for n in range(N):
        for c in range(3):
            so_p = scale_ori.clone(); so_p[n, c] += fd_eps
            so_m = scale_ori.clone(); so_m[n, c] -= fd_eps

            c_p, _ = parameter_transform_fwd(quats, so_p, scale_min, scale_max)
            c_m, _ = parameter_transform_fwd(quats, so_m, scale_min, scale_max)

            dc = (c_p - c_m) / (2 * fd_eps)
            v_so_fd[n, c] = (dc[n] * v_covars[n]).sum()

    # Compare
    atol = 1e-5

    diff_q = (v_quats_cuda - v_quats_fd).abs()
    diff_s = (v_so_cuda - v_so_fd).abs()

    max_diff_q = diff_q.max().item()
    max_diff_s = diff_s.max().item()

    print(f"  v_quats max abs diff vs FD:     {max_diff_q:.2e}")
    print(f"  v_scale_ori max abs diff vs FD: {max_diff_s:.2e}")

    for i, name in enumerate(['w', 'x', 'y', 'z']):
        print(f"    v_{name}: {diff_q[:, i].max().item():.2e}")
    for i, name in enumerate(['so0', 'so1', 'so2']):
        print(f"    v_{name}: {diff_s[:, i].max().item():.2e}")

    q_ok = max_diff_q < atol
    s_ok = max_diff_s < atol
    print(f"  Numerical quats match:     {'PASS' if q_ok else 'FAIL'} (atol={atol:.0e})")
    print(f"  Numerical scale_ori match: {'PASS' if s_ok else 'FAIL'} (atol={atol:.0e})")

    return q_ok and s_ok


# =============================================================================
# Test 4: Bwd2 — Downstream Quadratic Loss Construction
# =============================================================================

def compute_quadratic_downstream(
    covars: torch.Tensor,       # [N, 6] detached
    means: torch.Tensor,        # [N, 3] detached
    opacity: torch.Tensor,      # [N] detached
    W: torch.Tensor,            # [6, 6] symmetric
    A_mean: torch.Tensor,       # [6, 3]
    A_opa: torch.Tensor,        # [6, 1]
):
    """Analytically compute downstream quantities for bwd2 kernel.

    For the quadratic loss:
        L = 0.5 * c^T W c + c^T A_mean m + c^T A_opa o

    Downstream quantities:
        v_covars = W @ c + A_mean @ m + A_opa * o   [N, 6]
        h_covars = W upper-tri                       [N, 21]
        h_covars_means = A_mean                      [N, 18] (6x3 row-major)
        h_covars_opacity = A_opa                     [N, 6]
    """
    N = covars.shape[0]
    device = covars.device
    dtype = covars.dtype

    # v_covars = dL/dc = W @ c + A_mean @ m + A_opa * o
    v_covars = (
        (covars @ W)  # W symmetric => covars @ W = (W @ c^T)^T per row
        + (means @ A_mean.T)  # [N, 6]
        + (opacity.unsqueeze(-1) * A_opa.squeeze(-1).unsqueeze(0))
    )

    # h_covars: upper-tri of W [6x6] -> 21 elements
    h_covars = torch.zeros(N, 21, device=device, dtype=dtype)
    idx = 0
    for i in range(6):
        for j in range(i, 6):
            h_covars[:, idx] = W[i, j]
            idx += 1

    # h_covars_means: A_mean [6, 3] row-major, broadcast to [N, 18]
    h_covars_means = A_mean.reshape(1, 18).expand(N, -1).contiguous()

    # h_covars_opacity: A_opa [6, 1] -> [6], broadcast to [N, 6]
    h_covars_opacity = A_opa.squeeze(-1).unsqueeze(0).expand(N, -1).contiguous()

    return {
        'v_covars': v_covars,
        'h_covars': h_covars,
        'h_covars_means': h_covars_means,
        'h_covars_opacity': h_covars_opacity,
    }


def compute_ground_truth_autograd(
    quats: torch.Tensor,        # [N, 4] detached
    scale_ori: torch.Tensor,    # [N, 3] detached
    means: torch.Tensor,        # [N, 3] detached
    opacity: torch.Tensor,      # [N] detached
    scale_min: float,
    scale_max: float,
    W: torch.Tensor,            # [6, 6] symmetric
    A_mean: torch.Tensor,       # [6, 3]
    A_opa: torch.Tensor,        # [6, 1]
):
    """Compute ground-truth Hessians using torch.autograd double-backward.

    Constructs a quadratic loss in (covars, means, opacity):
        L = 0.5 * c^T W c + c^T A_mean m + c^T A_opa o

    Then chain-rules through covars = f(quats, scale_ori) to get all
    Hessian blocks.
    """
    N = quats.shape[0]
    device = quats.device
    dtype = quats.dtype

    # Create leaf tensors
    q = quats.detach().clone().requires_grad_(True)
    so = scale_ori.detach().clone().requires_grad_(True)
    m = means.detach().clone().requires_grad_(True)
    o = opacity.detach().clone().requires_grad_(True)

    # Forward
    covars = _compute_covars(q, so, scale_min, scale_max)  # [N, 6]

    # Quadratic loss (summed over all Gaussians)
    Wc = covars @ W.T  # [N, 6]
    term1 = 0.5 * (covars * Wc).sum()
    Am = covars @ A_mean  # [N, 3]
    term2 = (Am * m).sum()
    Ao = (covars @ A_opa.squeeze(-1))  # [N]
    term3 = (Ao * o).sum()
    L_total = term1 + term2 + term3

    # First-order gradients
    v_q, v_so, v_m, v_o = torch.autograd.grad(
        L_total, [q, so, m, o], create_graph=True
    )

    # Helper function for second-order
    def hess_block(v_source, source_var, target_var, source_dim, target_dim):
        """Compute d(v_source)/d(target_var) block."""
        H = torch.zeros(N, source_dim, target_dim, device=device, dtype=dtype)
        for a in range(source_dim):
            selector = torch.zeros_like(v_source)
            if v_source.dim() == 2:
                selector[:, a] = 1.0
            else:
                selector[:] = 1.0  # scalar per-element
            grads = torch.autograd.grad(
                v_source, target_var, grad_outputs=selector, retain_graph=True
            )[0]
            if grads.dim() == 2:
                H[:, a, :] = grads
            else:
                H[:, a, 0] = grads
        return H

    # H_{q,q} [N, 4, 4]
    h_qq = hess_block(v_q, q, q, 4, 4)
    # H_{so,so} [N, 3, 3]
    h_ss = hess_block(v_so, so, so, 3, 3)
    # H_{q,so} [N, 4, 3]
    h_qs = hess_block(v_q, q, so, 4, 3)
    # H_{q,mean} [N, 4, 3]
    h_qm = hess_block(v_q, q, m, 4, 3)
    # H_{q,opacity} [N, 4, 1] -> [N, 4]
    h_qo = hess_block(v_q, q, o, 4, 1).squeeze(-1)
    # H_{so,mean} [N, 3, 3]
    h_sm = hess_block(v_so, so, m, 3, 3)
    # H_{so,opacity} [N, 3, 1] -> [N, 3]
    h_soo = hess_block(v_so, so, o, 3, 1).squeeze(-1)

    return {
        'v_q': v_q.detach(),
        'v_so': v_so.detach(),
        'h_qq': h_qq.detach(),
        'h_ss': h_ss.detach(),
        'h_qs': h_qs.detach(),
        'h_qm': h_qm.detach(),
        'h_qo': h_qo.detach(),
        'h_sm': h_sm.detach(),
        'h_so': h_soo.detach(),
    }


# =============================================================================
# Test 5: Bwd2 CUDA vs Autograd Ground Truth
# =============================================================================

def test_bwd2_cuda_vs_autograd_ground_truth(
    N: int = 16,
    scale_min: float = 0.01,
    scale_max: float = 2.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test CUDA bwd2 against autograd ground truth.

    Uses a random quadratic loss function to provide nonzero downstream
    Hessians, then compares all output blocks.
    """
    print(f"\n{'='*70}")
    print(f"Test bwd2 CUDA vs Autograd Ground Truth")
    print(f"  N={N}, scale=({scale_min}, {scale_max}), dtype={dtype}, seed={seed}")
    print(f"{'='*70}")

    device = torch.device("cuda")

    # Generate inputs
    quats = generate_random_unit_quaternions(N, device, dtype, seed=seed)
    scale_ori = generate_random_scale_ori(N, device, dtype, seed=seed)

    torch.manual_seed(seed + 100)
    means = torch.randn(N, 3, device=device, dtype=dtype)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1

    # Generate random quadratic coefficients
    torch.manual_seed(seed + 200)
    W = generate_random_symmetric_matrix(6, device, dtype)
    A_mean = torch.randn(6, 3, device=device, dtype=dtype)
    A_opa = torch.randn(6, 1, device=device, dtype=dtype)

    # --- Ground truth via autograd double-backward ---
    gt = compute_ground_truth_autograd(
        quats, scale_ori, means, opacity, scale_min, scale_max,
        W, A_mean, A_opa,
    )

    # --- CUDA bwd2 via ops interface ---
    from gsplat2ct.cuda._wrapper import parameter_transform_fwd
    covars, _ = parameter_transform_fwd(quats, scale_ori, scale_min, scale_max)

    downstream = compute_quadratic_downstream(
        covars, means, opacity, W, A_mean, A_opa
    )

    params = Params()
    params.add_param('quats', ParamWrapper(quats))
    params.add_param('scale_ori', ParamWrapper(scale_ori))
    config = Config()
    config.set('scale_min', scale_min)
    config.set('scale_max', scale_max)
    data_obj = Data()
    for key, val in downstream.items():
        data_obj.set(key, val)

    cuda_op = ParameterTransformCUDA()
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
        gt_abs = gt_val.abs()
        gt_max = gt_abs.max().item()

        rel_diff = diff / (gt_abs + 1e-15)
        max_rel = rel_diff.max().item()

        ok = max_diff < atol or max_rel < rtol
        status = "PASS" if ok else "FAIL"

        print(f"  {name:30s}: max_abs={max_diff:.2e}, mean_abs={mean_diff:.2e}, "
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
    compare("v_quats (grad)", grad_c.param_grads['quats'], gt['v_q'])
    compare("v_scale_ori (grad)", grad_c.param_grads['scale_ori'], gt['v_so'])

    # Theta-theta Hessians
    # h_quats: upper-tri of 4x4 -> 10 elements
    h_qq_utri = torch.zeros(N, 10, device=device, dtype=dtype)
    idx = 0
    for i in range(4):
        for j in range(i, 4):
            h_qq_utri[:, idx] = gt['h_qq'][:, i, j]
            idx += 1
    compare("h_quats (upper-tri 4x4)", hess_c.param_hess['quats'], h_qq_utri)

    # h_scale_ori: upper-tri of 3x3 -> 6 elements
    h_ss_utri = torch.stack([
        gt['h_ss'][:, 0, 0],
        gt['h_ss'][:, 0, 1],
        gt['h_ss'][:, 0, 2],
        gt['h_ss'][:, 1, 1],
        gt['h_ss'][:, 1, 2],
        gt['h_ss'][:, 2, 2],
    ], dim=-1)
    compare("h_scale_ori (upper-tri 3x3)", hess_c.param_hess['scale_ori'], h_ss_utri)

    # Check that off-diagonal of h_ss is small (should be ~0 by design)
    h_ss_offdiag_max = 0.0
    for i in range(3):
        for j in range(3):
            if i != j:
                h_ss_offdiag_max = max(h_ss_offdiag_max,
                                       gt['h_ss'][:, i, j].abs().max().item())
    print(f"  {'h_ss off-diagonal (GT)':30s}: max_abs={h_ss_offdiag_max:.2e} (GN term, non-zero expected)")

    # h_quats_scale_ori: 4x3 row-major -> 12 elements
    h_qs_flat = gt['h_qs'].reshape(N, 12)
    compare("h_quats_scale_ori (4x3)", hess_c.cross_hess['quats_scale_ori'], h_qs_flat)

    # Cross-Hessians with external parameters
    h_qm_flat = gt['h_qm'].reshape(N, 12)
    compare("h_quats_means (4x3)", hess_c.cross_hess['quats_means'], h_qm_flat)

    compare("h_quats_opacity (4)", hess_c.cross_hess['quats_opacity'], gt['h_qo'])

    h_sm_flat = gt['h_sm'].reshape(N, 9)
    compare("h_scale_ori_means (3x3)", hess_c.cross_hess['scale_ori_means'], h_sm_flat)

    compare("h_scale_ori_opacity (3)", hess_c.cross_hess['scale_ori_opacity'], gt['h_so'])

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# =============================================================================
# Test 6: Bwd2 Autograd Op vs Ground Truth
# =============================================================================

def test_bwd2_autograd_op_vs_ground_truth(
    N: int = 16,
    scale_min: float = 0.01,
    scale_max: float = 2.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test ParameterTransformAutograd.forward_backward2 against full autograd ground truth.

    This validates the autograd ops integration independently of CUDA.
    """
    print(f"\n{'='*70}")
    print(f"Test bwd2 Autograd Op vs Ground Truth")
    print(f"  N={N}, scale=({scale_min}, {scale_max}), dtype={dtype}, seed={seed}")
    print(f"{'='*70}")

    device = torch.device("cuda")

    quats = generate_random_unit_quaternions(N, device, dtype, seed=seed)
    scale_ori = generate_random_scale_ori(N, device, dtype, seed=seed)

    torch.manual_seed(seed + 100)
    means = torch.randn(N, 3, device=device, dtype=dtype)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1

    torch.manual_seed(seed + 200)
    W = generate_random_symmetric_matrix(6, device, dtype)
    A_mean = torch.randn(6, 3, device=device, dtype=dtype)
    A_opa = torch.randn(6, 1, device=device, dtype=dtype)

    # Ground truth
    gt = compute_ground_truth_autograd(
        quats, scale_ori, means, opacity, scale_min, scale_max,
        W, A_mean, A_opa,
    )

    # Autograd Op
    with torch.no_grad():
        covars = _compute_covars(quats, scale_ori, scale_min, scale_max)

    downstream = compute_quadratic_downstream(
        covars, means, opacity, W, A_mean, A_opa
    )

    params = Params()
    params.add_param('quats', ParamWrapper(quats))
    params.add_param('scale_ori', ParamWrapper(scale_ori))
    config = Config()
    config.set('scale_min', scale_min)
    config.set('scale_max', scale_max)
    data_obj = Data()
    for key, val in downstream.items():
        data_obj.set(key, val)

    ag_op = ParameterTransformAutograd()
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
        print(f"  {name:30s}: max_abs={max_diff:.2e}, max_rel={max_rel:.2e}, "
              f"gt_max={gt_max:.2e} [{status}]")
        if not ok:
            all_pass = False
        return ok

    compare("v_quats (grad)", grad_a.param_grads['quats'], gt['v_q'])
    compare("v_scale_ori (grad)", grad_a.param_grads['scale_ori'], gt['v_so'])

    # h_quats
    h_qq_utri = torch.zeros(N, 10, device=device, dtype=dtype)
    idx = 0
    for i in range(4):
        for j in range(i, 4):
            h_qq_utri[:, idx] = gt['h_qq'][:, i, j]
            idx += 1
    compare("h_quats (upper-tri)", hess_a.param_hess['quats'], h_qq_utri)

    # h_scale_ori
    h_ss_utri = torch.stack([
        gt['h_ss'][:, 0, 0],
        gt['h_ss'][:, 0, 1],
        gt['h_ss'][:, 0, 2],
        gt['h_ss'][:, 1, 1],
        gt['h_ss'][:, 1, 2],
        gt['h_ss'][:, 2, 2],
    ], dim=-1)
    compare("h_scale_ori (upper-tri 3x3)", hess_a.param_hess['scale_ori'], h_ss_utri)

    # h_quats_scale_ori
    compare("h_quats_scale_ori", hess_a.cross_hess['quats_scale_ori'], gt['h_qs'].reshape(N, 12))

    # Cross-Hessians
    compare("h_quats_means", hess_a.cross_hess['quats_means'], gt['h_qm'].reshape(N, 12))
    compare("h_quats_opacity", hess_a.cross_hess['quats_opacity'], gt['h_qo'])
    compare("h_scale_ori_means", hess_a.cross_hess['scale_ori_means'], gt['h_sm'].reshape(N, 9))
    compare("h_scale_ori_opacity", hess_a.cross_hess['scale_ori_opacity'], gt['h_so'])

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# =============================================================================
# Test 7: Bwd2 CUDA vs Autograd Op (Direct)
# =============================================================================

def test_bwd2_cuda_vs_autograd_op(
    N: int = 16,
    scale_min: float = 0.01,
    scale_max: float = 2.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Direct comparison: ParameterTransformCUDA.forward_backward2 vs Autograd."""
    print(f"\n{'='*70}")
    print(f"Test bwd2 CUDA Op vs Autograd Op (direct)")
    print(f"  N={N}, scale=({scale_min}, {scale_max}), dtype={dtype}, seed={seed}")
    print(f"{'='*70}")

    device = torch.device("cuda")

    quats = generate_random_unit_quaternions(N, device, dtype, seed=seed)
    scale_ori = generate_random_scale_ori(N, device, dtype, seed=seed)

    torch.manual_seed(seed + 100)
    means = torch.randn(N, 3, device=device, dtype=dtype)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1

    torch.manual_seed(seed + 200)
    W = generate_random_symmetric_matrix(6, device, dtype)
    A_mean = torch.randn(6, 3, device=device, dtype=dtype)
    A_opa = torch.randn(6, 1, device=device, dtype=dtype)

    # Forward
    with torch.no_grad():
        covars = _compute_covars(quats, scale_ori, scale_min, scale_max)

    downstream = compute_quadratic_downstream(
        covars, means, opacity, W, A_mean, A_opa
    )

    config = Config()
    config.set('scale_min', scale_min)
    config.set('scale_max', scale_max)

    # --- CUDA ---
    params_c = Params()
    params_c.add_param('quats', ParamWrapper(quats.clone()))
    params_c.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    data_c = Data()
    for key, val in downstream.items():
        data_c.set(key, val.clone())

    cuda_op = ParameterTransformCUDA()
    cuda_op.initialize(params_c, config)
    _, grad_c, hess_c = cuda_op.forward_backward2(params_c, data_c, config)

    # --- Autograd ---
    params_a = Params()
    params_a.add_param('quats', ParamWrapper(quats.clone()))
    params_a.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    data_a = Data()
    for key, val in downstream.items():
        data_a.set(key, val.clone())

    ag_op = ParameterTransformAutograd()
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
        print(f"  {name:30s}: max_abs={max_diff:.2e}, max_rel={max_rel:.2e}, "
              f"ref_max={ref_max:.2e} [{status}]")
        if not ok:
            all_pass = False
        return ok

    compare("v_quats (grad)",
            grad_c.param_grads['quats'], grad_a.param_grads['quats'])
    compare("v_scale_ori (grad)",
            grad_c.param_grads['scale_ori'], grad_a.param_grads['scale_ori'])
    compare("h_quats",
            hess_c.param_hess['quats'], hess_a.param_hess['quats'])
    compare("h_scale_ori",
            hess_c.param_hess['scale_ori'], hess_a.param_hess['scale_ori'])
    compare("h_quats_scale_ori",
            hess_c.cross_hess['quats_scale_ori'], hess_a.cross_hess['quats_scale_ori'])
    compare("h_quats_means",
            hess_c.cross_hess['quats_means'], hess_a.cross_hess['quats_means'])
    compare("h_quats_opacity",
            hess_c.cross_hess['quats_opacity'], hess_a.cross_hess['quats_opacity'])
    compare("h_scale_ori_means",
            hess_c.cross_hess['scale_ori_means'], hess_a.cross_hess['scale_ori_means'])
    compare("h_scale_ori_opacity",
            hess_c.cross_hess['scale_ori_opacity'], hess_a.cross_hess['scale_ori_opacity'])

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


# =============================================================================
# Test 8: Bwd2 Hessian Symmetry
# =============================================================================

def test_bwd2_hessian_symmetry(
    N: int = 8,
    scale_min: float = 0.01,
    scale_max: float = 2.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 42,
):
    """Test that H_{q,q} is symmetric by checking upper-tri vs lower-tri."""
    print(f"\n{'='*70}")
    print(f"Test bwd2 Hessian Symmetry")
    print(f"  N={N}, dtype={dtype}")
    print(f"{'='*70}")

    device = torch.device("cuda")
    quats = generate_random_unit_quaternions(N, device, dtype, seed=seed)
    scale_ori = generate_random_scale_ori(N, device, dtype, seed=seed)

    torch.manual_seed(seed + 100)
    means = torch.randn(N, 3, device=device, dtype=dtype)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.8 + 0.1

    torch.manual_seed(seed + 200)
    W = generate_random_symmetric_matrix(6, device, dtype)
    A_mean = torch.randn(6, 3, device=device, dtype=dtype)
    A_opa = torch.randn(6, 1, device=device, dtype=dtype)

    gt = compute_ground_truth_autograd(
        quats, scale_ori, means, opacity, scale_min, scale_max,
        W, A_mean, A_opa,
    )

    # Check h_qq symmetry
    h_qq = gt['h_qq']  # [N, 4, 4]
    sym_diff_qq = (h_qq - h_qq.transpose(-1, -2)).abs().max().item()

    # Check h_ss symmetry (should be diagonal, so symmetric by definition)
    h_ss = gt['h_ss']  # [N, 3, 3]
    sym_diff_ss = (h_ss - h_ss.transpose(-1, -2)).abs().max().item()

    atol = 1e-10
    ok_qq = sym_diff_qq < atol
    ok_ss = sym_diff_ss < atol

    print(f"  H_qq symmetry max diff: {sym_diff_qq:.2e} [{'PASS' if ok_qq else 'FAIL'}]")
    print(f"  H_ss symmetry max diff: {sym_diff_ss:.2e} [{'PASS' if ok_ss else 'FAIL'}]")

    # Note: H_ss off-diagonal is NOT zero in general because the Gauss-Newton
    # term J^T H_cc J contributes off-diagonal elements.  The CUDA kernel
    # intentionally stores only the diagonal as an approximation.
    # Here we merely report the magnitude for information.
    h_ss_offdiag = 0.0
    for i in range(3):
        for j in range(3):
            if i != j:
                h_ss_offdiag = max(h_ss_offdiag, h_ss[:, i, j].abs().max().item())
    print(f"  H_ss off-diagonal max:  {h_ss_offdiag:.2e} (non-zero expected from GN term)")

    return ok_qq and ok_ss


# =============================================================================
# Test 9: Edge Cases
# =============================================================================

def test_edge_cases(
    scale_min: float = 0.01,
    scale_max: float = 2.0,
    dtype: torch.dtype = torch.float64,
):
    """Test edge cases: identity quaternion, zero scale_ori, N=0, N=1."""
    print(f"\n{'='*60}")
    print(f"Test Edge Cases")
    print(f"{'='*60}")

    device = torch.device("cuda")
    all_ok = True

    config = Config()
    config.set('scale_min', scale_min)
    config.set('scale_max', scale_max)

    # --- Identity quaternion ---
    print("  Identity quaternion:")
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
    so = torch.tensor([[0.0, 0.0, 0.0]], device=device, dtype=dtype)

    params = Params()
    params.add_param('quats', ParamWrapper(q))
    params.add_param('scale_ori', ParamWrapper(so))
    data = Data()

    cuda_op = ParameterTransformCUDA()
    cuda_op.initialize(params, config)
    result = cuda_op.forward_only(params, data, config)
    covars = result.outputs['covars']

    ag_op = ParameterTransformAutograd()
    ag_op.initialize(params, config)
    result_ag = ag_op.forward_only(params, data, config)
    covars_ag = result_ag.outputs['covars']

    diff = (covars - covars_ag).abs().max().item()
    ok = diff < 1e-12
    print(f"    fwd diff: {diff:.2e} {'PASS' if ok else 'FAIL'}")
    all_ok &= ok

    # For identity quat, R=I, so C = diag(s0^2, s1^2, s2^2)
    sig = torch.sigmoid(torch.tensor(0.0, dtype=dtype))
    s_val = sig.item() * (scale_max - scale_min) + scale_min
    expected_diag = s_val ** 2
    ok2 = abs(covars[0, 0].item() - expected_diag) < 1e-10
    ok3 = abs(covars[0, 1].item()) < 1e-10  # off-diagonal should be 0
    print(f"    C00={covars[0,0].item():.6f} (expected {expected_diag:.6f}) "
          f"{'PASS' if ok2 else 'FAIL'}")
    print(f"    C01={covars[0,1].item():.2e} (expected ~0) "
          f"{'PASS' if ok3 else 'FAIL'}")
    all_ok &= ok2 and ok3

    # --- N=0 ---
    print("  N=0:")
    q0 = torch.empty(0, 4, device=device, dtype=dtype)
    so0 = torch.empty(0, 3, device=device, dtype=dtype)
    params0 = Params()
    params0.add_param('quats', ParamWrapper(q0))
    params0.add_param('scale_ori', ParamWrapper(so0))

    cuda_op0 = ParameterTransformCUDA()
    cuda_op0.initialize(params0, config)
    result0 = cuda_op0.forward_only(params0, data, config)
    ok_n0 = result0.outputs['covars'].shape == (0, 6)
    print(f"    output shape: {result0.outputs['covars'].shape} "
          f"{'PASS' if ok_n0 else 'FAIL'}")
    all_ok &= ok_n0

    # --- N=1 backward ---
    print("  N=1 backward:")
    q1 = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device, dtype=dtype)
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    so1 = torch.tensor([[1.0, -1.0, 0.5]], device=device, dtype=dtype)
    v_c1 = torch.ones(1, 6, device=device, dtype=dtype)

    params1 = Params()
    params1.add_param('quats', ParamWrapper(q1))
    params1.add_param('scale_ori', ParamWrapper(so1))
    data1 = Data()
    data1.set('v_covars', v_c1)

    cuda_op1 = ParameterTransformCUDA()
    cuda_op1.initialize(params1, config)
    _, grad1 = cuda_op1.forward_backward1(params1, data1, config)

    vq = grad1.param_grads['quats']
    vso = grad1.param_grads['scale_ori']
    ok_shape = vq.shape == (1, 4) and vso.shape == (1, 3)
    ok_finite = torch.isfinite(vq).all().item() and torch.isfinite(vso).all().item()
    print(f"    shapes: vq={vq.shape}, vso={vso.shape} "
          f"{'PASS' if ok_shape else 'FAIL'}")
    print(f"    all finite: {'PASS' if ok_finite else 'FAIL'}")
    all_ok &= ok_shape and ok_finite

    print(f"  Edge cases: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return all_ok


# =============================================================================
# Test 10: Factory Function
# =============================================================================

def test_factory_function():
    """Test the create_parameter_transform factory function."""
    print(f"\n{'='*60}")
    print(f"Test Factory Function")
    print(f"{'='*60}")

    try:
        cuda_op = create_parameter_transform("cuda")
        assert isinstance(cuda_op, ParameterTransformCUDA)
        print("  create_parameter_transform('cuda'): PASS")
    except Exception as e:
        print(f"  create_parameter_transform('cuda'): FAIL ({e})")
        return False

    try:
        ag_op = create_parameter_transform("autograd")
        assert isinstance(ag_op, ParameterTransformAutograd)
        print("  create_parameter_transform('autograd'): PASS")
    except Exception as e:
        print(f"  create_parameter_transform('autograd'): FAIL ({e})")
        return False

    try:
        create_parameter_transform("invalid")
        print("  create_parameter_transform('invalid') should raise: FAIL")
        return False
    except ValueError:
        print("  create_parameter_transform('invalid') raises ValueError: PASS")

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
            N=N, scale_min=0.01, scale_max=2.0,
            dtype=torch.float64, seed=42
        )
        all_pass = all_pass and ok

    # Test with different scale ranges
    for s_min, s_max in [(0.001, 1.0), (0.1, 5.0), (0.5, 0.6)]:
        ok = test_bwd2_cuda_vs_autograd_ground_truth(
            N=8, scale_min=s_min, scale_max=s_max,
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
    print("  ParameterTransform Module Test Suite")
    print("=" * 70)

    results = []

    # Test 1: Forward consistency
    results.append(("Forward f64", test_forward_consistency(
        N=1000, dtype=torch.float64)))
    results.append(("Forward f32", test_forward_consistency(
        N=1000, dtype=torch.float32)))

    # Test 2: Backward consistency
    results.append(("Backward f64", test_backward_consistency(
        N=1000, dtype=torch.float64)))
    results.append(("Backward f32", test_backward_consistency(
        N=1000, dtype=torch.float32)))

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
