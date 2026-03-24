"""
Test script for ConicsMu module.

Validates the CUDA implementation against the PyTorch autograd
reference implementation by comparing:
1. Forward outputs (conics, mus)
2. Backward outputs (v_cov2d_3x3)

Usage:
    python tests/test_conics_mu.py

Author: Dy164
Date: 2026-02-10
"""

import torch
import math
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.data import Params, Data, Config
from gsplat2ct.ops.conics_mu import ConicsMuCUDA, ConicsMuAutograd, create_conics_mu


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
    
    Args:
        C: Number of cameras
        N: Number of Gaussians per camera
        device: PyTorch device
        dtype: PyTorch dtype
        seed: Random seed
    
    Returns:
        cov2d_3x3: (C, N, 6) in upper-triangular storage
    """
    torch.manual_seed(seed)
    
    # Generate lower triangular L with positive diagonal
    # L = [[l00, 0, 0],
    #       [l10, l11, 0],
    #       [l20, l21, l22]]
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


def test_forward_consistency(
    C: int = 2,
    N: int = 100,
    eps: float = 0.3,
    dtype: torch.dtype = torch.float32,
):
    """Test that CUDA and autograd forward passes produce the same results."""
    print(f"\n{'='*60}")
    print(f"Test Forward Consistency: C={C}, N={N}, eps={eps}, dtype={dtype}")
    print(f"{'='*60}")
    
    device = torch.device("cuda")
    cov2d_3x3 = generate_random_spd_cov2d_3x3(C, N, device, dtype)
    
    # Create a simple parameter-like wrapper
    class ParamWrapper:
        def __init__(self, data):
            self.data = data
    
    params = Params()
    params.add_param('cov2d_3x3', ParamWrapper(cov2d_3x3))
    config = Config()
    config.set('eps', eps)
    data = Data()
    
    # CUDA forward
    cuda_op = ConicsMuCUDA()
    cuda_op.initialize(params, config)
    cuda_result = cuda_op.forward_only(params, data, config)
    
    conics_cuda = cuda_result.outputs['conics']
    mus_cuda = cuda_result.outputs['mus']
    
    # Autograd forward
    autograd_op = ConicsMuAutograd()
    autograd_op.initialize(params, config)
    autograd_result = autograd_op.forward_only(params, data, config)
    
    conics_autograd = autograd_result.outputs['conics']
    mus_autograd = autograd_result.outputs['mus']
    
    # Compare
    conics_diff = (conics_cuda - conics_autograd).abs().max().item()
    mus_diff = (mus_cuda - mus_autograd).abs().max().item()
    
    atol = 1e-5 if dtype == torch.float32 else 1e-10
    
    print(f"  conics max abs diff: {conics_diff:.2e}")
    print(f"  mus max abs diff:    {mus_diff:.2e}")
    
    conics_ok = conics_diff < atol
    mus_ok = mus_diff < atol
    
    print(f"  conics match: {'PASS' if conics_ok else 'FAIL'} (atol={atol:.0e})")
    print(f"  mus match:    {'PASS' if mus_ok else 'FAIL'} (atol={atol:.0e})")
    
    return conics_ok and mus_ok


def test_backward_consistency(
    C: int = 2,
    N: int = 100,
    eps: float = 0.3,
    dtype: torch.dtype = torch.float32,
):
    """Test that CUDA and autograd backward passes produce the same gradients."""
    print(f"\n{'='*60}")
    print(f"Test Backward Consistency: C={C}, N={N}, eps={eps}, dtype={dtype}")
    print(f"{'='*60}")
    
    device = torch.device("cuda")
    cov2d_3x3 = generate_random_spd_cov2d_3x3(C, N, device, dtype)
    
    # Generate random upstream gradients
    torch.manual_seed(123)
    v_conics = torch.randn(C, N, 3, device=device, dtype=dtype)
    v_mus = torch.randn(C, N, device=device, dtype=dtype)
    
    class ParamWrapper:
        def __init__(self, data):
            self.data = data
    
    config = Config()
    config.set('eps', eps)
    
    # --- CUDA backward ---
    params_cuda = Params()
    params_cuda.add_param('cov2d_3x3', ParamWrapper(cov2d_3x3.clone()))
    data_cuda = Data()
    data_cuda.set('v_conics', v_conics.clone())
    data_cuda.set('v_mus', v_mus.clone())
    
    cuda_op = ConicsMuCUDA()
    cuda_op.initialize(params_cuda, config)
    cuda_fwd, cuda_grad = cuda_op.forward_backward1(params_cuda, data_cuda, config)
    
    v_cov2d_3x3_cuda = cuda_grad.param_grads['cov2d_3x3']
    
    # --- Autograd backward ---
    params_ag = Params()
    params_ag.add_param('cov2d_3x3', ParamWrapper(cov2d_3x3.clone()))
    data_ag = Data()
    data_ag.set('v_conics', v_conics.clone())
    data_ag.set('v_mus', v_mus.clone())
    
    autograd_op = ConicsMuAutograd()
    autograd_op.initialize(params_ag, config)
    ag_fwd, ag_grad = autograd_op.forward_backward1(params_ag, data_ag, config)
    
    v_cov2d_3x3_ag = ag_grad.param_grads['cov2d_3x3']
    
    # Compare forward outputs first
    conics_diff = (cuda_fwd.outputs['conics'] - ag_fwd.outputs['conics']).abs().max().item()
    mus_diff = (cuda_fwd.outputs['mus'] - ag_fwd.outputs['mus']).abs().max().item()
    
    print(f"  Forward conics max diff: {conics_diff:.2e}")
    print(f"  Forward mus max diff:    {mus_diff:.2e}")
    
    # Compare gradients
    grad_diff = (v_cov2d_3x3_cuda - v_cov2d_3x3_ag).abs()
    grad_max_diff = grad_diff.max().item()
    grad_mean_diff = grad_diff.mean().item()
    
    # Per-element max diff
    for i, name in enumerate(['a', 'b', 'c', 'd', 'e', 'f']):
        elem_diff = (v_cov2d_3x3_cuda[..., i] - v_cov2d_3x3_ag[..., i]).abs().max().item()
        print(f"  v_{name} max abs diff: {elem_diff:.2e}")
    
    atol = 1e-4 if dtype == torch.float32 else 1e-8
    
    print(f"  Overall grad max abs diff:  {grad_max_diff:.2e}")
    print(f"  Overall grad mean abs diff: {grad_mean_diff:.2e}")
    
    grad_ok = grad_max_diff < atol
    print(f"  Backward match: {'PASS' if grad_ok else 'FAIL'} (atol={atol:.0e})")
    
    return grad_ok


def test_backward_vs_numerical(
    C: int = 1,
    N: int = 10,
    eps: float = 0.3,
    dtype: torch.dtype = torch.float64,
    fd_eps: float = 1e-6,
):
    """Validate backward using finite differences (gold standard).
    
    Uses float64 for better numerical accuracy.
    """
    print(f"\n{'='*60}")
    print(f"Test Backward vs Numerical (Finite Diff): C={C}, N={N}, eps={eps}")
    print(f"{'='*60}")
    
    device = torch.device("cuda")
    cov2d_3x3 = generate_random_spd_cov2d_3x3(C, N, device, dtype, seed=99)
    
    # Random upstream gradients
    torch.manual_seed(77)
    v_conics = torch.randn(C, N, 3, device=device, dtype=dtype)
    v_mus = torch.randn(C, N, device=device, dtype=dtype)
    
    # Compute autograd gradient
    class ParamWrapper:
        def __init__(self, data):
            self.data = data
    
    config = Config()
    config.set('eps', eps)
    
    params_ag = Params()
    params_ag.add_param('cov2d_3x3', ParamWrapper(cov2d_3x3.clone()))
    data_ag = Data()
    data_ag.set('v_conics', v_conics.clone())
    data_ag.set('v_mus', v_mus.clone())
    
    autograd_op = ConicsMuAutograd()
    autograd_op.initialize(params_ag, config)
    _, ag_grad = autograd_op.forward_backward1(params_ag, data_ag, config)
    v_cov_ag = ag_grad.param_grads['cov2d_3x3']
    
    # Compute numerical gradient via finite differences
    # f(x) = v_conics . conics(x) + v_mus . mus(x)
    # df/dx_i ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
    
    from gsplat2ct.ops.conics_mu import _compute_conics_mu
    
    def scalar_fn(cov_flat):
        """Compute scalar loss = v_conics * conics + v_mus * mus."""
        cov = cov_flat.view(C, N, 6)
        with torch.no_grad():
            conics, mus = _compute_conics_mu(cov, eps)
        return (v_conics * conics).sum() + (v_mus * mus).sum()
    
    cov_flat = cov2d_3x3.reshape(-1).clone()
    grad_numerical = torch.zeros_like(cov_flat)
    
    for i in range(cov_flat.numel()):
        cov_plus = cov_flat.clone()
        cov_plus[i] += fd_eps
        cov_minus = cov_flat.clone()
        cov_minus[i] -= fd_eps
        
        f_plus = scalar_fn(cov_plus)
        f_minus = scalar_fn(cov_minus)
        grad_numerical[i] = (f_plus - f_minus) / (2.0 * fd_eps)
    
    grad_numerical = grad_numerical.view(C, N, 6)
    
    # Compare
    grad_diff = (v_cov_ag - grad_numerical).abs()
    max_diff = grad_diff.max().item()
    mean_diff = grad_diff.mean().item()
    
    # Per-element
    for i, name in enumerate(['a', 'b', 'c', 'd', 'e', 'f']):
        elem_diff = grad_diff[..., i].max().item()
        print(f"  v_{name} max abs diff vs numerical: {elem_diff:.2e}")
    
    atol = 1e-4  # Finite diff has limited accuracy
    
    print(f"  Overall max abs diff:  {max_diff:.2e}")
    print(f"  Overall mean abs diff: {mean_diff:.2e}")
    
    ok = max_diff < atol
    print(f"  Numerical gradient check: {'PASS' if ok else 'FAIL'} (atol={atol:.0e})")
    
    return ok


def test_factory_function():
    """Test the create_conics_mu factory function."""
    print(f"\n{'='*60}")
    print(f"Test Factory Function")
    print(f"{'='*60}")
    
    try:
        cuda_op = create_conics_mu("cuda")
        assert isinstance(cuda_op, ConicsMuCUDA)
        print("  create_conics_mu('cuda'): PASS")
    except Exception as e:
        print(f"  create_conics_mu('cuda'): FAIL ({e})")
        return False
    
    try:
        ag_op = create_conics_mu("autograd")
        assert isinstance(ag_op, ConicsMuAutograd)
        print("  create_conics_mu('autograd'): PASS")
    except Exception as e:
        print(f"  create_conics_mu('autograd'): FAIL ({e})")
        return False
    
    try:
        create_conics_mu("invalid")
        print("  create_conics_mu('invalid') should raise: FAIL")
        return False
    except ValueError:
        print("  create_conics_mu('invalid') raises ValueError: PASS")
    
    return True


def test_edge_cases(dtype: torch.dtype = torch.float32):
    """Test edge cases: single element, large eps, near-singular."""
    print(f"\n{'='*60}")
    print(f"Test Edge Cases")
    print(f"{'='*60}")
    
    device = torch.device("cuda")
    all_ok = True
    
    # Test 1: Single Gaussian (C=1, N=1)
    print("  [1] Single Gaussian (C=1, N=1)...")
    cov = generate_random_spd_cov2d_3x3(1, 1, device, dtype, seed=0)
    
    class ParamWrapper:
        def __init__(self, data):
            self.data = data
    
    config = Config()
    config.set('eps', 0.3)
    
    params = Params()
    params.add_param('cov2d_3x3', ParamWrapper(cov))
    data = Data()
    
    cuda_op = ConicsMuCUDA()
    cuda_op.initialize(params, config)
    result = cuda_op.forward_only(params, data, config)
    
    conics = result.outputs['conics']
    mus = result.outputs['mus']
    
    ok = conics.shape == (1, 1, 3) and mus.shape == (1, 1) and mus.item() > 0
    print(f"    Shape & value check: {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok
    
    # Test 2: Large eps
    print("  [2] Large blur (eps=10.0)...")
    config_large = Config()
    config_large.set('eps', 10.0)
    
    result_large = cuda_op.forward_only(params, data, config_large)
    conics_large = result_large.outputs['conics']
    mus_large = result_large.outputs['mus']
    
    ok = torch.isfinite(conics_large).all().item() and torch.isfinite(mus_large).all().item()
    print(f"    Finite check: {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok
    
    # Test 3: Larger batch
    print("  [3] Larger batch (C=4, N=1000)...")
    cov_large = generate_random_spd_cov2d_3x3(4, 1000, device, dtype, seed=7)
    params_large = Params()
    params_large.add_param('cov2d_3x3', ParamWrapper(cov_large))
    
    cuda_op2 = ConicsMuCUDA()
    cuda_op2.initialize(params_large, config)
    result_large2 = cuda_op2.forward_only(params_large, data, config)
    
    conics_l = result_large2.outputs['conics']
    mus_l = result_large2.outputs['mus']
    
    ok = (conics_l.shape == (4, 1000, 3) and mus_l.shape == (4, 1000)
          and torch.isfinite(conics_l).all().item()
          and torch.isfinite(mus_l).all().item()
          and (mus_l > 0).all().item())
    print(f"    Shape & finite & positive check: {'PASS' if ok else 'FAIL'}")
    all_ok = all_ok and ok
    
    return all_ok


def main():
    """Run all tests."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        return
    
    print("=" * 60)
    print("ConicsMu Module Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Forward consistency
    results['forward_f32'] = test_forward_consistency(C=7, N=10000, eps=2.5, dtype=torch.float32)
    results['forward_f64'] = test_forward_consistency(C=7, N=10000, eps=2.5, dtype=torch.float64)
    
    # Test 2: Backward consistency (CUDA vs autograd)
    results['backward_f32'] = test_backward_consistency(C=7, N=10000, eps=2.5, dtype=torch.float32)
    results['backward_f64'] = test_backward_consistency(C=7, N=10000, eps=2.5, dtype=torch.float64)
    
    # Test 3: Backward vs numerical (finite differences)
    results['numerical'] = test_backward_vs_numerical(C=2, N=10, eps=0.1, dtype=torch.float64)
    
    # Test 4: Factory function
    results['factory'] = test_factory_function()
    
    # Test 5: Edge cases
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
