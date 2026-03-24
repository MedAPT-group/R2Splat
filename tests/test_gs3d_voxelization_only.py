"""
Test script for GS3D Voxelization-Only Pipeline.

This script validates the CUDA chain implementation against PyTorch autograd
for the full 4-stage pipeline:
    ParameterTransform → WorldToVolumeMapping → VoxelizeToVoxels → L2Loss

Input parameters:
    - means:     (N, 3)  world-space Gaussian centres
    - quats:     (N, 4)  unit quaternions
    - scale_ori: (N, 3)  raw scale parameters
    - opacity:   (N,)    per-Gaussian opacity

Since this uses L2 loss only, the results should match well between
CUDA and autograd implementations.

Test stages:
1. Forward consistency: CUDA vs Autograd render output and loss
2. First-order backward consistency: Gradients match
3. Second-order backward consistency: Hessians match
4. Loss value sanity check
5. Gradient vs finite difference validation

Usage:
    python tests/test_gs3d_voxelization_only.py

Author: Dy164
Date: 2026-02-23
"""

import torch
import math
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.data import Params, Data, Config
from gsplat2ct.ops.gs3d_voxelization_only import (
    GS3DVoxelizationOnlyCUDA,
    GS3DVoxelizationOnlyAutograd,
    create_gs3d_voxelization_only,
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

def generate_random_unit_quaternions(
    N: int,
    device: torch.device, dtype: torch.dtype,
    seed: int = 45,
) -> torch.Tensor:
    """Generate random unit quaternions (N, 4) as (w, x, y, z)."""
    torch.manual_seed(seed)
    quats = torch.randn(N, 4, device=device, dtype=dtype)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    # Ensure w > 0 for canonical form
    quats[quats[:, 0] < 0] *= -1
    return quats


def generate_test_data(
    N: int = 5,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    dim0: int = 8,
    dim1: int = 8,
    dim2: int = 8,
    seed: int = 42,
):
    """Generate a complete test dataset for the GS3D voxelization-only pipeline.

    Parameters are chosen so that:
    - Gaussians are within the volume extent
    - Unit quaternions produce well-conditioned covariances
    - Scale parameters are moderate
    - Opacities are moderate
    - Small volume size to keep the test fast

    Returns:
        means:     (N, 3)
        quats:     (N, 4)
        scale_ori: (N, 3)
        opacity:   (N,)
        ranges:    (3,)
        centers:   (3,)
        target:    (dim0, dim1, dim2)
    """
    if device is None:
        device = torch.device("cuda")

    # Volume physical extent
    # ranges defines how large the physical space is per axis
    # centers defines the physical center
    ranges = torch.tensor([10.0, 10.0, 10.0], dtype=dtype, device=device)
    centers = torch.tensor([0.0, 0.0, 0.0], dtype=dtype, device=device)

    # Generate Gaussian means within the physical extent [-5, 5]^3
    torch.manual_seed(seed + 1)
    means = (torch.rand(N, 3, device=device, dtype=dtype) - 0.5) * 6.0  # [-3, 3]

    # Generate unit quaternions
    quats = generate_random_unit_quaternions(N, device, dtype, seed=seed + 2)

    # Scale parameters: moderate values
    # After sigmoid scaling: scale_min + (scale_max - scale_min) * sigmoid(scale_ori)
    # With scale_min=0.01, scale_max=2.0, scale_ori≈0 gives scale≈1.0
    torch.manual_seed(seed + 3)
    scale_ori = torch.randn(N, 3, device=device, dtype=dtype) * 0.5

    # Opacity: moderate values
    torch.manual_seed(seed + 4)
    opacity = torch.rand(N, device=device, dtype=dtype) * 0.2 - 0.1  # [-0.1, 0.1]

    # Target: zeros (the actual target doesn't matter much for gradient validation)
    target = torch.zeros(dim0, dim1, dim2, device=device, dtype=dtype)

    return means, quats, scale_ori, opacity, ranges, centers, target


# ============================================================================
# Test Functions
# ============================================================================

def test_forward_consistency(
    N: int = 5,
    dim0: int = 8,
    dim1: int = 8,
    dim2: int = 8,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 1e-4,
) -> bool:
    """Test 1: Forward pass consistency between CUDA and Autograd."""
    print("\n" + "=" * 70)
    print("TEST 1: Forward Pass Consistency (CUDA vs Autograd)")
    print("=" * 70)

    device = torch.device(device)
    means, quats, scale_ori, opacity, ranges, centers, target = generate_test_data(
        N, device, dtype, dim0, dim1, dim2
    )

    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('quats', ParamWrapper(quats))
    params.add_param('scale_ori', ParamWrapper(scale_ori))
    params.add_param('opacity', ParamWrapper(opacity))

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', 1e-4)
    config.set('radius_clip', 3.0)
    config.set('scale_min', 0.01)
    config.set('scale_max', 2.0)
    config.set('tile_size', 4)

    data_obj = Data()
    data_obj.set('ranges', ranges)
    data_obj.set('centers', centers)
    data_obj.set('target', target)

    # CUDA forward
    cuda_op = GS3DVoxelizationOnlyCUDA()
    cuda_op.initialize(params, config)
    cuda_fwd = cuda_op.forward_only(params, data_obj, config)
    loss_cuda = cuda_fwd.outputs['loss']
    render_cuda = cuda_fwd.outputs['render_output']

    # Autograd forward
    ag_op = GS3DVoxelizationOnlyAutograd()
    ag_op.initialize(params, config)
    ag_fwd = ag_op.forward_only(params, data_obj, config)
    loss_ag = ag_fwd.outputs['loss']
    render_ag = ag_fwd.outputs['render_output']

    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}")
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
    N: int = 5,
    dim0: int = 8,
    dim1: int = 8,
    dim2: int = 8,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.05,
) -> bool:
    """Test 2: First-order backward consistency."""
    print("\n" + "=" * 70)
    print("TEST 2: First-Order Backward Consistency (CUDA vs Autograd)")
    print("=" * 70)

    device = torch.device(device)
    means, quats, scale_ori, opacity, ranges, centers, target = generate_test_data(
        N, device, dtype, dim0, dim1, dim2
    )

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', 1e-4)
    config.set('radius_clip', 3.0)
    config.set('scale_min', 0.01)
    config.set('scale_max', 2.0)
    config.set('tile_size', 4)

    data_obj = Data()
    data_obj.set('ranges', ranges)
    data_obj.set('centers', centers)
    data_obj.set('target', target)

    # CUDA
    params_cuda = Params()
    params_cuda.add_param('means', ParamWrapper(means.clone()))
    params_cuda.add_param('quats', ParamWrapper(quats.clone()))
    params_cuda.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    params_cuda.add_param('opacity', ParamWrapper(opacity.clone()))

    cuda_op = GS3DVoxelizationOnlyCUDA()
    cuda_op.initialize(params_cuda, config)
    _, cuda_grad = cuda_op.forward_backward1(params_cuda, data_obj, config)

    # Autograd
    params_ag = Params()
    params_ag.add_param('means', ParamWrapper(means.clone()))
    params_ag.add_param('quats', ParamWrapper(quats.clone()))
    params_ag.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    params_ag.add_param('opacity', ParamWrapper(opacity.clone()))

    ag_op = GS3DVoxelizationOnlyAutograd()
    ag_op.initialize(params_ag, config)
    _, ag_grad = ag_op.forward_backward1(params_ag, data_obj, config)

    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}")

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
    all_pass &= compare_tensors("v_means    ", cuda_grad.param_grads['means'], ag_grad.param_grads['means'])
    all_pass &= compare_tensors("v_quats    ", cuda_grad.param_grads['quats'], ag_grad.param_grads['quats'])
    all_pass &= compare_tensors("v_scale_ori", cuda_grad.param_grads['scale_ori'], ag_grad.param_grads['scale_ori'])
    all_pass &= compare_tensors("v_opacity  ", cuda_grad.param_grads['opacity'], ag_grad.param_grads['opacity'])

    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 2 {'PASSED' if all_pass else 'FAILED'}")

    return all_pass


def test_backward2_consistency(
    N: int = 3,
    dim0: int = 8,
    dim1: int = 8,
    dim2: int = 8,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    rtol: float = 0.15,
) -> bool:
    """Test 3: Second-order backward consistency."""
    print("\n" + "=" * 70)
    print("TEST 3: Second-Order Backward Consistency (CUDA vs Autograd)")
    print("=" * 70)

    device = torch.device(device)
    means, quats, scale_ori, opacity, ranges, centers, target = generate_test_data(
        N, device, dtype, dim0, dim1, dim2
    )

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', 1e-4)
    config.set('radius_clip', 3.0)
    config.set('scale_min', 0.01)
    config.set('scale_max', 2.0)
    config.set('tile_size', 4)

    data_obj = Data()
    data_obj.set('ranges', ranges)
    data_obj.set('centers', centers)
    data_obj.set('target', target)

    # CUDA
    params_cuda = Params()
    params_cuda.add_param('means', ParamWrapper(means.clone()))
    params_cuda.add_param('quats', ParamWrapper(quats.clone()))
    params_cuda.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    params_cuda.add_param('opacity', ParamWrapper(opacity.clone()))

    cuda_op = GS3DVoxelizationOnlyCUDA()
    cuda_op.initialize(params_cuda, config)
    _, cuda_grad, cuda_hess = cuda_op.forward_backward2(params_cuda, data_obj, config)

    # Autograd
    params_ag = Params()
    params_ag.add_param('means', ParamWrapper(means.clone()))
    params_ag.add_param('quats', ParamWrapper(quats.clone()))
    params_ag.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    params_ag.add_param('opacity', ParamWrapper(opacity.clone()))

    ag_op = GS3DVoxelizationOnlyAutograd()
    ag_op.initialize(params_ag, config)
    _, ag_grad, ag_hess = ag_op.forward_backward2(params_ag, data_obj, config)

    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}")

    def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor,
                        threshold: float = None) -> bool:
        if t1 is None or t2 is None:
            print(f"  {name}: SKIPPED (None)")
            return True
        threshold = threshold or rtol
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = max(t1.abs().max().item(), t2.abs().max().item(), 1e-10)
        rel_error = max_diff / scale
        passed = rel_error < threshold
        status = "✓" if passed else "✗"
        print(f"  {name}: max_diff={max_diff:.6e}, scale={scale:.6e}, rel_error={rel_error:.4f} {status}")
        if not passed:
            print(f"    t1 range: [{t1.min().item():.6e}, {t1.max().item():.6e}]")
            print(f"    t2 range: [{t2.min().item():.6e}, {t2.max().item():.6e}]")
        return passed

    print("\n--- First-Order Gradients ---")
    all_pass = True
    all_pass &= compare_tensors("v_means    ", cuda_grad.param_grads['means'], ag_grad.param_grads['means'])
    all_pass &= compare_tensors("v_quats    ", cuda_grad.param_grads['quats'], ag_grad.param_grads['quats'])
    all_pass &= compare_tensors("v_scale_ori", cuda_grad.param_grads['scale_ori'], ag_grad.param_grads['scale_ori'])
    all_pass &= compare_tensors("v_opacity  ", cuda_grad.param_grads['opacity'], ag_grad.param_grads['opacity'])

    print("\n--- Self Hessians ---")
    all_pass &= compare_tensors("h_means    ", cuda_hess.param_hess['means'], ag_hess.param_hess['means'])
    all_pass &= compare_tensors("h_quats    ", cuda_hess.param_hess['quats'], ag_hess.param_hess['quats'])
    all_pass &= compare_tensors("h_scale_ori", cuda_hess.param_hess['scale_ori'], ag_hess.param_hess['scale_ori'])
    all_pass &= compare_tensors("h_opacity  ", cuda_hess.param_hess['opacity'], ag_hess.param_hess['opacity'])

    print("\n--- Cross-Term Hessians ---")
    common_cross_keys = [
        'quats_scale_ori',
        'quats_means',
        'scale_ori_means',
        'means_opacity',
        'quats_opacity',
        'scale_ori_opacity',
    ]
    for key in common_cross_keys:
        cuda_val = cuda_hess.cross_hess.get(key)
        ag_val = ag_hess.cross_hess.get(key)
        if cuda_val is not None and ag_val is not None:
            all_pass &= compare_tensors(f"h_{key}", cuda_val, ag_val)
        else:
            print(f"  h_{key}: SKIPPED (not present in both)")

    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 3 {'PASSED' if all_pass else 'FAILED'}")

    return all_pass


def test_loss_value_sanity(
    N: int = 10,
    dim0: int = 16,
    dim1: int = 16,
    dim2: int = 16,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> bool:
    """Test 4: Verify loss value is reasonable."""
    print("\n" + "=" * 70)
    print("TEST 4: Loss Value Sanity Check")
    print("=" * 70)

    device = torch.device(device)
    means, quats, scale_ori, opacity, ranges, centers, target = generate_test_data(
        N, device, dtype, dim0, dim1, dim2
    )

    params = Params()
    params.add_param('means', ParamWrapper(means))
    params.add_param('quats', ParamWrapper(quats))
    params.add_param('scale_ori', ParamWrapper(scale_ori))
    params.add_param('opacity', ParamWrapper(opacity))

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', 1e-4)
    config.set('radius_clip', 3.0)
    config.set('scale_min', 0.01)
    config.set('scale_max', 2.0)
    config.set('tile_size', 4)

    data_obj = Data()
    data_obj.set('ranges', ranges)
    data_obj.set('centers', centers)
    data_obj.set('target', target)

    cuda_op = GS3DVoxelizationOnlyCUDA()
    cuda_op.initialize(params, config)
    result = cuda_op.forward_only(params, data_obj, config)

    loss = result.outputs['loss'].item()
    render_output = result.outputs['render_output']

    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}")
    print(f"Loss: {loss:.6f}")
    print(f"Render output range: [{render_output.min().item():.4f}, {render_output.max().item():.4f}]")
    print(f"Render output mean: {render_output.mean().item():.6f}")
    print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")

    is_finite = np.isfinite(loss)
    is_positive = loss >= 0

    status = "✓" if (is_finite and is_positive) else "✗"
    print(f"\n{status} TEST 4 {'PASSED' if (is_finite and is_positive) else 'FAILED'}")

    return is_finite and is_positive


def test_gradient_finite_difference(
    N: int = 3,
    dim0: int = 8,
    dim1: int = 8,
    dim2: int = 8,
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
    means, quats, scale_ori, opacity, ranges, centers, target = generate_test_data(
        N, device, dtype, dim0, dim1, dim2, seed=99
    )

    config = Config()
    config.set('dim2', dim2)
    config.set('dim1', dim1)
    config.set('dim0', dim0)
    config.set('eps3d', 1e-4)
    config.set('radius_clip', 3.0)
    config.set('scale_min', 0.01)
    config.set('scale_max', 2.0)
    config.set('tile_size', 4)

    def compute_loss(m, q, so, opa):
        """Helper: compute loss via autograd pipeline."""
        p = Params()
        p.add_param('means', ParamWrapper(m))
        p.add_param('quats', ParamWrapper(q))
        p.add_param('scale_ori', ParamWrapper(so))
        p.add_param('opacity', ParamWrapper(opa))
        d = Data()
        d.set('ranges', ranges)
        d.set('centers', centers)
        d.set('target', target)
        op = GS3DVoxelizationOnlyAutograd()
        op.initialize(p, config)
        fwd = op.forward_only(p, d, config)
        return fwd.outputs['loss'].item()

    # Compute autograd gradients
    params = Params()
    params.add_param('means', ParamWrapper(means.clone()))
    params.add_param('quats', ParamWrapper(quats.clone()))
    params.add_param('scale_ori', ParamWrapper(scale_ori.clone()))
    params.add_param('opacity', ParamWrapper(opacity.clone()))
    data_obj = Data()
    data_obj.set('ranges', ranges)
    data_obj.set('centers', centers)
    data_obj.set('target', target)

    ag_op = GS3DVoxelizationOnlyAutograd()
    ag_op.initialize(params, config)
    _, ag_grad = ag_op.forward_backward1(params, data_obj, config)

    ag_v_means = ag_grad.param_grads['means']
    ag_v_quats = ag_grad.param_grads['quats']
    ag_v_scale_ori = ag_grad.param_grads['scale_ori']
    ag_v_opacity = ag_grad.param_grads['opacity']

    print(f"Configuration: N={N}, dim0={dim0}, dim1={dim1}, dim2={dim2}, dtype=float64")

    # Per-element FD for means
    fd_v_means = torch.zeros_like(means)
    for n in range(N):
        for i in range(3):
            m_p = means.clone(); m_p[n, i] += fd_eps
            m_m = means.clone(); m_m[n, i] -= fd_eps
            fd_v_means[n, i] = (
                compute_loss(m_p, quats, scale_ori, opacity)
                - compute_loss(m_m, quats, scale_ori, opacity)
            ) / (2 * fd_eps)

    # Per-element FD for quats
    fd_v_quats = torch.zeros_like(quats)
    for n in range(N):
        for i in range(4):
            q_p = quats.clone(); q_p[n, i] += fd_eps
            q_m = quats.clone(); q_m[n, i] -= fd_eps
            fd_v_quats[n, i] = (
                compute_loss(means, q_p, scale_ori, opacity)
                - compute_loss(means, q_m, scale_ori, opacity)
            ) / (2 * fd_eps)

    # Per-element FD for scale_ori
    fd_v_scale_ori = torch.zeros_like(scale_ori)
    for n in range(N):
        for i in range(3):
            s_p = scale_ori.clone(); s_p[n, i] += fd_eps
            s_m = scale_ori.clone(); s_m[n, i] -= fd_eps
            fd_v_scale_ori[n, i] = (
                compute_loss(means, quats, s_p, opacity)
                - compute_loss(means, quats, s_m, opacity)
            ) / (2 * fd_eps)

    # Per-element FD for opacity
    fd_v_opacity = torch.zeros_like(opacity)
    for n in range(N):
        o_p = opacity.clone(); o_p[n] += fd_eps
        o_m = opacity.clone(); o_m[n] -= fd_eps
        fd_v_opacity[n] = (
            compute_loss(means, quats, scale_ori, o_p)
            - compute_loss(means, quats, scale_ori, o_m)
        ) / (2 * fd_eps)

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
    all_pass &= compare_tensors("d_means     (autograd vs FD)", ag_v_means, fd_v_means)
    all_pass &= compare_tensors("d_quats     (autograd vs FD)", ag_v_quats, fd_v_quats)
    all_pass &= compare_tensors("d_scale_ori (autograd vs FD)", ag_v_scale_ori, fd_v_scale_ori)
    all_pass &= compare_tensors("d_opacity   (autograd vs FD)", ag_v_opacity, fd_v_opacity)

    status = "✓" if all_pass else "✗"
    print(f"\n{status} TEST 5 {'PASSED' if all_pass else 'FAILED'}")

    return all_pass


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("GS3DVoxelizationOnly (Full 4-Stage Pipeline) Test Suite")
    print("=" * 70)

    results = {}

    # Test 1: Forward consistency
    results['forward'] = test_forward_consistency(N=7, dim0=10, dim1=12, dim2=8)

    # Test 2: First-order backward
    results['backward1'] = test_backward1_consistency(N=7, dim0=10, dim1=12, dim2=8)

    # Test 3: Second-order backward
    results['backward2'] = test_backward2_consistency(N=3, dim0=8, dim1=8, dim2=8)

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
