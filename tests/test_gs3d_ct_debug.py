"""
Test script for GS3DCTLossCUDA: CUDA vs torch-autograd consistency.

This validates the full training loss pipeline with:
  - Image loss variants: L1+SSIM and L2+SSIM
  - Volume regularization: TV (lambda_tv > 0)

Test stages:
1. Forward consistency
2. First-order backward consistency
3. Second-order backward consistency (gradients + Hessians)

Usage:
    python tests/test_gs3d_ct_debug.py

Author: Dy164
Date: 2026-03-19
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.data import Params, Data, Config
from gsplat2ct.ops.gs3d_ct_train import GS3DCTLossCUDA, GS3DCTLossAutograd


class _ParamWrapper:
    def __init__(self, data):
        self.data = data


def generate_test_data(
    N=47, C=3,
    img_dim0=31, img_dim1=43,
    vol_dim0=13, vol_dim1=12, vol_dim2=11,
    device='cuda', dtype=torch.float32,
    seed=42,
):
    """Generate a complete test dataset."""
    torch.manual_seed(seed)
    device = torch.device(device)

    means = (torch.rand(N, 3, device=device, dtype=dtype) - 0.5) * 4.0

    quats = torch.randn(N, 4, device=device, dtype=dtype)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    quats[quats[:, 0] < 0] *= -1

    scale_ori = torch.randn(N, 3, device=device, dtype=dtype) * 0.3

    opacity = torch.rand(N, device=device, dtype=dtype) * 0.1 - 0.05

    # Camera parameters: simple identity-like views
    viewmats = torch.zeros(C, 4, 4, device=device, dtype=dtype)
    for c in range(C):
        viewmats[c] = torch.eye(4, device=device, dtype=dtype)
        viewmats[c, 2, 3] = 5.0 + c * 2.0

    Ks = torch.zeros(C, 3, 3, device=device, dtype=dtype)
    for c in range(C):
        Ks[c, 0, 0] = 20.0
        Ks[c, 1, 1] = 20.0
        Ks[c, 0, 2] = img_dim1 / 2.0
        Ks[c, 1, 2] = img_dim0 / 2.0
        Ks[c, 2, 2] = 1.0

    targets = torch.rand(C, img_dim0, img_dim1, device=device, dtype=dtype) * 0.1

    vol_ranges = torch.tensor([10.0, 10.0, 10.0], device=device, dtype=dtype)
    vol_centers = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype)

    return (means, quats, scale_ori, opacity,
            viewmats, Ks, targets, vol_ranges, vol_centers)


def make_params_data_config(
    means, quats, scale_ori, opacity,
    viewmats, Ks, targets, vol_ranges, vol_centers,
    img_dim0, img_dim1, vol_dim0, vol_dim1, vol_dim2, image_loss_type,
):
    params = Params()
    params.add_param('means', _ParamWrapper(means))
    params.add_param('quats', _ParamWrapper(quats))
    params.add_param('scale_ori', _ParamWrapper(scale_ori))
    params.add_param('opacity', _ParamWrapper(opacity))

    data_obj = Data()
    data_obj.set('viewmats', viewmats)
    data_obj.set('Ks', Ks)
    data_obj.set('targets', targets)
    data_obj.set('vol_ranges', vol_ranges)
    data_obj.set('vol_centers', vol_centers)
    config = Config()
    config.set('img_dim1', img_dim1)
    config.set('img_dim0', img_dim0)
    config.set('img_tile_size', 7)
    config.set('eps_blur', 0.0001)
    config.set('lambda_l2', 1.0)
    config.set('lambda_ssim', 1.0) # 0.25
    config.set('window_size', 11)
    config.set('vol_dim2', vol_dim2)
    config.set('vol_dim1', vol_dim1)
    config.set('vol_dim0', vol_dim0)
    config.set('vol_tile_size', 8)
    config.set('eps3d', 1e-4)
    config.set('radius_clip', 3.0)
    config.set('lambda_tv', 1.0) # 0.05
    config.set('vol_loss_eps', 1e-4)
    config.set('scale_min', 0.0005)
    config.set('scale_max', 0.5)
    config.set('image_loss_type', image_loss_type)

    return params, data_obj, config


# ============================================================================
# Test Functions
# ============================================================================

def test_forward(tag, N=47, C=3, img_dim0=31, img_dim1=43,
                 vol_dim0=13, vol_dim1=12, vol_dim2=11):
    print("\n" + "=" * 70)
    print(f"TEST 1 ({tag}): Forward Pass Consistency (CUDA vs Autograd)")
    print("=" * 70)

    test_data = generate_test_data(N, C, img_dim0, img_dim1, vol_dim0, vol_dim1, vol_dim2)
    params, data_obj, config = make_params_data_config(
        *test_data, img_dim0, img_dim1, vol_dim0, vol_dim1, vol_dim2, tag.lower()
    )

    cuda_op = GS3DCTLossCUDA()
    cuda_op.initialize(params)
    cuda_fwd = cuda_op.forward_only(params, data_obj, config)

    ag_op = GS3DCTLossAutograd()
    ag_op.initialize(params)
    ag_fwd = ag_op.forward_only(params, data_obj, config)

    loss_cuda = cuda_fwd.outputs['loss'].item()
    loss_ag = ag_fwd.outputs['loss'].item()

    print(f"Configuration: N={N}, C={C}, img=({img_dim0},{img_dim1}), vol=({vol_dim0},{vol_dim1},{vol_dim2})")
    print(f"CUDA loss:     {loss_cuda:.6f}")
    print(f"Autograd loss: {loss_ag:.6f}")
    print(f"Loss diff:     {abs(loss_cuda - loss_ag):.6e}")

    passed = abs(loss_cuda - loss_ag) < 1e-3
    print(f"\n{'✓' if passed else '✗'} TEST 1 {'PASSED' if passed else 'FAILED'}")
    return passed


def test_backward1(tag, N=47, C=3, img_dim0=31, img_dim1=43,
                   vol_dim0=13, vol_dim1=12, vol_dim2=11, rtol=0.1):
    print("\n" + "=" * 70)
    print(f"TEST 2 ({tag}): First-Order Backward Consistency (CUDA vs Autograd)")
    print("=" * 70)

    test_data = generate_test_data(N, C, img_dim0, img_dim1, vol_dim0, vol_dim1, vol_dim2)
    params, data_obj, config = make_params_data_config(
        *test_data, img_dim0, img_dim1, vol_dim0, vol_dim1, vol_dim2, tag.lower()
    )

    cuda_op = GS3DCTLossCUDA()
    cuda_op.initialize(params)
    _, cuda_grad = cuda_op.forward_backward1(params, data_obj, config)

    ag_op = GS3DCTLossAutograd()
    ag_op.initialize(params)
    _, ag_grad = ag_op.forward_backward1(params, data_obj, config)

    print(f"Configuration: N={N}, C={C}")

    def compare(name, t1, t2):
        if t1 is None or t2 is None:
            print(f"  {name}: SKIPPED (None)")
            return True
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = max(t1.abs().max().item(), t2.abs().max().item(), 1e-10)
        rel = max_diff / scale
        ok = rel < rtol
        print(f"  {name}: max_diff={max_diff:.6e}, scale={scale:.6e}, rel={rel:.4f} {'✓' if ok else '✗'}")
        return ok

    ok = True
    for key in ['means', 'quats', 'scale_ori', 'opacity']:
        ok &= compare(f"v_{key}", cuda_grad.param_grads.get(key), ag_grad.param_grads.get(key))

    print(f"\n{'✓' if ok else '✗'} TEST 2 {'PASSED' if ok else 'FAILED'}")
    return ok


def test_backward2(tag, N=47, C=3, img_dim0=31, img_dim1=43,
                   vol_dim0=13, vol_dim1=12, vol_dim2=11, rtol=0.25):
    print("\n" + "=" * 70)
    print(f"TEST 3 ({tag}): Second-Order Backward Consistency (CUDA vs Autograd)")
    print("=" * 70)

    test_data = generate_test_data(N, C, img_dim0, img_dim1, vol_dim0, vol_dim1, vol_dim2)
    params, data_obj, config = make_params_data_config(
        *test_data, img_dim0, img_dim1, vol_dim0, vol_dim1, vol_dim2, tag.lower()
    )

    cuda_op = GS3DCTLossCUDA()
    cuda_op.initialize(params)
    _, cuda_grad, cuda_hess = cuda_op.forward_backward2(params, data_obj, config)

    ag_op = GS3DCTLossAutograd()
    ag_op.initialize(params)
    _, ag_grad, ag_hess = ag_op.forward_backward2(params, data_obj, config)

    print(f"Configuration: N={N}, C={C}")

    def compare(name, t1, t2, tol=rtol):
        if t1 is None or t2 is None:
            print(f"  {name}: SKIPPED (None)")
            return True
        if t1.shape != t2.shape:
            print(f"  {name}: SHAPE MISMATCH {list(t1.shape)} vs {list(t2.shape)} ✗")
            return False
        diff = (t1 - t2).abs()
        max_diff = diff.max().item()
        scale = max(t1.abs().max().item(), t2.abs().max().item(), 1e-10)
        rel = max_diff / scale
        ok = rel < tol
        print(f"  {name}: max_diff={max_diff:.6e}, scale={scale:.6e}, rel={rel:.4f} {'✓' if ok else '✗'}")
        if not ok:
            flat_diff = diff.flatten()
            top_k = min(3, flat_diff.numel())
            top_vals, top_idxs = torch.topk(flat_diff, top_k)
            for i in range(top_k):
                coords = np.unravel_index(top_idxs[i].item(), t1.shape)
                v1 = t1.flatten()[top_idxs[i]].item()
                v2 = t2.flatten()[top_idxs[i]].item()
                print(f"    [{coords}]: ag={v1:.6e}, cuda={v2:.6e}, diff={top_vals[i].item():.6e}")
        return ok

    ok = True

    print("\n--- First-Order Gradients ---")
    for key in ['means', 'quats', 'scale_ori', 'opacity']:
        ok &= compare(f"v_{key}", ag_grad.param_grads.get(key), cuda_grad.param_grads.get(key))

    print("\n--- Self Hessians ---")
    for key in ['means', 'quats', 'scale_ori', 'opacity']:
        ok &= compare(f"h_{key}", ag_hess.param_hess.get(key), cuda_hess.param_hess.get(key))

    print("\n--- Cross-Term Hessians ---")
    for key in ['quats_scale_ori', 'quats_means', 'quats_opacity',
                'scale_ori_means', 'scale_ori_opacity',
                'means_opacity']:
        ag_val = ag_hess.cross_hess.get(key)
        cuda_val = cuda_hess.cross_hess.get(key)
        if ag_val is not None and cuda_val is not None:
            ok &= compare(f"h_{key}", ag_val, cuda_val)
        else:
            print(f"  h_{key}: SKIPPED (not present)")

    print(f"\n{'✓' if ok else '✗'} TEST 3 {'PASSED' if ok else 'FAILED'}")
    return ok


# ============================================================================
# Run All
# ============================================================================

def run_all_tests():
    print("=" * 70)
    print("GS3DCTLoss (Lx+SSIM + TV) Test Suite")
    print("=" * 70)

    variants = ["L1", "L2"]
    results = {}
    for tag in variants:
        results[f"{tag}_forward"] = test_forward(tag)
        results[f"{tag}_backward1"] = test_backward1(tag)
        results[f"{tag}_backward2"] = test_backward2(tag)

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
