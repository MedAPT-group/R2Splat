"""
Test L2 Loss Functions with JSON Output

This test compares L2LossAutograd vs L2LossCUDA and saves detailed results to JSON.

Author: Dy164
Date: 2026-02-06
"""

import torch
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops import Params, Data, Config
from gsplat2ct.ops.loss import L2LossAutograd, L2LossCUDA


class SimpleParam:
    """Simple parameter wrapper with .data attribute."""
    def __init__(self, data: torch.Tensor):
        self.data = data


def make_params_data(prediction: torch.Tensor, target: torch.Tensor):
    """Create Params and Data objects for L2 loss testing."""
    params = Params()
    params.param_map['prediction'] = SimpleParam(prediction)
    
    data = Data()
    data.data_map['target'] = target
    
    return params, data


def compare_tensors(name: str, autograd_val, cuda_val, rtol=1e-5, atol=1e-6):
    """Compare two tensors and return detailed statistics."""
    if autograd_val is None and cuda_val is None:
        return {
            "name": name,
            "passed": True,
            "both_none": True
        }
    
    if torch.is_tensor(autograd_val):
        autograd_val = autograd_val.detach().cpu()
    if torch.is_tensor(cuda_val):
        cuda_val = cuda_val.detach().cpu()
    
    if autograd_val.shape != cuda_val.shape:
        return {
            "name": name,
            "passed": False,
            "error": "shape_mismatch",
            "autograd_shape": list(autograd_val.shape),
            "cuda_shape": list(cuda_val.shape)
        }
    
    max_abs_diff = (autograd_val - cuda_val).abs().max().item()
    max_rel_diff = 0.0
    nonzero_mask = autograd_val.abs() > 1e-10
    if nonzero_mask.any():
        max_rel_diff = ((autograd_val - cuda_val).abs()[nonzero_mask] / autograd_val.abs()[nonzero_mask]).max().item()
    
    is_close = torch.allclose(autograd_val.float(), cuda_val.float(), rtol=rtol, atol=atol)
    
    # Build stats dict, skip std for scalars (0-D tensors)
    autograd_stats = {
        "min": autograd_val.min().item(),
        "max": autograd_val.max().item(),
        "mean": autograd_val.mean().item()
    }
    cuda_stats = {
        "min": cuda_val.min().item(),
        "max": cuda_val.max().item(),
        "mean": cuda_val.mean().item()
    }
    
    # Only compute std for non-scalar tensors
    if autograd_val.numel() > 1:
        autograd_stats["std"] = autograd_val.std().item()
        cuda_stats["std"] = cuda_val.std().item()
    
    return {
        "name": name,
        "passed": is_close,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "shape": list(autograd_val.shape),
        "rtol": rtol,
        "atol": atol,
        "autograd_stats": autograd_stats,
        "cuda_stats": cuda_stats
    }


def test_forward(prediction, target):
    """Test forward pass."""
    params, data = make_params_data(prediction, target)
    
    # Autograd
    autograd_op = L2LossAutograd()
    autograd_op.initialize(params)
    fwd_autograd = autograd_op.forward_only(params, data)
    
    # CUDA
    cuda_op = L2LossCUDA()
    cuda_op.initialize(params)
    fwd_cuda = cuda_op.forward_only(params, data)
    
    # Compare
    results = []
    results.append(compare_tensors("loss", fwd_autograd.outputs.get("loss"), fwd_cuda.outputs.get("loss")))
    results.append(compare_tensors("l2_map", fwd_autograd.outputs.get("l2_map"), fwd_cuda.outputs.get("l2_map")))
    
    # Manual verification
    manual_l2_map = (prediction - target) ** 2
    manual_loss = manual_l2_map.mean()
    results.append(compare_tensors("loss_vs_manual", fwd_cuda.outputs.get("loss"), manual_loss))
    
    return {
        "phase": "forward",
        "comparisons": results,
        "all_passed": all(r["passed"] for r in results)
    }


def test_backward1(prediction, target):
    """Test first-order backward (gradient)."""
    params, data = make_params_data(prediction, target)
    
    # Autograd
    autograd_op = L2LossAutograd()
    autograd_op.initialize(params)
    fwd_autograd, grad_autograd = autograd_op.forward_backward1(params, data)
    
    # CUDA
    cuda_op = L2LossCUDA()
    cuda_op.initialize(params)
    fwd_cuda, grad_cuda = cuda_op.forward_backward1(params, data)
    
    # Compare
    results = []
    results.append(compare_tensors("loss", fwd_autograd.outputs.get("loss"), fwd_cuda.outputs.get("loss")))
    results.append(compare_tensors("grad_prediction", grad_autograd.param_grads.get("prediction"), grad_cuda.param_grads.get("prediction")))
    
    # Manual gradient verification
    N = prediction.numel()
    manual_grad = 2.0 * (prediction - target) / N
    results.append(compare_tensors("grad_vs_manual", grad_cuda.param_grads.get("prediction"), manual_grad))
    
    # PyTorch autograd verification
    pred_torch = prediction.clone().requires_grad_(True)
    loss_torch = ((pred_torch - target) ** 2).mean()
    loss_torch.backward()
    results.append(compare_tensors("grad_vs_torch_autograd", grad_cuda.param_grads.get("prediction"), pred_torch.grad))
    
    return {
        "phase": "backward1_gradient",
        "comparisons": results,
        "all_passed": all(r["passed"] for r in results)
    }


def test_backward2(prediction, target):
    """Test second-order backward (Hessian)."""
    params, data = make_params_data(prediction, target)
    
    # Autograd
    autograd_op = L2LossAutograd()
    autograd_op.initialize(params)
    fwd_autograd, grad_autograd, hess_autograd = autograd_op.forward_backward2(params, data)
    
    # CUDA
    cuda_op = L2LossCUDA()
    cuda_op.initialize(params)
    fwd_cuda, grad_cuda, hess_cuda = cuda_op.forward_backward2(params, data)
    
    # Compare
    results = []
    results.append(compare_tensors("loss", fwd_autograd.outputs.get("loss"), fwd_cuda.outputs.get("loss")))
    results.append(compare_tensors("grad_prediction", grad_autograd.param_grads.get("prediction"), grad_cuda.param_grads.get("prediction")))
    results.append(compare_tensors("hess_prediction", hess_autograd.param_hess.get("prediction"), hess_cuda.param_hess.get("prediction")))
    
    # Manual Hessian verification (constant 2/N)
    N = prediction.numel()
    manual_hess = torch.full_like(prediction, 2.0 / N)
    results.append(compare_tensors("hess_vs_manual", hess_cuda.param_hess.get("prediction"), manual_hess))
    
    return {
        "phase": "backward2_hessian",
        "comparisons": results,
        "all_passed": all(r["passed"] for r in results)
    }


def run_test_suite(prediction, target, test_name):
    """Run complete test suite for given tensors."""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"Shape: {list(prediction.shape)}, dtype: {prediction.dtype}, device: {prediction.device}")
    print(f"{'='*80}")
    
    results = {
        "test_name": test_name,
        "input_shape": list(prediction.shape),
        "dtype": str(prediction.dtype),
        "device": str(prediction.device),
        "forward": test_forward(prediction, target),
        "backward1": test_backward1(prediction, target),
        "backward2": test_backward2(prediction, target)
    }
    
    # Print summary
    all_passed = (
        results["forward"]["all_passed"] and
        results["backward1"]["all_passed"] and
        results["backward2"]["all_passed"]
    )
    status = "✓ PASS" if all_passed else "✗ FAIL"
    print(f"\n{test_name}: {status}")
    
    results["all_passed"] = all_passed
    return results


def main():
    """Run all tests and save results to JSON."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print(f"L2 Loss Test Suite - JSON Output")
    print(f"Device: {device}")
    print(f"{'='*80}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "output" / "test_output" / f"l2_loss_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        "timestamp": timestamp,
        "device": device,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "tests": []
    }
    
    # Test 1: Batched Images (B, H, W) - reduced size for autograd efficiency
    pred = torch.randn(3, 16, 16, device=device)
    target = torch.randn(3, 16, 16, device=device)
    all_results["tests"].append(run_test_suite(pred, target, "Batched Images (3, 16, 16)"))
    
    # Test 2: Unbatched Volume (D, H, W) - reduced size
    pred = torch.randn(8, 12, 12, device=device)
    target = torch.randn(8, 12, 12, device=device)
    all_results["tests"].append(run_test_suite(pred, target, "Unbatched Volume (8, 12, 12)"))
    
    # Test 3: Single Batch - reduced size
    pred = torch.randn(2, 20, 20, device=device)
    target = torch.randn(2, 20, 20, device=device)
    all_results["tests"].append(run_test_suite(pred, target, "Small Batch (2, 20, 20)"))
    
    # Test 4: Identical Inputs
    pred = torch.randn(2, 12, 12, device=device)
    target = pred.clone()
    all_results["tests"].append(run_test_suite(pred, target, "Identical Inputs (pred == target)"))
    
    # Test 5: Medium Tensor - reduced from large
    pred = torch.randn(4, 32, 32, device=device)
    target = torch.randn(4, 32, 32, device=device)
    all_results["tests"].append(run_test_suite(pred, target, "Medium Tensor (4, 32, 32)"))
    
    # Test 6: Double Precision - reduced size
    pred = torch.randn(3, 16, 16, device=device, dtype=torch.float64)
    target = torch.randn(3, 16, 16, device=device, dtype=torch.float64)
    all_results["tests"].append(run_test_suite(pred, target, "Double Precision (float64)"))
    
    # Summary
    total_tests = len(all_results["tests"])
    passed_tests = sum(1 for t in all_results["tests"] if t["all_passed"])
    
    all_results["summary"] = {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "all_passed": passed_tests == total_tests
    }
    
    # Save to JSON
    json_path = output_dir / "test_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Summary: {passed_tests}/{total_tests} tests passed")
    print(f"Results saved to: {json_path}")
    print(f"{'='*80}\n")
    
    return all_results["summary"]["all_passed"]


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
