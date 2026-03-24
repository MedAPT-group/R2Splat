"""
Test MLP Classification with Three-Way Comparison

This test compares three implementations:
1. CUDA implementation (MLPClassificationCUDA)
2. PyTorch manual implementation (MLPClassificationTorch)
3. PyTorch autograd implementation (MLPClassificationAutograd)

All three should produce identical results within numerical tolerance.

Author: Dy164
Date: 2025-12-02
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# All imports remain the same - backward compatible!
from gsplat2ct import (
    MLPClassificationCUDA,
    MLPClassificationTorch,
    MLPClassificationAutograd,
    create_mlp_classification,
    ForwardResult,  # Add this for type hints
)
from gsplat2ct.ops.data import Params, Data, Config
from gsplat2ct.cuda._backend import is_cuda_available


def print_separator(title="", char="=", width=80):
    """Print a formatted separator."""
    if title:
        side_len = (width - len(title) - 2) // 2
        print(f"{char * side_len} {title} {char * side_len}")
    else:
        print(char * width)


def tensor_to_list(tensor):
    """Convert tensor to nested list for JSON serialization."""
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy().tolist()
    return tensor


def save_results_to_json(output_dir, method_name, fwd_result, grad_result, hess_result, params):
    """Save forward, gradient, and Hessian results to JSON file."""
    results = {
        "method": method_name,
        "timestamp": datetime.now().isoformat(),
        "forward": {
            "loss": tensor_to_list(fwd_result.get_loss()),
            "predictions": tensor_to_list(fwd_result.outputs['predictions']),
            "predicted_class": int(torch.argmax(fwd_result.outputs['predictions'][0]).item()),
        },
        "gradients": {},
        "hessians": {},
        "parameters": {}
    }
    
    # Save gradients
    for name in ['w1', 'b1', 'w2', 'b2']:
        grad = grad_result.get(name)
        if grad is not None:
            results["gradients"][name] = {
                "shape": list(grad.shape),
                "values": tensor_to_list(grad),
                "norm": float(torch.norm(grad).item())
            }
    
    # Save Hessians
    for name in ['w1', 'b1', 'w2', 'b2']:
        hess = hess_result.get(name)
        if hess is not None:
            hess_data = {
                "shape": list(hess.shape) if hess is not None else None,
            }
            if hess is not None:
                hess_data["values"] = tensor_to_list(hess)
                hess_data["norm"] = float(torch.norm(hess).item())
            if hess.diag is not None:
                hess_data["diagonal"] = {
                    "shape": list(hess.diag.shape),
                    "values": tensor_to_list(hess.diag),
                    "norm": float(torch.norm(hess.diag).item())
                }
            results["hessians"][name] = hess_data
    
    # Save parameters (for reference)
    for name in ['w1', 'b1', 'w2', 'b2']:
        param = getattr(params, name, None)
        if param is not None:
            results["parameters"][name] = {
                "shape": list(param.shape),
                "values": tensor_to_list(param),
                "norm": float(torch.norm(param).item())
            }
    
    # Write to file
    filename = os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_results.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved results to: {filename}")
    return filename


def test_three_way_comparison():
    """Test three-way comparison: CUDA vs Torch vs Autograd."""
    print_separator("MLP Classification Three-Way Comparison Test")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    output_dir = os.path.join(root_dir, 'output', 'test_output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Set random seed for reproducibility
    torch.manual_seed(123)
    np.random.seed(123)
    
    # Network dimensions
    i, m, o = 3, 4, 3  # 3-class classification
    
    print(f"\nNetwork Configuration:")
    print(f"  Input dimension: {i}")
    print(f"  Hidden dimension: {m}")
    print(f"  Output dimension: {o}")
    
    # Generate test data with configurable batch size
    # Test both single sample and batch scenarios
    batch_size = 3  # Changed to 3 to test batch functionality
    x = torch.randn(batch_size, i, requires_grad=False)
    
    # Generate one-hot labels for each sample in batch
    true_classes = [1, 0, 2]  # Different class for each sample
    y = torch.zeros(batch_size, o)
    for b, cls in enumerate(true_classes):
        y[b, cls] = 1.0
    
    print(f"\nTest Data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input shape: {x.shape} (batch_size, i)")
    print(f"  Target shape: {y.shape} (batch_size, o)")
    print(f"  True classes: {true_classes}")
    for b in range(batch_size):
        print(f"  Sample {b} - class: {true_classes[b]}, one-hot: {y[b].tolist()}")
    
    # Initialize parameters (same for all three implementations)
    w1 = torch.randn(m, i, requires_grad=False)
    b1 = torch.randn(m, 1, requires_grad=False)
    w2 = torch.randn(o, m, requires_grad=False)
    b2 = torch.randn(o, 1, requires_grad=False)
    
    # Prepare data structures
    params = Params()
    params.add_param('w1', type('Param', (), {'data': w1.clone()})())
    params.add_param('b1', type('Param', (), {'data': b1.clone()})())
    params.add_param('w2', type('Param', (), {'data': w2.clone()})())
    params.add_param('b2', type('Param', (), {'data': b2.clone()})())
    data = Data()
    data.set(inputs=x, targets=y, task_type='classification')
    config = Config()
    config.set(hessian_type='full')
    
    # ========================================================================
    # Test 1: PyTorch Manual Implementation (Reference)
    # ========================================================================
    print_separator("Test 1: PyTorch Manual Implementation")
    
    torch_impl = MLPClassificationTorch(i, m, o)
    torch_impl.initialize(params, config)
    
    # Use fresh copies of parameters for each test
    params_torch = Params()
    params_torch.add_param('w1', type('Param', (), {'data': w1.clone()})())
    params_torch.add_param('b1', type('Param', (), {'data': b1.clone()})())
    params_torch.add_param('w2', type('Param', (), {'data': w2.clone()})())
    params_torch.add_param('b2', type('Param', (), {'data': b2.clone()})())
    fwd_torch, grad_torch, hess_torch = torch_impl.forward_backward2(params_torch, data, config)
    
    print(f"  Loss (averaged over batch): {fwd_torch.get_loss().item():.6f}")
    print(f"  Predictions shape: {fwd_torch.outputs['predictions'].shape}")
    for b in range(batch_size):
        pred_probs = fwd_torch.outputs['predictions'][b].tolist()
        pred_class = torch.argmax(fwd_torch.outputs['predictions'][b]).item()
        print(f"  Sample {b} - Predicted class: {pred_class}, Probs: {[f'{p:.4f}' for p in pred_probs]}")
    
    # Save results
    save_results_to_json(output_dir, "PyTorch_Manual", fwd_torch, grad_torch, hess_torch, params_torch)
    
    # ========================================================================
    # Test 2: PyTorch Autograd Implementation (Ground Truth)
    # ========================================================================
    print_separator("Test 2: PyTorch Autograd Implementation")
    
    autograd_impl = MLPClassificationAutograd(i, m, o)
    autograd_impl.initialize(params, config)
    
    params_auto = Params()
    params_auto.add_param('w1', type('Param', (), {'data': w1.clone()})())
    params_auto.add_param('b1', type('Param', (), {'data': b1.clone()})())
    params_auto.add_param('w2', type('Param', (), {'data': w2.clone()})())
    params_auto.add_param('b2', type('Param', (), {'data': b2.clone()})())
    fwd_auto, grad_auto, hess_auto = autograd_impl.forward_backward2(params_auto, data, config)
    
    print(f"  Loss (averaged over batch): {fwd_auto.get_loss().item():.6f}")
    print(f"  Predictions shape: {fwd_auto.outputs['predictions'].shape}")
    for b in range(batch_size):
        pred_probs = fwd_auto.outputs['predictions'][b].tolist()
        pred_class = torch.argmax(fwd_auto.outputs['predictions'][b]).item()
        print(f"  Sample {b} - Predicted class: {pred_class}, Probs: {[f'{p:.4f}' for p in pred_probs]}")
    
    # Save results
    save_results_to_json(output_dir, "PyTorch_Autograd", fwd_auto, grad_auto, hess_auto, params_auto)
    
    # ========================================================================
    # Test 3: CUDA Implementation (if available)
    # ========================================================================
    cuda_available = is_cuda_available()
    print_separator("Test 3: CUDA Implementation")
    
    if cuda_available:
        # Move parameters to CUDA
        w1_cuda = w1.clone().cuda()
        b1_cuda = b1.clone().cuda()
        w2_cuda = w2.clone().cuda()
        b2_cuda = b2.clone().cuda()
        x_cuda = x.clone().cuda()
        y_cuda = y.clone().cuda()
        
        params_cuda = Params()
        params_cuda.add_param('w1', type('Param', (), {'data': w1_cuda})())
        params_cuda.add_param('b1', type('Param', (), {'data': b1_cuda})())
        params_cuda.add_param('w2', type('Param', (), {'data': w2_cuda})())
        params_cuda.add_param('b2', type('Param', (), {'data': b2_cuda})())
        data_cuda = Data()
        data_cuda.set(inputs=x_cuda, targets=y_cuda, task_type='classification')
        
        cuda_impl = MLPClassificationCUDA(i, m, o)
        cuda_impl.initialize(params_cuda, config)
        
        fwd_cuda, grad_cuda, hess_cuda = cuda_impl.forward_backward2(params_cuda, data_cuda, config)
        
        print(f"  Loss (averaged over batch): {fwd_cuda.get_loss().item():.6f}")
        print(f"  Predictions shape: {fwd_cuda.outputs['predictions'].shape}")
        for b in range(batch_size):
            pred_probs = fwd_cuda.outputs['predictions'][b].tolist()
            pred_class = torch.argmax(fwd_cuda.outputs['predictions'][b]).item()
            print(f"  Sample {b} - Predicted class: {pred_class}, Probs: {[f'{p:.4f}' for p in pred_probs]}")
        
        # Save results (move to CPU first)
        params_cuda_cpu = Params()
        params_cuda_cpu.add_param('w1', type('Param', (), {'data': params_cuda.w1.cpu()})())
        params_cuda_cpu.add_param('b1', type('Param', (), {'data': params_cuda.b1.cpu()})())
        params_cuda_cpu.add_param('w2', type('Param', (), {'data': params_cuda.w2.cpu()})())
        params_cuda_cpu.add_param('b2', type('Param', (), {'data': params_cuda.b2.cpu()})())
        fwd_cuda_cpu = ForwardResult(outputs={
            'loss': fwd_cuda.outputs['loss'].cpu(),
            'predictions': fwd_cuda.outputs['predictions'].cpu(),
        })
        from gsplat2ct.ops.data import GradResult, HessResult
        grad_cuda_cpu = GradResult(param_grads={
            name: grad_cuda.get(name).cpu() for name in ['w1', 'b1', 'w2', 'b2']
        })
        hess_cuda_cpu = HessResult(param_hess={
            name: hess_cuda.get(name).cpu() if hess_cuda.get(name) is not None else None
            for name in ['w1', 'b1', 'w2', 'b2']
        })
        save_results_to_json(output_dir, "CUDA_Manual", fwd_cuda_cpu, grad_cuda_cpu, hess_cuda_cpu, params_cuda_cpu)
    else:
        print("  CUDA extension not available - skipping CUDA test")
        print("  Hint: Run 'pip install -e .' from gsplat2ct directory to compile")
    
    # ========================================================================
    # Comparison: Torch vs Autograd
    # ========================================================================
    print_separator("Comparison: Torch Manual vs Autograd")
    
    # Loss comparison
    loss_diff_ta = abs(fwd_torch.get_loss().item() - fwd_auto.get_loss().item())
    print(f"\nLoss difference: {loss_diff_ta:.6e}")
    
    # Prediction comparison
    pred_diff_ta = torch.norm(fwd_torch.outputs['predictions'] - fwd_auto.outputs['predictions'])
    print(f"Prediction difference: {pred_diff_ta.item():.6e}")
    
    # Gradient comparison
    print("\nGradient comparison:")
    max_grad_diff_ta = 0
    for name in ['w1', 'b1', 'w2', 'b2']:
        grad_t = grad_torch.get(name)
        grad_a = grad_auto.get(name)
        diff = torch.norm(grad_t - grad_a)
        rel_diff = diff / (torch.norm(grad_t) + 1e-8)
        print(f"  {name}: abs={diff.item():.6e}, rel={rel_diff.item():.6e}")
        max_grad_diff_ta = max(max_grad_diff_ta, diff.item())
    
    # Hessian comparison
    print("\nHessian comparison:")
    max_hess_diff_ta = 0
    for name in ['w1', 'b1', 'w2', 'b2']:
        hess_t = hess_torch.get(name)
        hess_a = hess_auto.get(name)
        diff = torch.norm(hess_t - hess_a)
        rel_diff = diff / (torch.norm(hess_t) + 1e-8)
        print(f"  {name}: abs={diff.item():.6e}, rel={rel_diff.item():.6e}")
        max_hess_diff_ta = max(max_hess_diff_ta, diff.item())
    
    # ========================================================================
    # Batch Hessian Verification: H_batch = avg(H_individual)
    # ========================================================================
    print_separator("Batch Hessian Verification")
    print("\nVerifying that batch Hessian equals average of individual Hessians...")
    print(f"Testing with autograd implementation (ground truth)\n")
    
    # Compute Hessian for each sample individually
    individual_hessians = {name: [] for name in ['w1', 'b1', 'w2', 'b2']}
    
    for b in range(batch_size):
        x_ind = x[b:b+1]  # Keep batch dimension
        y_ind = y[b:b+1]
        data_ind = Data()
        data_ind.set(inputs=x_ind, targets=y_ind, task_type='classification')
        params_ind = Params()
        params_ind.add_param('w1', type('Param', (), {'data': w1.clone()})())
        params_ind.add_param('b1', type('Param', (), {'data': b1.clone()})())
        params_ind.add_param('w2', type('Param', (), {'data': w2.clone()})())
        params_ind.add_param('b2', type('Param', (), {'data': b2.clone()})())
        
        _, _, hess_ind = autograd_impl.forward_backward2(params_ind, data_ind, config)
        
        for name in ['w1', 'b1', 'w2', 'b2']:
            individual_hessians[name].append(hess_ind.get(name))
    
    # Compute average of individual Hessians
    print("Individual sample Hessians computed. Computing averages...\n")
    max_batch_verification_error = 0
    
    for name in ['w1', 'b1', 'w2', 'b2']:
        hess_avg = torch.stack(individual_hessians[name]).mean(dim=0)
        hess_batch = hess_auto.get(name)
        diff = torch.norm(hess_batch - hess_avg)
        rel_diff = diff / (torch.norm(hess_batch) + 1e-8)
        
        print(f"  {name}:")
        print(f"    Batch Hessian norm: {torch.norm(hess_batch).item():.6e}")
        print(f"    Averaged individual Hessian norm: {torch.norm(hess_avg).item():.6e}")
        print(f"    Difference: abs={diff.item():.6e}, rel={rel_diff.item():.6e}")
        
        max_batch_verification_error = max(max_batch_verification_error, diff.item())
    
    batch_verification_passed = max_batch_verification_error < 1e-5
    print(f"\n  Max verification error: {max_batch_verification_error:.6e}")
    print(f"  Status: {'✓ PASS - Batch Hessian = avg(Individual Hessians)' if batch_verification_passed else '✗ FAIL'}")
    
    # ========================================================================
    # Comparison: CUDA vs Autograd (if CUDA available)
    # ========================================================================
    if cuda_available:
        print_separator("Comparison: CUDA vs Autograd")
        
        # Move CUDA results back to CPU for comparison
        fwd_cuda_cpu = ForwardResult(
            outputs={
                'loss': fwd_cuda.outputs['loss'].cpu(),
                'predictions': fwd_cuda.outputs['predictions'].cpu(),
            }
        )
        
        # Loss comparison
        loss_diff_ca = abs(fwd_cuda_cpu.outputs['loss'].item() - fwd_auto.get_loss().item())
        print(f"\nLoss difference: {loss_diff_ca:.6e}")
        
        # Prediction comparison
        pred_diff_ca = torch.norm(fwd_cuda_cpu.outputs['predictions'] - fwd_auto.outputs['predictions'])
        print(f"Prediction difference: {pred_diff_ca.item():.6e}")
        
        # Gradient comparison
        print("\nGradient comparison:")
        max_grad_diff_ca = 0
        for name in ['w1', 'b1', 'w2', 'b2']:
            grad_c = grad_cuda.get(name).cpu()
            grad_a = grad_auto.get(name)
            diff = torch.norm(grad_c - grad_a)
            rel_diff = diff / (torch.norm(grad_a) + 1e-8)
            print(f"  {name}: abs={diff.item():.6e}, rel={rel_diff.item():.6e}")
            max_grad_diff_ca = max(max_grad_diff_ca, diff.item())
        
        # Hessian comparison (now all parameters have full Hessians)
        print("\nHessian comparison:")
        max_hess_diff_ca = 0
        for name in ['w1', 'b1', 'w2', 'b2']:
            hess_c = hess_cuda.get(name).cpu()
            hess_a = hess_auto.get(name)
            diff = torch.norm(hess_c - hess_a)
            rel_diff = diff / (torch.norm(hess_a) + 1e-8)
            print(f"  {name}: abs={diff.item():.6e}, rel={rel_diff.item():.6e}")
            max_hess_diff_ca = max(max_hess_diff_ca, diff.item())
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("Test Summary")
    
    tolerance_grad = 1e-5
    tolerance_hess = 1e-5
    
    print(f"\nTolerance settings:")
    print(f"  Gradient: {tolerance_grad:.1e}")
    print(f"  Hessian: {tolerance_hess:.1e}")
    
    print(f"\nTorch vs Autograd:")
    print(f"  Max gradient diff: {max_grad_diff_ta:.6e} {'✓ PASS' if max_grad_diff_ta < tolerance_grad else '✗ FAIL'}")
    print(f"  Max Hessian diff: {max_hess_diff_ta:.6e} {'✓ PASS' if max_hess_diff_ta < tolerance_hess else '✗ FAIL'}")
    
    if cuda_available:
        print(f"\nCUDA vs Autograd:")
        print(f"  Max gradient diff: {max_grad_diff_ca:.6e} {'✓ PASS' if max_grad_diff_ca < tolerance_grad else '✗ FAIL'}")
        print(f"  Max Hessian diff: {max_hess_diff_ca:.6e} {'✓ PASS' if max_hess_diff_ca < tolerance_hess else '✗ FAIL'}")
    
    # Overall result
    print_separator()
    torch_pass = max_grad_diff_ta < tolerance_grad and max_hess_diff_ta < tolerance_hess
    if cuda_available:
        cuda_pass = max_grad_diff_ca < tolerance_grad and max_hess_diff_ca < tolerance_hess
        overall_pass = torch_pass and cuda_pass
    else:
        overall_pass = torch_pass
    
    if overall_pass:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    
    print_separator()
    
    # Save summary
    summary = {
        "timestamp": timestamp,
        "network_config": {
            "input_dim": i,
            "hidden_dim": m,
            "output_dim": o
        },
        "test_data": {
            "batch_size": batch_size,
            "true_classes": true_classes,
            "input_shape": list(x.shape)
        },
        "batch_hessian_verification": {
            "max_error": float(max_batch_verification_error),
            "passed": batch_verification_passed
        },
        "comparisons": {
            "torch_vs_autograd": {
                "loss_diff": float(loss_diff_ta),
                "pred_diff": float(pred_diff_ta.item()),
                "max_grad_diff": float(max_grad_diff_ta),
                "max_hess_diff": float(max_hess_diff_ta),
                "passed": torch_pass
            }
        },
        "tolerance": {
            "gradient": float(tolerance_grad),
            "hessian": float(tolerance_hess)
        },
        "overall_pass": overall_pass
    }
    
    if cuda_available:
        summary["comparisons"]["cuda_vs_autograd"] = {
            "loss_diff": float(loss_diff_ca),
            "pred_diff": float(pred_diff_ca.item()),
            "max_grad_diff": float(max_grad_diff_ca),
            "max_hess_diff": float(max_hess_diff_ca),
            "passed": cuda_pass
        }
    
    summary_file = os.path.join(output_dir, "test_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"   - pytorch_manual_results.json")
    print(f"   - pytorch_autograd_results.json")
    if cuda_available:
        print(f"   - cuda_manual_results.json")
    print(f"   - test_summary.json")
    
    return overall_pass


def test_factory_function():
    """Test the factory function with different backends."""
    print_separator("Factory Function Test")
    
    i, m, o = 3, 4, 3
    
    # Test auto backend
    print("\n1. Testing 'auto' backend:")
    impl = create_mlp_classification(i, m, o, backend='auto')
    print(f"   Created: {type(impl).__name__}")
    
    # Test torch backend
    print("\n2. Testing 'torch' backend:")
    impl = create_mlp_classification(i, m, o, backend='torch')
    print(f"   Created: {type(impl).__name__}")
    assert isinstance(impl, MLPClassificationTorch)
    
    # Test autograd backend
    print("\n3. Testing 'autograd' backend:")
    impl = create_mlp_classification(i, m, o, backend='autograd')
    print(f"   Created: {type(impl).__name__}")
    assert isinstance(impl, MLPClassificationAutograd)
    
    # Test cuda backend (if available)
    if is_cuda_available():
        print("\n4. Testing 'cuda' backend:")
        impl = create_mlp_classification(i, m, o, backend='cuda')
        print(f"   Created: {type(impl).__name__}")
        assert isinstance(impl, MLPClassificationCUDA)
    else:
        print("\n4. Skipping 'cuda' backend (not available)")
    
    print("\n✓ Factory function tests passed")
    print_separator()


if __name__ == "__main__":
    print("\n" * 2)
    print("=" * 80)
    print("  MLP Classification Test Suite")
    print("  Testing: CUDA vs PyTorch Manual vs PyTorch Autograd")
    print("=" * 80)
    print()
    
    # Run factory function test
    try:
        test_factory_function()
    except Exception as e:
        print(f"\n✗ Factory function test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")
    
    # Run three-way comparison test
    try:
        success = test_three_way_comparison()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
