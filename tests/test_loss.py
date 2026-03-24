"""
Test Loss Functions with Two-Way Comparison

This test compares two implementations:
1. PyTorch autograd implementation (ground truth)
2. PyTorch manual implementation (formula verification)

Tests:
- ImageLoss: L2 + D-SSIM loss for images
- VolumeLoss: Smooth TV loss for 3D volumes

Author: Dy164
Date: 2025-12-14
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Module-level output directory (created once per run)
_ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
_TEST_TIMESTAMP = None
_OUTPUT_DIR = None


def get_output_dir():
    """Get or create the unified output directory for this test run."""
    global _TEST_TIMESTAMP, _OUTPUT_DIR
    if _OUTPUT_DIR is None:
        _TEST_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
        _OUTPUT_DIR = os.path.join(_ROOT_DIR, 'output', 'test_output', f'loss_{_TEST_TIMESTAMP}')
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
    return _OUTPUT_DIR

from gsplat2ct.ops import (
    ImageLossAutograd,
    ImageLossTorch,
    ImageLossCUDA,
    VolumeLossAutograd,
    VolumeLossTorch,
    Params,
    Data,
    Config,
)


def _is_cuda_available():
    """Check if CUDA extension is available."""
    try:
        from gsplat2ct.cuda._backend import is_cuda_available
        return is_cuda_available()
    except ImportError:
        return False


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


def save_loss_results_to_json(output_dir, method_name, test_type, fwd_result, grad_result, hess_result, config_dict, save_full_data=True):
    """Save forward, gradient, and Hessian results to JSON file.
    
    Args:
        save_full_data: If True, save complete gradient and Hessian arrays for debugging
    """
    results = {
        "method": method_name,
        "test_type": test_type,
        "timestamp": datetime.now().isoformat(),
        "config": config_dict,
        "forward": {},
        "gradients": {},
        "hessians": {}
    }
    
    # Save forward results
    if fwd_result is not None:
        results["forward"]["loss"] = float(fwd_result.get_loss().item())
        for key, value in fwd_result.outputs.items():
            if torch.is_tensor(value):
                results["forward"][key] = float(value.item()) if value.numel() == 1 else {
                    "shape": list(value.shape),
                    "mean": float(value.mean().item()),
                    "std": float(value.std().item()),
                    "norm": float(torch.norm(value).item())
                }
    
    # Save gradients
    if grad_result is not None:
        param_name = 'prediction' if test_type == 'image' else 'volume'
        grad = grad_result.get(param_name)
        if grad is not None:
            grad_stats = {
                "shape": list(grad.shape),
                "norm": float(torch.norm(grad).item()),
                "mean": float(grad.mean().item()),
                "std": float(grad.std().item()),
                "min": float(grad.min().item()),
                "max": float(grad.max().item())
            }
            
            # Save full gradient data for debugging
            if save_full_data:
                grad_stats["full_data"] = tensor_to_list(grad)
                # Also save some sample values for quick inspection
                grad_flat = grad.flatten()
                grad_stats["samples"] = {
                    "first_10": tensor_to_list(grad_flat[:10]),
                    "last_10": tensor_to_list(grad_flat[-10:]),
                    "middle_10": tensor_to_list(grad_flat[len(grad_flat)//2:len(grad_flat)//2+10])
                }
            
            results["gradients"][param_name] = grad_stats
    
    # Save Hessians
    if hess_result is not None:
        param_name = 'prediction' if test_type == 'image' else 'volume'
        hess = hess_result.get(param_name)
        if hess is not None:
            hess_stats = {
                "shape": list(hess.shape),
                "norm": float(torch.norm(hess).item()),
                "mean": float(hess.mean().item()),
                "std": float(hess.std().item()),
                "min": float(hess.min().item()),
                "max": float(hess.max().item())
            }
            
            # Save full Hessian data for debugging
            if save_full_data:
                hess_stats["full_data"] = tensor_to_list(hess)
                # Also save some sample values for quick inspection
                hess_flat = hess.flatten()
                hess_stats["samples"] = {
                    "first_10": tensor_to_list(hess_flat[:10]),
                    "last_10": tensor_to_list(hess_flat[-10:]),
                    "middle_10": tensor_to_list(hess_flat[len(hess_flat)//2:len(hess_flat)//2+10])
                }
            
            results["hessians"][param_name] = hess_stats
        else:
            results["hessians"][param_name] = None
    
    # Write to file
    filename = os.path.join(output_dir, f"{test_type}_loss_{method_name.lower().replace(' ', '_')}_results.json")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved results to: {filename}")
    return filename


def test_image_loss_comparison(output_dir=None):
    """Test ImageLossAutograd vs ImageLossTorch."""
    print_separator("Image Loss Comparison Test")
    
    # Use provided output directory or get the unified one
    if output_dir is None:
        output_dir = get_output_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Set random seed
    torch.manual_seed(456)
    np.random.seed(456)
    
    # Test configuration
    B, H, W = 2, 32, 32
    window_size = 11
    lambda_l2 = 0.8
    lambda_ssim = 0.2
    
    print(f"\nTest Configuration:")
    print(f"  Image shape: (B={B}, H={H}, W={W})")
    print(f"  Window size: {window_size}")
    print(f"  Lambda L2: {lambda_l2}")
    print(f"  Lambda SSIM: {lambda_ssim}")
    print(f"  Number of windows: ({H}-{window_size}+1) × ({W}-{window_size}+1) = {(H-window_size+1) * (W-window_size+1)}")
    
    # Generate test data
    prediction = torch.randn(B, H, W, requires_grad=False)
    target = torch.randn(B, H, W, requires_grad=False)
    
    print(f"\nTest Data:")
    print(f"  Prediction: mean={prediction.mean().item():.4f}, std={prediction.std().item():.4f}")
    print(f"  Target: mean={target.mean().item():.4f}, std={target.std().item():.4f}")
    
    config_dict = {
        "batch_size": B,
        "height": H,
        "width": W,
        "window_size": window_size,
        "lambda_l2": lambda_l2,
        "lambda_ssim": lambda_ssim
    }
    
    # ========================================================================
    # Test 1: PyTorch Autograd Implementation (Ground Truth)
    # ========================================================================
    print_separator("Test 1: ImageLossAutograd")
    
    params_auto = Params()
    params_auto.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
    data_auto = Data()
    data_auto.set('target', target.clone())
    config_auto = Config()
    config_auto.set(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim, window_size=window_size)
    
    autograd_impl = ImageLossAutograd()
    autograd_impl.initialize(params_auto, config_auto)
    
    # Forward only
    fwd_auto_only = autograd_impl.forward_only(params_auto, data_auto, config_auto)
    print(f"  Forward only - Loss: {fwd_auto_only.get_loss().item():.6f}")
    
    # Forward + Backward1
    params_auto_1 = Params()
    params_auto_1.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
    fwd_auto_1, grad_auto_1 = autograd_impl.forward_backward1(params_auto_1, data_auto, config_auto)
    grad_pred_auto_1 = grad_auto_1.get('prediction')
    print(f"  Forward + Backward1:")
    print(f"    Loss: {fwd_auto_1.get_loss().item():.6f}")
    print(f"    Gradient norm: {torch.norm(grad_pred_auto_1).item():.6f}")
    
    # Forward + Backward2
    params_auto_2 = Params()
    params_auto_2.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
    fwd_auto_2, grad_auto_2, hess_auto_2 = autograd_impl.forward_backward2(params_auto_2, data_auto, config_auto)
    grad_pred_auto_2 = grad_auto_2.get('prediction')
    hess_pred_auto_2 = hess_auto_2.get('prediction')
    print(f"  Forward + Backward2:")
    print(f"    Loss: {fwd_auto_2.get_loss().item():.6f}")
    print(f"    Gradient norm: {torch.norm(grad_pred_auto_2).item():.6f}")
    print(f"    Hessian norm: {torch.norm(hess_pred_auto_2).item():.6f}")
    
    # Save results
    save_loss_results_to_json(output_dir, "autograd", "image", fwd_auto_2, grad_auto_2, hess_auto_2, config_dict)
    
    # ========================================================================
    # Test 2: PyTorch Manual Implementation
    # ========================================================================
    print_separator("Test 2: ImageLossTorch")
    
    params_torch = Params()
    params_torch.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
    data_torch = Data()
    data_torch.set('target', target.clone())
    config_torch = Config()
    config_torch.set(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim, window_size=window_size)
    
    torch_impl = ImageLossTorch()
    torch_impl.initialize(params_torch, config_torch)
    
    # Forward only
    fwd_torch_only = torch_impl.forward_only(params_torch, data_torch, config_torch)
    print(f"  Forward only - Loss: {fwd_torch_only.get_loss().item():.6f}")
    
    # Forward + Backward1
    params_torch_1 = Params()
    params_torch_1.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
    fwd_torch_1, grad_torch_1 = torch_impl.forward_backward1(params_torch_1, data_torch, config_torch)
    grad_pred_torch_1 = grad_torch_1.get('prediction')
    print(f"  Forward + Backward1:")
    print(f"    Loss: {fwd_torch_1.get_loss().item():.6f}")
    print(f"    Gradient norm: {torch.norm(grad_pred_torch_1).item():.6f}")
    
    # Forward + Backward2
    params_torch_2 = Params()
    params_torch_2.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
    try:
        fwd_torch_2, grad_torch_2, hess_torch_2 = torch_impl.forward_backward2(params_torch_2, data_torch, config_torch)
        grad_pred_torch_2 = grad_torch_2.get('prediction')
        hess_pred_torch_2 = hess_torch_2.get('prediction')
        print(f"  Forward + Backward2:")
        print(f"    Loss: {fwd_torch_2.get_loss().item():.6f}")
        print(f"    Gradient norm: {torch.norm(grad_pred_torch_2).item():.6f}")
        print(f"    Hessian norm: {torch.norm(hess_pred_torch_2).item():.6f}")
        
        # Save results
        save_loss_results_to_json(output_dir, "torch_manual", "image", fwd_torch_2, grad_torch_2, hess_torch_2, config_dict)
    except Exception as e:
        print(f"  ✗ Torch manual forward_backward2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Test 3: CUDA Implementation (if available)
    # ========================================================================
    cuda_available = _is_cuda_available()
    print_separator("Test 3: ImageLossCUDA")
    
    fwd_cuda_2 = None
    grad_pred_cuda_2 = None
    hess_pred_cuda_2 = None
    
    if cuda_available:
        try:
            # Move data to CUDA
            prediction_cuda = prediction.clone().cuda()
            target_cuda = target.clone().cuda()
            
            params_cuda = Params()
            params_cuda.add_param('prediction', type('Param', (), {'data': prediction_cuda})())
            data_cuda = Data()
            data_cuda.set('target', target_cuda)
            config_cuda = Config()
            config_cuda.set(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim, window_size=window_size)
            
            cuda_impl = ImageLossCUDA(lambda_l2=lambda_l2, lambda_ssim=lambda_ssim, window_size=window_size)
            cuda_impl.initialize(params_cuda, config_cuda)
            
            # Forward only
            fwd_cuda_only = cuda_impl.forward_only(params_cuda, data_cuda, config_cuda)
            print(f"  Forward only - Loss: {fwd_cuda_only.get_loss().item():.6f}")
            
            # Forward + Backward1
            params_cuda_1 = Params()
            params_cuda_1.add_param('prediction', type('Param', (), {'data': prediction_cuda.clone()})())
            fwd_cuda_1, grad_cuda_1 = cuda_impl.forward_backward1(params_cuda_1, data_cuda, config_cuda)
            grad_pred_cuda_1 = grad_cuda_1.get('prediction')
            print(f"  Forward + Backward1:")
            print(f"    Loss: {fwd_cuda_1.get_loss().item():.6f}")
            print(f"    Gradient norm: {torch.norm(grad_pred_cuda_1).item():.6f}")
            
            # Forward + Backward2
            params_cuda_2 = Params()
            params_cuda_2.add_param('prediction', type('Param', (), {'data': prediction_cuda.clone()})())
            fwd_cuda_2, grad_cuda_2, hess_cuda_2 = cuda_impl.forward_backward2(params_cuda_2, data_cuda, config_cuda)
            grad_pred_cuda_2 = grad_cuda_2.get('prediction').cpu()
            hess_pred_cuda_2 = hess_cuda_2.get('prediction').cpu()
            fwd_cuda_loss = fwd_cuda_2.get_loss().cpu()
            print(f"  Forward + Backward2:")
            print(f"    Loss: {fwd_cuda_loss.item():.6f}")
            print(f"    Gradient norm: {torch.norm(grad_pred_cuda_2).item():.6f}")
            print(f"    Hessian norm: {torch.norm(hess_pred_cuda_2).item():.6f}")
            
            # Save results (move to CPU first)
            from gsplat2ct.ops.data import ForwardResult, GradResult, HessResult
            fwd_cuda_cpu = ForwardResult(outputs={
                'loss': fwd_cuda_2.outputs['loss'].cpu(),
            })
            grad_cuda_cpu = GradResult(param_grads={
                'prediction': grad_pred_cuda_2
            })
            hess_cuda_cpu = HessResult(param_hess={
                'prediction': hess_pred_cuda_2
            })
            save_loss_results_to_json(output_dir, "cuda", "image", fwd_cuda_cpu, grad_cuda_cpu, hess_cuda_cpu, config_dict)
            
        except Exception as e:
            print(f"  ✗ CUDA forward_backward2 failed: {e}")
            import traceback
            traceback.print_exc()
            cuda_available = False  # Mark as unavailable for comparison
    else:
        print("  CUDA extension not available - skipping CUDA test")
        print("  Hint: Run 'pip install -e .' from gsplat2ct directory to compile")
    
    # ========================================================================
    # Comparison: Autograd vs Torch Manual
    # ========================================================================
    print_separator("Comparison: Autograd vs Torch Manual")
    
    try:
        # Forward comparison
        loss_diff_fwd = abs(fwd_auto_only.get_loss().item() - fwd_torch_only.get_loss().item())
        print(f"\nForward only comparison:")
        print(f"  Loss difference: {loss_diff_fwd:.6e}")
    except Exception as e:
        print(f"\nForward comparison failed: {e}")
        loss_diff_fwd = float('inf')
    
    try:
        # Backward1 comparison
        loss_diff_1 = abs(fwd_auto_1.get_loss().item() - fwd_torch_1.get_loss().item())
        grad_diff_1 = torch.norm(grad_pred_auto_1 - grad_pred_torch_1)
        grad_rel_diff_1 = grad_diff_1 / (torch.norm(grad_pred_auto_1) + 1e-8)
        print(f"\nForward + Backward1 comparison:")
        print(f"  Loss difference: {loss_diff_1:.6e}")
        print(f"  Gradient difference: abs={grad_diff_1.item():.6e}, rel={grad_rel_diff_1.item():.6e}")
    except Exception as e:
        print(f"\nBackward1 comparison failed: {e}")
        loss_diff_1 = grad_diff_1 = float('inf')
    
    try:
        # Backward2 comparison
        loss_diff_2 = abs(fwd_auto_2.get_loss().item() - fwd_torch_2.get_loss().item())
        grad_diff_2 = torch.norm(grad_pred_auto_2 - grad_pred_torch_2)
        grad_rel_diff_2 = grad_diff_2 / (torch.norm(grad_pred_auto_2) + 1e-8)
        hess_diff_2 = torch.norm(hess_pred_auto_2 - hess_pred_torch_2)
        hess_rel_diff_2 = hess_diff_2 / (torch.norm(hess_pred_auto_2) + 1e-8)
        print(f"\nForward + Backward2 comparison:")
        print(f"  Loss difference: {loss_diff_2:.6e}")
        print(f"  Gradient difference: abs={grad_diff_2.item():.6e}, rel={grad_rel_diff_2.item():.6e}")
        print(f"  Hessian difference: abs={hess_diff_2.item():.6e}, rel={hess_rel_diff_2.item():.6e}")
    except Exception as e:
        print(f"\nBackward2 comparison failed: {e}")
        loss_diff_2 = grad_diff_2 = hess_diff_2 = float('inf')
    
    # Check tolerances
    tolerance_forward = 1e-5
    tolerance_grad = 1e-5
    tolerance_hess = 1e-4
    
    forward_pass = loss_diff_fwd < tolerance_forward and loss_diff_1 < tolerance_forward and loss_diff_2 < tolerance_forward
    grad_pass = grad_diff_1.item() < tolerance_grad and grad_diff_2.item() < tolerance_grad
    hess_pass = hess_diff_2.item() < tolerance_hess
    
    print(f"\nTolerance checks (Torch vs Autograd):")
    print(f"  Forward: {loss_diff_2:.6e} < {tolerance_forward:.1e} {'✓ PASS' if forward_pass else '✗ FAIL'}")
    print(f"  Gradient: {grad_diff_2.item():.6e} < {tolerance_grad:.1e} {'✓ PASS' if grad_pass else '✗ FAIL'}")
    print(f"  Hessian: {hess_diff_2.item():.6e} < {tolerance_hess:.1e} {'✓ PASS' if hess_pass else '✗ FAIL'}")
    
    torch_vs_autograd_pass = forward_pass and grad_pass and hess_pass
    
    # ========================================================================
    # Comparison: CUDA vs Autograd (if CUDA available)
    # ========================================================================
    cuda_vs_autograd_pass = True
    cuda_comparison_results = {}
    
    if cuda_available and grad_pred_cuda_2 is not None:
        print_separator("Comparison: CUDA vs Autograd")
        
        try:
            # Loss comparison
            cuda_loss_val = fwd_cuda_2.get_loss().cpu().item()
            loss_diff_cuda = abs(fwd_auto_2.get_loss().item() - cuda_loss_val)
            print(f"\nLoss difference: {loss_diff_cuda:.6e}")
            
            # Gradient comparison
            grad_diff_cuda = torch.norm(grad_pred_auto_2 - grad_pred_cuda_2)
            grad_rel_diff_cuda = grad_diff_cuda / (torch.norm(grad_pred_auto_2) + 1e-8)
            print(f"Gradient difference: abs={grad_diff_cuda.item():.6e}, rel={grad_rel_diff_cuda.item():.6e}")
            
            # Hessian comparison
            hess_diff_cuda = torch.norm(hess_pred_auto_2 - hess_pred_cuda_2)
            hess_rel_diff_cuda = hess_diff_cuda / (torch.norm(hess_pred_auto_2) + 1e-8)
            print(f"Hessian difference: abs={hess_diff_cuda.item():.6e}, rel={hess_rel_diff_cuda.item():.6e}")
            
            # Check tolerances
            cuda_loss_pass = loss_diff_cuda < tolerance_forward
            cuda_grad_pass = grad_diff_cuda.item() < tolerance_grad
            cuda_hess_pass = hess_diff_cuda.item() < tolerance_hess
            
            print(f"\nTolerance checks (CUDA vs Autograd):")
            print(f"  Forward: {loss_diff_cuda:.6e} < {tolerance_forward:.1e} {'✓ PASS' if cuda_loss_pass else '✗ FAIL'}")
            print(f"  Gradient: {grad_diff_cuda.item():.6e} < {tolerance_grad:.1e} {'✓ PASS' if cuda_grad_pass else '✗ FAIL'}")
            print(f"  Hessian: {hess_diff_cuda.item():.6e} < {tolerance_hess:.1e} {'✓ PASS' if cuda_hess_pass else '✗ FAIL'}")
            
            # All forward, gradient, and Hessian affect pass status
            cuda_vs_autograd_pass = cuda_loss_pass and cuda_grad_pass and cuda_hess_pass
            cuda_comparison_results = {
                "loss_diff": float(loss_diff_cuda),
                "grad_diff": float(grad_diff_cuda.item()),
                "hess_diff": float(hess_diff_cuda.item()),
                "loss_pass": cuda_loss_pass,
                "grad_pass": cuda_grad_pass,
                "hess_pass": cuda_hess_pass,
                "passed": cuda_vs_autograd_pass
            }
            
        except Exception as e:
            print(f"\nCUDA comparison failed: {e}")
            import traceback
            traceback.print_exc()
            cuda_vs_autograd_pass = False
    
    overall_pass = torch_vs_autograd_pass and cuda_vs_autograd_pass
    
    results = {
        "torch_vs_autograd": {
            "forward_pass": forward_pass,
            "grad_pass": grad_pass,
            "hess_pass": hess_pass,
            "loss_diff": float(loss_diff_2) if isinstance(loss_diff_2, (int, float)) else float(loss_diff_2.item()) if hasattr(loss_diff_2, 'item') else loss_diff_2,
            "grad_diff": float(grad_diff_2.item()) if hasattr(grad_diff_2, 'item') else grad_diff_2,
            "hess_diff": float(hess_diff_2.item()) if hasattr(hess_diff_2, 'item') else hess_diff_2,
            "passed": torch_vs_autograd_pass
        },
        "cuda_available": cuda_available,
    }
    
    if cuda_comparison_results:
        results["cuda_vs_autograd"] = cuda_comparison_results
    
    return output_dir, overall_pass, results


def test_volume_loss_comparison(output_dir=None):
    """Test VolumeLossAutograd vs VolumeLossTorch."""
    print_separator("Volume Loss Comparison Test")
    
    # Use provided output directory or get the unified one
    if output_dir is None:
        output_dir = get_output_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Set random seed
    torch.manual_seed(789)
    np.random.seed(789)
    
    # Test 4D volumes only (B, D, H, W) - simplified version
    test_cases = [
        ("4D_B1", torch.randn(1, 16, 16, 16, requires_grad=False)),
        ("4D_B2", torch.randn(2, 16, 16, 16, requires_grad=False))
    ]
    
    all_pass = True
    comparison_results = {}
    
    for case_name, volume in test_cases:
        print_separator(f"Test Case: {case_name} Volume")
        
        print(f"\nTest Configuration:")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Volume mean: {volume.mean().item():.4f}")
        print(f"  Volume std: {volume.std().item():.4f}")
        
        config_dict = {
            "volume_shape": list(volume.shape),
            "is_4d": len(volume.shape) == 4
        }
        
        # ====================================================================
        # Test 1: PyTorch Autograd Implementation (Ground Truth)
        # ====================================================================
        print(f"\n{case_name} - VolumeLossAutograd:")
        
        params_auto = Params()
        params_auto.add_param('volume', type('Param', (), {'data': volume.clone()})())
        data_auto = Data()
        config_auto = Config()
        
        autograd_impl = VolumeLossAutograd()
        autograd_impl.initialize(params_auto, config_auto)
        
        # Forward only
        fwd_auto_only = autograd_impl.forward_only(params_auto, data_auto, config_auto)
        print(f"  Forward only - Loss: {fwd_auto_only.get_loss().item():.6f}")
        
        # Forward + Backward1
        params_auto_1 = Params()
        params_auto_1.add_param('volume', type('Param', (), {'data': volume.clone()})())
        fwd_auto_1, grad_auto_1 = autograd_impl.forward_backward1(params_auto_1, data_auto, config_auto)
        grad_vol_auto_1 = grad_auto_1.get('volume')
        print(f"  Forward + Backward1:")
        print(f"    Loss: {fwd_auto_1.get_loss().item():.6f}")
        print(f"    Gradient norm: {torch.norm(grad_vol_auto_1).item():.6f}")
        
        # Forward + Backward2
        params_auto_2 = Params()
        params_auto_2.add_param('volume', type('Param', (), {'data': volume.clone()})())
        fwd_auto_2, grad_auto_2, hess_auto_2 = autograd_impl.forward_backward2(params_auto_2, data_auto, config_auto)
        grad_vol_auto_2 = grad_auto_2.get('volume')
        hess_vol_auto_2 = hess_auto_2.get('volume')
        print(f"  Forward + Backward2:")
        print(f"    Loss: {fwd_auto_2.get_loss().item():.6f}")
        print(f"    Gradient norm: {torch.norm(grad_vol_auto_2).item():.6f}")
        if hess_vol_auto_2 is not None:
            print(f"    Hessian norm: {torch.norm(hess_vol_auto_2).item():.6f}")
        else:
            print(f"    Hessian: None (unexpected for smooth TV)")
        
        # Save results
        save_loss_results_to_json(output_dir, f"autograd_{case_name.lower()}", "volume", fwd_auto_2, grad_auto_2, hess_auto_2, config_dict)
        
        # ====================================================================
        # Test 2: PyTorch Manual Implementation
        # ====================================================================
        print(f"\n{case_name} - VolumeLossTorch:")
        
        params_torch = Params()
        params_torch.add_param('volume', type('Param', (), {'data': volume.clone()})())
        data_torch = Data()
        config_torch = Config()
        
        torch_impl = VolumeLossTorch()
        torch_impl.initialize(params_torch, config_torch)
        
        # Forward only
        fwd_torch_only = torch_impl.forward_only(params_torch, data_torch, config_torch)
        print(f"  Forward only - Loss: {fwd_torch_only.get_loss().item():.6f}")
        
        # Forward + Backward1
        params_torch_1 = Params()
        params_torch_1.add_param('volume', type('Param', (), {'data': volume.clone()})())
        fwd_torch_1, grad_torch_1 = torch_impl.forward_backward1(params_torch_1, data_torch, config_torch)
        grad_vol_torch_1 = grad_torch_1.get('volume')
        print(f"  Forward + Backward1:")
        print(f"    Loss: {fwd_torch_1.get_loss().item():.6f}")
        print(f"    Gradient norm: {torch.norm(grad_vol_torch_1).item():.6f}")
        
        # Forward + Backward2
        params_torch_2 = Params()
        params_torch_2.add_param('volume', type('Param', (), {'data': volume.clone()})())
        fwd_torch_2, grad_torch_2, hess_torch_2 = torch_impl.forward_backward2(params_torch_2, data_torch, config_torch)
        grad_vol_torch_2 = grad_torch_2.get('volume')
        hess_vol_torch_2 = hess_torch_2.get('volume')
        print(f"  Forward + Backward2:")
        print(f"    Loss: {fwd_torch_2.get_loss().item():.6f}")
        print(f"    Gradient norm: {torch.norm(grad_vol_torch_2).item():.6f}")
        if hess_vol_torch_2 is not None:
            print(f"    Hessian norm: {torch.norm(hess_vol_torch_2).item():.6f}")
        else:
            print(f"    Hessian: None (unexpected for smooth TV)")
        
        # Save results
        save_loss_results_to_json(output_dir, f"torch_manual_{case_name.lower()}", "volume", fwd_torch_2, grad_torch_2, hess_torch_2, config_dict)
        
        # ====================================================================
        # Comparison
        # ====================================================================
        print(f"\n{case_name} - Comparison: Autograd vs Torch Manual")
        
        # Forward comparison
        loss_diff_fwd = abs(fwd_auto_only.get_loss().item() - fwd_torch_only.get_loss().item())
        print(f"  Forward only comparison:")
        print(f"    Loss difference: {loss_diff_fwd:.6e}")
        
        # Backward1 comparison
        loss_diff_1 = abs(fwd_auto_1.get_loss().item() - fwd_torch_1.get_loss().item())
        grad_diff_1 = torch.norm(grad_vol_auto_1 - grad_vol_torch_1)
        grad_rel_diff_1 = grad_diff_1 / (torch.norm(grad_vol_auto_1) + 1e-8)
        print(f"  Forward + Backward1 comparison:")
        print(f"    Loss difference: {loss_diff_1:.6e}")
        print(f"    Gradient difference: abs={grad_diff_1.item():.6e}, rel={grad_rel_diff_1.item():.6e}")
        
        # Backward2 comparison
        loss_diff_2 = abs(fwd_auto_2.get_loss().item() - fwd_torch_2.get_loss().item())
        grad_diff_2 = torch.norm(grad_vol_auto_2 - grad_vol_torch_2)
        grad_rel_diff_2 = grad_diff_2 / (torch.norm(grad_vol_auto_2) + 1e-8)
        print(f"  Forward + Backward2 comparison:")
        print(f"    Loss difference: {loss_diff_2:.6e}")
        print(f"    Gradient difference: abs={grad_diff_2.item():.6e}, rel={grad_rel_diff_2.item():.6e}")
        
        # Hessian comparison (smooth TV has non-trivial Hessian)
        hess_has_auto = hess_vol_auto_2 is not None
        hess_has_torch = hess_vol_torch_2 is not None
        hess_diff_2 = None
        if hess_has_auto and hess_has_torch:
            hess_diff_2 = torch.norm(hess_vol_auto_2 - hess_vol_torch_2)
            hess_rel_diff_2 = hess_diff_2 / (torch.norm(hess_vol_auto_2) + 1e-8)
            print(f"    Hessian difference: abs={hess_diff_2.item():.6e}, rel={hess_rel_diff_2.item():.6e}")
        else:
            print(f"    Hessian present (Autograd): {hess_has_auto}")
            print(f"    Hessian present (Torch): {hess_has_torch}")
        
        # Check tolerances
        tolerance_forward = 1e-5
        tolerance_grad = 1e-5
        tolerance_hess = 1e-4
        
        forward_pass = loss_diff_fwd < tolerance_forward and loss_diff_1 < tolerance_forward and loss_diff_2 < tolerance_forward
        grad_pass = grad_diff_1.item() < tolerance_grad and grad_diff_2.item() < tolerance_grad
        hess_pass = True
        if hess_diff_2 is not None:
            hess_pass = hess_diff_2.item() < tolerance_hess
        else:
            hess_pass = hess_has_auto == hess_has_torch  # both None or both present
        
        print(f"\n  Tolerance checks:")
        print(f"    Forward: {loss_diff_2:.6e} < {tolerance_forward:.1e} {'✓ PASS' if forward_pass else '✗ FAIL'}")
        print(f"    Gradient: {grad_diff_2.item():.6e} < {tolerance_grad:.1e} {'✓ PASS' if grad_pass else '✗ FAIL'}")
        if hess_diff_2 is not None:
            print(f"    Hessian: {hess_diff_2.item():.6e} < {tolerance_hess:.1e} {'✓ PASS' if hess_pass else '✗ FAIL'}")
        else:
            print(f"    Hessian consistency: {'✓ PASS' if hess_pass else '✗ FAIL'}")
        
        case_pass = forward_pass and grad_pass and hess_pass
        all_pass = all_pass and case_pass
        
        comparison_results[case_name] = {
            "forward_pass": forward_pass,
            "grad_pass": grad_pass,
            "hess_pass": hess_pass,
            "loss_diff": float(loss_diff_2),
            "grad_diff": float(grad_diff_2.item()),
            "hess_diff": float(hess_diff_2.item()) if hess_diff_2 is not None else None
        }
    
    return output_dir, all_pass, comparison_results


def test_window_size_variations(output_dir=None):
    """Test ImageLoss with different window sizes."""
    print_separator("Window Size Variation Test")
    
    # Use provided output directory or get the unified one, with subdirectory
    if output_dir is None:
        base_dir = get_output_dir()
    else:
        base_dir = output_dir
    output_dir = os.path.join(base_dir, 'window_size_tests')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Set random seed
    torch.manual_seed(101112)
    
    # Test configuration
    B, H, W = 1, 32, 32
    prediction = torch.randn(B, H, W, requires_grad=False)
    target = torch.randn(B, H, W, requires_grad=False)
    
    # Test different window sizes
    window_sizes = [3, 5, 7, 11, 31, 32]
    
    print(f"\nTesting window sizes: {window_sizes}")
    print(f"Image size: {H}×{W}")
    
    all_pass = True
    results = {}
    
    for ws in window_sizes:
        print(f"\n--- Window size: {ws} ---")
        num_windows = (H - ws + 1) * (W - ws + 1)
        print(f"  Number of windows: ({H}-{ws}+1) × ({W}-{ws}+1) = {num_windows}")
        
        params = Params()
        params.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
        data = Data()
        data.set('target', target.clone())
        config = Config()
        config.set(lambda_l2=0.8, lambda_ssim=0.2, window_size=ws)
        
        # Test with autograd
        autograd_impl = ImageLossAutograd()
        autograd_impl.initialize(params, config)
        fwd_auto, grad_auto, hess_auto = autograd_impl.forward_backward2(params, data, config)
        
        # Test with torch manual
        params_torch = Params()
        params_torch.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
        torch_impl = ImageLossTorch()
        torch_impl.initialize(params_torch, config)
        fwd_torch, grad_torch, hess_torch = torch_impl.forward_backward2(params_torch, data, config)
        
        # Compare
        loss_diff = abs(fwd_auto.get_loss().item() - fwd_torch.get_loss().item())
        grad_diff = torch.norm(grad_auto.get('prediction') - grad_torch.get('prediction'))
        hess_diff = torch.norm(hess_auto.get('prediction') - hess_torch.get('prediction'))
        
        print(f"  Loss diff: {loss_diff:.6e}")
        print(f"  Grad diff: {grad_diff.item():.6e}")
        print(f"  Hess diff: {hess_diff.item():.6e}")
        
        ws_pass = loss_diff < 1e-5 and grad_diff.item() < 1e-5 and hess_diff.item() < 1e-4
        print(f"  Status: {'✓ PASS' if ws_pass else '✗ FAIL'}")
        
        all_pass = all_pass and ws_pass
        results[f"ws_{ws}"] = {
            "num_windows": num_windows,
            "loss_diff": float(loss_diff),
            "grad_diff": float(grad_diff.item()),
            "hess_diff": float(hess_diff.item()),
            "passed": ws_pass
        }
    
    # Save results
    summary_file = os.path.join(output_dir, "window_size_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved window size test results to: {summary_file}")
    
    return all_pass, results


def test_edge_cases():
    """Test edge cases and error handling."""
    print_separator("Edge Cases and Error Handling Test")
    
    print("\n1. Testing invalid window sizes:")
    
    B, H, W = 1, 32, 32
    prediction = torch.randn(B, H, W)
    target = torch.randn(B, H, W)
    
    invalid_cases = [
        (33, "window size > image size"),
        (0, "window size = 0"),
        (-1, "window size < 0"),
    ]
    
    all_pass = True
    
    for ws, description in invalid_cases:
        print(f"\n  Case: {description} (ws={ws})")
        params = Params()
        params.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
        data = Data()
        data.set('target', target.clone())
        config = Config()
        config.set(lambda_l2=0.8, lambda_ssim=0.2, window_size=ws)
        
        try:
            torch_impl = ImageLossTorch()
            torch_impl.initialize(params, config)
            print(f"    ✗ FAIL: Should have raised ValueError")
            all_pass = False
        except ValueError as e:
            print(f"    ✓ PASS: Correctly raised ValueError: {e}")
        except Exception as e:
            print(f"    ✗ FAIL: Unexpected exception: {e}")
            all_pass = False
    
    print("\n2. Testing shape mismatches:")
    
    prediction = torch.randn(B, H, W)
    target_wrong = torch.randn(B, H, W+1)
    
    params = Params()
    params.add_param('prediction', type('Param', (), {'data': prediction.clone()})())
    data = Data()
    data.set('target', target_wrong.clone())
    config = Config()
    config.set(lambda_l2=0.8, lambda_ssim=0.2, window_size=11)
    
    try:
        autograd_impl = ImageLossAutograd()
        autograd_impl.forward_only(params, data, config)
        print(f"  ✗ FAIL: Should have raised ValueError for shape mismatch")
        all_pass = False
    except ValueError as e:
        print(f"  ✓ PASS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"  ✗ FAIL: Unexpected exception: {e}")
        all_pass = False
    
    return all_pass


if __name__ == "__main__":
    print("\n" * 2)
    print("=" * 80)
    print("  Loss Functions Test Suite")
    print("  Testing: Autograd vs Manual Torch Implementation")
    print("=" * 80)
    print()
    
    overall_success = True
    
    # Create unified output directory for all tests
    output_dir = get_output_dir()
    print(f"\nUnified output directory: {output_dir}\n")
    
    # Test 1: Image Loss
    try:
        print_separator("IMAGE LOSS TESTS", "=")
        output_dir_img, img_pass, img_results = test_image_loss_comparison(output_dir)
        print(f"\nImage Loss Test: {'✓ PASS' if img_pass else '✗ FAIL'}")
        overall_success = overall_success and img_pass
    except Exception as e:
        print(f"\n✗ Image Loss Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        overall_success = False
        img_results = None
    
    print("\n" * 2)
    
    # Test 2: Volume Loss
    try:
        print_separator("VOLUME LOSS TESTS", "=")
        output_dir_vol, vol_pass, vol_results = test_volume_loss_comparison(output_dir)
        print(f"\nVolume Loss Test: {'✓ PASS' if vol_pass else '✗ FAIL'}")
        overall_success = overall_success and vol_pass
    except Exception as e:
        print(f"\n✗ Volume Loss Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        overall_success = False
        vol_results = None
    
    print("\n" * 2)
    
    # Test 3: Window Size Variations
    try:
        print_separator("WINDOW SIZE VARIATION TESTS", "=")
        ws_pass, ws_results = test_window_size_variations(output_dir)
        print(f"\nWindow Size Test: {'✓ PASS' if ws_pass else '✗ FAIL'}")
        overall_success = overall_success and ws_pass
    except Exception as e:
        print(f"\n✗ Window Size Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        overall_success = False
        ws_results = None
    
    print("\n" * 2)
    
    # Test 4: Edge Cases
    try:
        print_separator("EDGE CASES TESTS", "=")
        edge_pass = test_edge_cases()
        print(f"\nEdge Cases Test: {'✓ PASS' if edge_pass else '✗ FAIL'}")
        overall_success = overall_success and edge_pass
    except Exception as e:
        print(f"\n✗ Edge Cases Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        overall_success = False
    
    # Final Summary
    print("\n" * 2)
    print_separator("FINAL TEST SUMMARY", "=")
    
    if img_results:
        print("\nImage Loss (Torch vs Autograd):")
        torch_results = img_results.get('torch_vs_autograd', img_results)
        print(f"  Forward: {'✓' if torch_results.get('forward_pass', False) else '✗'}")
        print(f"  Gradient: {'✓' if torch_results.get('grad_pass', False) else '✗'}")
        print(f"  Hessian: {'✓' if torch_results.get('hess_pass', False) else '✗'}")
        
        if img_results.get('cuda_available', False):
            cuda_results = img_results.get('cuda_vs_autograd', {})
            if cuda_results:
                print("\nImage Loss (CUDA vs Autograd):")
                print(f"  Loss: {'✓' if cuda_results.get('loss_pass', False) else '✗'} (diff: {cuda_results.get('loss_diff', 'N/A'):.6e})")
                print(f"  Gradient: {'✓' if cuda_results.get('grad_pass', False) else '✗'} (diff: {cuda_results.get('grad_diff', 'N/A'):.6e})")
                print(f"  Hessian: {'✓' if cuda_results.get('hess_pass', False) else '✗'} (diff: {cuda_results.get('hess_diff', 'N/A'):.6e})")
        else:
            print("\n  (CUDA tests skipped - extension not available)")
    
    if vol_results:
        print("\nVolume Loss:")
        for case_name, case_results in vol_results.items():
            print(f"  {case_name}:")
            print(f"    Forward: {'✓' if case_results['forward_pass'] else '✗'}")
            print(f"    Gradient: {'✓' if case_results['grad_pass'] else '✗'}")
            print(f"    Hessian: {'✓' if case_results['hess_pass'] else '✗'}")
    
    print_separator()
    if overall_success:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print_separator()
    
    sys.exit(0 if overall_success else 1)
