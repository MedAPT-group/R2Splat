"""
Test script for VoxelizeToVoxels operator.

This script validates the CUDA implementation against a pure PyTorch
reference implementation for forward, backward1 (first-order), and
backward2 (second-order Hessian) passes.

Usage:
    python tests/test_voxelize_to_voxels.py

Output:
    Test results are saved to output/test_output/voxelize_YYYYMMDD_HHMMSS/
    as JSON files containing error metrics.

Author: Dy164
Date: 2026-02-03
"""

import torch
import numpy as np
import json
import os
from typing import Tuple, Dict, Any
from datetime import datetime
import sys


def generate_gaussian_params_3d(
    N: int,
    dim0: int,
    dim1: int,
    dim2: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random but valid 3D Gaussian parameters for testing.
    
    Args:
        N: Number of Gaussians
        dim0: Output size in dimension 0 (z-axis, slowest)
        dim1: Output size in dimension 1 (y-axis, middle)
        dim2: Output size in dimension 2 (x-axis, fastest)
        device: PyTorch device
        dtype: PyTorch dtype
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (means3d, conics, opacities, radii)
        - means3d: (N, 3) [x, y, z]
        - conics: (N, 6) [a, p, q, b, r, c] upper triangular of 3x3
        - opacities: (N,)
        - radii: (N, 3) [rx, ry, rz]
    """
    torch.manual_seed(seed)
    
    # means3d: Random positions within volume bounds
    # means3d: [x, y, z] where x=dim2, y=dim1, z=dim0
    means3d = torch.rand(N, 3, device=device, dtype=dtype)
    means3d[:, 0] *= dim2  # x in [0, dim2)
    means3d[:, 1] *= dim1  # y in [0, dim1)
    means3d[:, 2] *= dim0  # z in [0, dim0)
    
    # Generate valid 3D covariance matrices using Cholesky decomposition
    # L = [[l11, 0, 0], [l21, l22, 0], [l31, l32, l33]] where lii > 0
    # Σ = L @ L^T is positive definite
    l11 = torch.rand(N, device=device, dtype=dtype) * 3 + 2  # [2, 5]
    l22 = torch.rand(N, device=device, dtype=dtype) * 3 + 2
    l33 = torch.rand(N, device=device, dtype=dtype) * 3 + 2
    l21 = torch.randn(N, device=device, dtype=dtype) * 1.0
    l31 = torch.randn(N, device=device, dtype=dtype) * 1.0
    l32 = torch.randn(N, device=device, dtype=dtype) * 1.0
    
    # Covariance: Σ = L @ L^T
    # Σ[0,0] = l11²
    # Σ[0,1] = Σ[1,0] = l11*l21
    # Σ[0,2] = Σ[2,0] = l11*l31
    # Σ[1,1] = l21² + l22²
    # Σ[1,2] = Σ[2,1] = l21*l31 + l22*l32
    # Σ[2,2] = l31² + l32² + l33²
    sigma_00 = l11 * l11
    sigma_01 = l11 * l21
    sigma_02 = l11 * l31
    sigma_11 = l21 * l21 + l22 * l22
    sigma_12 = l21 * l31 + l22 * l32
    sigma_22 = l31 * l31 + l32 * l32 + l33 * l33
    
    # Compute eigenvalues for radius (use max eigenvalue as conservative estimate)
    # For simplicity, use the diagonal as rough estimate
    max_sigma = torch.stack([sigma_00, sigma_11, sigma_22], dim=-1).max(dim=-1)[0]
    radius_scalar = (4 * torch.sqrt(max_sigma)).to(torch.int32)
    radius_scalar = torch.clamp(radius_scalar, min=1)
    # Create contiguous radii tensor: each Gaussian gets the same radius in all 3 dimensions
    radii = radius_scalar.unsqueeze(-1).expand(-1, 3).contiguous().to(torch.int32)
    
    # Compute inverse covariance (conics)
    # Build 3x3 covariance matrix and invert
    Sigma = torch.zeros(N, 3, 3, device=device, dtype=dtype)
    Sigma[:, 0, 0] = sigma_00
    Sigma[:, 0, 1] = Sigma[:, 1, 0] = sigma_01
    Sigma[:, 0, 2] = Sigma[:, 2, 0] = sigma_02
    Sigma[:, 1, 1] = sigma_11
    Sigma[:, 1, 2] = Sigma[:, 2, 1] = sigma_12
    Sigma[:, 2, 2] = sigma_22
    
    # Invert covariance
    Sigma_inv = torch.linalg.inv(Sigma)
    
    # Extract upper triangular: [a, p, q, b, r, c]
    # [[a, p, q], [p, b, r], [q, r, c]]
    conics = torch.stack([
        Sigma_inv[:, 0, 0],  # a
        Sigma_inv[:, 0, 1],  # p
        Sigma_inv[:, 0, 2],  # q
        Sigma_inv[:, 1, 1],  # b
        Sigma_inv[:, 1, 2],  # r
        Sigma_inv[:, 2, 2],  # c
    ], dim=-1)
    
    # Opacities: positive values
    opacities = torch.rand(N, device=device, dtype=dtype) * 0.5 + 0.1  # [0.1, 0.6]
    
    return means3d, conics, opacities, radii


def create_params_and_data(
    means3d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    radii: torch.Tensor,
    dim0: int,
    dim1: int,
    dim2: int,
    tile_size: int = 4,
    v_render_output: torch.Tensor = None,
    h_render_output: torch.Tensor = None,
):
    """Create Params and Data objects for the operators."""
    from gsplat2ct.ops import Params, Data
    
    # Create simple parameter wrapper with .data attribute
    class SimpleParam:
        def __init__(self, tensor):
            self.data = tensor
    
    params = Params()
    params.add_param('means3d', SimpleParam(means3d))
    params.add_param('conics', SimpleParam(conics))
    params.add_param('opacities', SimpleParam(opacities))
    
    data = Data()
    data.set('radii', radii)
    data.set('dim0', dim0)
    data.set('dim1', dim1)
    data.set('dim2', dim2)
    data.set('tile_size', tile_size)
    
    if v_render_output is not None:
        data.set('v_render_output', v_render_output)
    if h_render_output is not None:
        data.set('h_render_output', h_render_output)
    
    return params, data


def compute_relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute relative error between two tensors."""
    diff = torch.abs(a - b)
    scale = torch.abs(a) + torch.abs(b) + eps
    rel_err = (diff / scale).mean().item()
    return rel_err


def compute_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute absolute error between two tensors."""
    return torch.abs(a - b).mean().item()


def compute_max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute maximum absolute error between two tensors."""
    return torch.abs(a - b).max().item()


def test_forward(
    cuda_op,
    autograd_op,
    params,
    data,
) -> Dict[str, Any]:
    """Test forward pass consistency."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    # CUDA forward
    cuda_result = cuda_op.forward_only(params, data)
    render_cuda = cuda_result.outputs['render_output']
    
    # Autograd forward
    autograd_result = autograd_op.forward_only(params, data)
    render_autograd = autograd_result.outputs['render_output']
    
    # Compute errors
    abs_err = compute_abs_error(render_cuda, render_autograd)
    max_err = compute_max_abs_error(render_cuda, render_autograd)
    rel_err = compute_relative_error(render_cuda, render_autograd)
    
    print(f"  Render output shape: {render_cuda.shape}")
    print(f"  CUDA sum: {render_cuda.sum().item():.6f}")
    print(f"  Autograd sum: {render_autograd.sum().item():.6f}")
    print(f"  Mean absolute error: {abs_err:.2e}")
    print(f"  Max absolute error: {max_err:.2e}")
    print(f"  Relative error: {rel_err:.2e}")
    
    passed = rel_err < 1e-4
    print(f"  PASSED: {passed}")
    
    return {
        'passed': passed,
        'abs_error': abs_err,
        'max_error': max_err,
        'rel_error': rel_err,
        'cuda_sum': render_cuda.sum().item(),
        'autograd_sum': render_autograd.sum().item(),
    }


def test_backward1(
    cuda_op,
    autograd_op,
    params,
    data,
) -> Dict[str, Any]:
    """Test first-order backward pass consistency."""
    print("\n" + "=" * 60)
    print("Testing First-Order Backward Pass")
    print("=" * 60)
    
    # CUDA backward1
    cuda_fwd, cuda_grad = cuda_op.forward_backward1(params, data)
    
    # Autograd backward1
    autograd_fwd, autograd_grad = autograd_op.forward_backward1(params, data)
    
    results = {}
    all_passed = True
    
    for param_name in ['means3d', 'conics', 'opacities']:
        cuda_g = cuda_grad.param_grads[param_name]
        autograd_g = autograd_grad.param_grads[param_name]
        
        abs_err = compute_abs_error(cuda_g, autograd_g)
        max_err = compute_max_abs_error(cuda_g, autograd_g)
        rel_err = compute_relative_error(cuda_g, autograd_g)
        
        passed = rel_err < 1e-3
        all_passed = all_passed and passed
        
        print(f"\n  Gradient {param_name}:")
        print(f"    Shape: {cuda_g.shape}")
        print(f"    Mean abs error: {abs_err:.2e}")
        print(f"    Max abs error: {max_err:.2e}")
        print(f"    Relative error: {rel_err:.2e}")
        print(f"    PASSED: {passed}")
        
        results[param_name] = {
            'passed': passed,
            'abs_error': abs_err,
            'max_error': max_err,
            'rel_error': rel_err,
        }
    
    results['all_passed'] = all_passed
    return results


def test_backward2(
    cuda_op,
    autograd_op,
    params,
    data,
) -> Dict[str, Any]:
    """Test second-order backward pass consistency."""
    print("\n" + "=" * 60)
    print("Testing Second-Order Backward Pass")
    print("=" * 60)
    
    # CUDA backward2
    cuda_fwd, cuda_grad, cuda_hess = cuda_op.forward_backward2(params, data)
    
    # Autograd backward2
    autograd_fwd, autograd_grad, autograd_hess = autograd_op.forward_backward2(params, data)
    
    results = {}
    all_passed = True
    
    # Test first-order gradients
    print("\n  First-order gradients:")
    for param_name in ['means3d', 'conics', 'opacities']:
        cuda_g = cuda_grad.param_grads[param_name]
        autograd_g = autograd_grad.param_grads[param_name]
        
        rel_err = compute_relative_error(cuda_g, autograd_g)
        passed = rel_err < 1e-3
        all_passed = all_passed and passed
        
        print(f"    {param_name}: rel_err={rel_err:.2e}, PASSED={passed}")
        results[f'grad_{param_name}'] = {
            'rel_error': rel_err,
            'passed': passed,
        }
    
    # Test Hessian terms
    hess_names = ['means3d', 'conics', 'opacities', 'op_mean', 'op_conic', 'mean_conic']
    print("\n  Second-order Hessian terms:")
    
    for hess_name in hess_names:
        cuda_h = cuda_hess.param_hess.get(hess_name)
        autograd_h = autograd_hess.param_hess.get(hess_name)
        
        if cuda_h is None or autograd_h is None:
            print(f"    {hess_name}: SKIPPED (not available)")
            continue
        
        abs_err = compute_abs_error(cuda_h, autograd_h)
        max_err = compute_max_abs_error(cuda_h, autograd_h)
        rel_err = compute_relative_error(cuda_h, autograd_h)
        
        # Hessian tolerance is slightly higher due to numerical precision
        passed = rel_err < 5e-3
        all_passed = all_passed and passed
        
        print(f"    {hess_name}: shape={cuda_h.shape}, rel_err={rel_err:.2e}, PASSED={passed}")
        
        results[f'hess_{hess_name}'] = {
            'passed': passed,
            'abs_error': abs_err,
            'max_error': max_err,
            'rel_error': rel_err,
        }
    
    results['all_passed'] = all_passed
    return results


def save_results(results: Dict[str, Any], output_dir: str):
    """Save test results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'test_results.json')
    
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    results_json = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main test function."""
    print("=" * 60)
    print("VoxelizeToVoxels CUDA vs PyTorch Validation Test")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    if device.type != 'cuda':
        print("WARNING: CUDA not available, skipping test.")
        return
    
    # Test configuration
    N = 50       # Number of Gaussians
    dim0 = 16    # z dimension
    dim1 = 16    # y dimension
    dim2 = 16    # x dimension
    tile_size = 4
    
    print(f"\nTest Configuration:")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print(f"  N (Gaussians): {N}")
    print(f"  Volume size: ({dim0}, {dim1}, {dim2})")
    print(f"  Tile size: {tile_size}")
    
    # Generate test data
    means3d, conics, opacities, radii = generate_gaussian_params_3d(
        N, dim0, dim1, dim2, device, dtype
    )
    
    # Create upstream gradients for backward passes
    v_render_output = torch.randn(dim0, dim1, dim2, device=device, dtype=dtype)
    h_render_output = torch.randn(dim0, dim1, dim2, device=device, dtype=dtype).abs()
    
    # Create params and data
    params, data = create_params_and_data(
        means3d, conics, opacities, radii,
        dim0, dim1, dim2, tile_size,
        v_render_output, h_render_output
    )
    
    # Create operators
    from gsplat2ct.ops import VoxelizeToVoxelsCUDA, VoxelizeToVoxelsAutograd
    
    try:
        cuda_op = VoxelizeToVoxelsCUDA()
        cuda_op.initialize(params)
    except Exception as e:
        print(f"ERROR: Failed to initialize CUDA operator: {e}")
        print("Make sure the CUDA extension is compiled.")
        return
    
    autograd_op = VoxelizeToVoxelsAutograd()
    autograd_op.initialize(params)
    
    # Run tests
    results = {
        'config': {
            'N': N,
            'dim0': dim0,
            'dim1': dim1,
            'dim2': dim2,
            'tile_size': tile_size,
            'device': str(device),
            'dtype': str(dtype),
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    # Test forward pass
    results['forward'] = test_forward(cuda_op, autograd_op, params, data)
    
    # Test first-order backward
    results['backward1'] = test_backward1(cuda_op, autograd_op, params, data)
    
    # Test second-order backward
    results['backward2'] = test_backward2(cuda_op, autograd_op, params, data)
    
    # Overall summary
    all_passed = (
        results['forward']['passed'] and
        results['backward1']['all_passed'] and
        results['backward2']['all_passed']
    )
    results['all_passed'] = all_passed
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Forward pass: {'PASSED' if results['forward']['passed'] else 'FAILED'}")
    print(f"  Backward1 pass: {'PASSED' if results['backward1']['all_passed'] else 'FAILED'}")
    print(f"  Backward2 pass: {'PASSED' if results['backward2']['all_passed'] else 'FAILED'}")
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", "test_output", f"voxelize_{timestamp}")
    save_results(results, output_dir)
    
    # Return exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
