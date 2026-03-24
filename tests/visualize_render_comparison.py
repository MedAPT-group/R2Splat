"""
Visualize saved render comparison data.

Usage:
    python tests/visualize_render_comparison.py output/test_output/render_comparison_TIMESTAMP.pt

Author: Dy164
Date: 2026-01-25
"""

import torch
import numpy as np
import sys
import os


def visualize_comparison(filepath):
    """Load and display render comparison data."""
    print("=" * 70)
    print("Render Comparison Visualization")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading data from: {filepath}")
    data = torch.load(filepath)
    
    render_cuda = data['render_cuda']
    render_torch = data['render_torch']
    diff = data['diff']
    config = data['config']
    
    C = config['C']
    N = config['N']
    H = config['image_height']
    W = config['image_width']
    
    print(f"\nConfiguration:")
    print(f"  Cameras: {C}")
    print(f"  Gaussians per camera: {N}")
    print(f"  Image size: {H} x {W}")
    
    # Statistics
    print(f"\nCUDA output:")
    print(f"  Range: [{render_cuda.min():.6f}, {render_cuda.max():.6f}]")
    print(f"  Mean: {render_cuda.mean():.6f}")
    print(f"  Std: {render_cuda.std():.6f}")
    
    print(f"\nPyTorch output:")
    print(f"  Range: [{render_torch.min():.6f}, {render_torch.max():.6f}]")
    print(f"  Mean: {render_torch.mean():.6f}")
    print(f"  Std: {render_torch.std():.6f}")
    
    print(f"\nDifference:")
    print(f"  Max abs diff: {diff.max():.6e}")
    print(f"  Mean abs diff: {diff.mean():.6e}")
    
    # Relative difference
    nonzero_mask = render_torch.abs() > 1e-6
    if nonzero_mask.any():
        rel_diff = diff[nonzero_mask] / render_torch.abs()[nonzero_mask]
        print(f"  Max rel diff: {rel_diff.max():.6e}")
        print(f"  Mean rel diff: {rel_diff.mean():.6e}")
    
    # Find top differences
    print(f"\nTop 10 largest absolute differences:")
    flat_diff = diff.view(-1)
    top_k = 10
    top_indices = flat_diff.argsort(descending=True)[:top_k]
    
    for i, idx in enumerate(top_indices, 1):
        c = idx // (H * W)
        remainder = idx % (H * W)
        h = remainder // W
        w = remainder % W
        
        cuda_val = render_cuda[c, h, w].item()
        torch_val = render_torch[c, h, w].item()
        diff_val = diff[c, h, w].item()
        
        if abs(torch_val) > 1e-6:
            rel_diff_val = diff_val / abs(torch_val)
            print(f"  {i}. [{c}, {h:2d}, {w:2d}]: CUDA={cuda_val:8.6f}, "
                  f"Torch={torch_val:8.6f}, diff={diff_val:.6e}, rel={rel_diff_val:.2%}")
        else:
            print(f"  {i}. [{c}, {h:2d}, {w:2d}]: CUDA={cuda_val:8.6f}, "
                  f"Torch={torch_val:8.6f}, diff={diff_val:.6e}")
    
    # Per-camera statistics
    print(f"\nPer-camera statistics:")
    for c in range(C):
        diff_c = diff[c]
        print(f"  Camera {c}:")
        print(f"    Max diff: {diff_c.max():.6e}")
        print(f"    Mean diff: {diff_c.mean():.6e}")
        print(f"    Pixels with diff > 1e-4: {(diff_c > 1e-4).sum().item()}/{H*W}")
    
    # Show sample regions
    print(f"\nSample region [camera 0, rows 30-35, cols 30-35]:")
    print("CUDA:")
    print(render_cuda[0, 30:35, 30:35].numpy())
    print("\nPyTorch:")
    print(render_torch[0, 30:35, 30:35].numpy())
    print("\nDifference:")
    print(diff[0, 30:35, 30:35].numpy())
    
    # Try to visualize with matplotlib if available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for c in range(min(C, 2)):
            # CUDA render
            im0 = axes[c, 0].imshow(render_cuda[c].numpy(), cmap='gray')
            axes[c, 0].set_title(f'Camera {c}: CUDA')
            axes[c, 0].axis('off')
            plt.colorbar(im0, ax=axes[c, 0])
            
            # PyTorch render
            im1 = axes[c, 1].imshow(render_torch[c].numpy(), cmap='gray')
            axes[c, 1].set_title(f'Camera {c}: PyTorch')
            axes[c, 1].axis('off')
            plt.colorbar(im1, ax=axes[c, 1])
            
            # Difference
            im2 = axes[c, 2].imshow(diff[c].numpy(), cmap='hot')
            axes[c, 2].set_title(f'Camera {c}: |CUDA - PyTorch|')
            axes[c, 2].axis('off')
            plt.colorbar(im2, ax=axes[c, 2])
        
        plt.tight_layout()
        
        # Save figure
        output_dir = os.path.dirname(filepath)
        filename = os.path.basename(filepath).replace('.pt', '.png')
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved to: {output_path}")
        
        # Try to show (may not work in headless environment)
        # plt.show()
        
    except ImportError:
        print("\nNote: matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"\nWarning: Could not create visualization: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find the latest render_comparison file
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  "output", "test_output")
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.startswith('render_comparison_')]
            if files:
                latest_file = sorted(files)[-1]
                filepath = os.path.join(output_dir, latest_file)
                print(f"Using latest file: {latest_file}")
            else:
                print("No render_comparison files found")
                print(f"Usage: python {sys.argv[0]} <path_to_render_comparison.pt>")
                sys.exit(1)
        else:
            print(f"Usage: python {sys.argv[0]} <path_to_render_comparison.pt>")
            sys.exit(1)
    else:
        filepath = sys.argv[1]
    
    visualize_comparison(filepath)
