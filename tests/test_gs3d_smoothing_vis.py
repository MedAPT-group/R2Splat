"""
3D Gaussian Splatting with TV Smoothing Optimization Visualization

This test initializes random 3D Gaussians and performs damped second-order
Newton optimization to minimize the TV smoothing loss.

Features:
- Initialize 3D Gaussians with random positions in a voxel volume
- Perform second-order optimization with damping
- Visualize three orthogonal cross-sections (XY, XZ, YZ planes)
- Generate GIF animation showing optimization progress

Output: 1 row × 3 columns of cross-section views
- Left: XY plane (z = dim0/2) - axial slice
- Middle: XZ plane (y = dim1/2) - coronal slice
- Right: YZ plane (x = dim2/2) - sagittal slice

Author: Dy164
Date: 2026-02-03
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Warning: imageio not available, GIF generation will be skipped")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available")

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops.gs3d_smoothing import (
    GS3DSmoothingCUDA,
    GS3DSmoothingParams, GS3DSmoothingData, GS3DSmoothingConfig
)
from gsplat2ct.ops.data import Params
from gsplat2ct.optim.gs3d_smoothing import (
    GS3DSmoothingOptimizerConfig,
    GS3DSmoothingSecondOrderOptimizer,
)


def print_separator(title="", char="=", width=80):
    """Print a formatted separator."""
    if title:
        side_len = (width - len(title) - 2) // 2
        print(f"{char * side_len} {title} {char * side_len}")
    else:
        print(char * width)


def create_param_wrapper(data: torch.Tensor):
    """Create a simple parameter wrapper."""
    class ParamWrapper:
        def __init__(self, data):
            self.data = data
    return ParamWrapper(data)


def initialize_3d_gaussians(
    dim0: int,
    dim1: int,
    dim2: int,
    num_gaussians: int,
    device: str = 'cuda',
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize random 3D Gaussians for testing.
    
    Args:
        dim0: Volume depth (z)
        dim1: Volume height (y)
        dim2: Volume width (x)
        num_gaussians: Number of Gaussians
        device: Device to use
        seed: Random seed
        
    Returns:
        means3d: (N, 3) Gaussian centers [x, y, z]
        conics: (N, 6) Inverse covariance [a, p, q, b, r, c]
        opacities: (N,) Gaussian opacities
        radii: (N, 3) Gaussian radii [rx, ry, rz]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = num_gaussians
    
    # Random positions within volume bounds (with margin)
    margin = 0.15
    means3d = torch.rand(N, 3, device=device, dtype=torch.float32)
    means3d[:, 0] = means3d[:, 0] * (1 - 2*margin) * dim2 + margin * dim2  # x
    means3d[:, 1] = means3d[:, 1] * (1 - 2*margin) * dim1 + margin * dim1  # y
    means3d[:, 2] = means3d[:, 2] * (1 - 2*margin) * dim0 + margin * dim0  # z
    
    # Generate valid positive definite covariance matrices
    # Use Cholesky decomposition: Sigma = L @ L^T
    sigma_range = (1.5, 4.0)  # Standard deviation range
    
    # Random isotropic + small anisotropy
    sigma_base = torch.rand(N, device=device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    sigma_x = sigma_base * (0.8 + 0.4 * torch.rand(N, device=device))
    sigma_y = sigma_base * (0.8 + 0.4 * torch.rand(N, device=device))
    sigma_z = sigma_base * (0.8 + 0.4 * torch.rand(N, device=device))
    
    # Build diagonal covariance and compute inverse
    # Sigma = diag(σx², σy², σz²) -> Sigma_inv = diag(1/σx², 1/σy², 1/σz²)
    inv_sigma_x = 1.0 / (sigma_x ** 2)
    inv_sigma_y = 1.0 / (sigma_y ** 2)
    inv_sigma_z = 1.0 / (sigma_z ** 2)
    
    # conics: [a, p, q, b, r, c] - upper triangular of 3x3 inverse covariance
    conics = torch.zeros(N, 6, device=device, dtype=torch.float32)
    conics[:, 0] = inv_sigma_x  # a
    conics[:, 1] = 0.0          # p (off-diagonal)
    conics[:, 2] = 0.0          # q (off-diagonal)
    conics[:, 3] = inv_sigma_y  # b
    conics[:, 4] = 0.0          # r (off-diagonal)
    conics[:, 5] = inv_sigma_z  # c
    
    # Random opacities
    opacities = torch.rand(N, device=device, dtype=torch.float32) * 0.6 + 0.2  # [0.2, 0.8]
    
    # Compute radii (4 * max_sigma)
    max_sigma = torch.maximum(sigma_x, torch.maximum(sigma_y, sigma_z))
    radius = (4.0 * max_sigma).int().clamp(min=2, max=max(dim0, dim1, dim2))
    radii = torch.stack([radius, radius, radius], dim=-1)
    
    print(f"\nInitialized {num_gaussians} 3D Gaussians:")
    print(f"  means3d: {means3d.shape}, range: x=[{means3d[:,0].min():.1f}, {means3d[:,0].max():.1f}], "
          f"y=[{means3d[:,1].min():.1f}, {means3d[:,1].max():.1f}], z=[{means3d[:,2].min():.1f}, {means3d[:,2].max():.1f}]")
    print(f"  conics: {conics.shape}, sigma range: [{sigma_range[0]:.1f}, {sigma_range[1]:.1f}]")
    print(f"  opacities: {opacities.shape}, range: [{opacities.min():.3f}, {opacities.max():.3f}]")
    print(f"  radii: {radii.shape}, range: [{radii.min().item()}, {radii.max().item()}]")
    
    return means3d, conics, opacities, radii


def extract_cross_sections(
    volume: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract three orthogonal cross-sections from a 3D volume.
    
    Args:
        volume: (dim0, dim1, dim2) voxel volume
        
    Returns:
        xy_slice: (dim1, dim2) XY plane at z = dim0/2
        xz_slice: (dim0, dim2) XZ plane at y = dim1/2
        yz_slice: (dim0, dim1) YZ plane at x = dim2/2
    """
    volume_np = volume.detach().cpu().numpy()
    dim0, dim1, dim2 = volume_np.shape
    
    # Extract slices at center
    xy_slice = volume_np[dim0 // 2, :, :]  # (dim1, dim2)
    xz_slice = volume_np[:, dim1 // 2, :]  # (dim0, dim2)
    yz_slice = volume_np[:, :, dim2 // 2]  # (dim0, dim1)
    
    return xy_slice, xz_slice, yz_slice


def create_three_panel_frame(
    volume: torch.Tensor,
    iteration: int,
    loss: float,
    frame_height: int = 200,
) -> np.ndarray:
    """
    Create a three-panel frame showing XY, XZ, YZ cross-sections.
    
    Args:
        volume: (dim0, dim1, dim2) rendered volume
        iteration: Current iteration number
        loss: Current loss value
        frame_height: Height of each panel
        
    Returns:
        np.ndarray: RGB image frame (H, 3*W, 3)
    """
    xy_slice, xz_slice, yz_slice = extract_cross_sections(volume)
    
    # Determine value range for consistent coloring
    vmin = 0.0
    vmax = max(xy_slice.max(), xz_slice.max(), yz_slice.max(), 0.1)
    
    def normalize_and_color(arr: np.ndarray) -> np.ndarray:
        """Normalize array and apply colormap."""
        # Normalize to [0, 1]
        arr_norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
        
        if MATPLOTLIB_AVAILABLE:
            # Apply colormap (viridis for density visualization)
            colored = (cm.viridis(arr_norm)[:, :, :3] * 255).astype(np.uint8)
        else:
            # Fallback: grayscale
            arr_uint8 = (arr_norm * 255).astype(np.uint8)
            colored = np.stack([arr_uint8, arr_uint8, arr_uint8], axis=-1)
        
        return colored
    
    def resize_panel(panel: np.ndarray, target_height: int) -> np.ndarray:
        """Resize panel to target height while maintaining aspect ratio."""
        h, w = panel.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        
        if PIL_AVAILABLE:
            img = Image.fromarray(panel)
            img = img.resize((new_w, target_height), Image.BILINEAR)
            return np.array(img)
        else:
            return panel  # Return original if no PIL
    
    # Color each slice
    xy_colored = normalize_and_color(xy_slice)
    xz_colored = normalize_and_color(xz_slice)
    yz_colored = normalize_and_color(yz_slice)
    
    # Resize to consistent height
    xy_resized = resize_panel(xy_colored, frame_height)
    xz_resized = resize_panel(xz_colored, frame_height)
    yz_resized = resize_panel(yz_colored, frame_height)
    
    # Add labels
    if PIL_AVAILABLE:
        def add_label(img_rgb: np.ndarray, text: str, subtitle: str = "") -> np.ndarray:
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except:
                font = ImageFont.load_default()
                font_small = font
            
            # Draw main label at top
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([(3, 3), (10 + text_w, 10 + text_h)], fill='black', outline=None)
            draw.text((6, 6), text, fill='white', font=font)
            
            # Draw subtitle if provided
            if subtitle:
                draw.text((6, 6 + text_h + 4), subtitle, fill='yellow', font=font_small)
            
            return np.array(img_pil)
        
        xy_resized = add_label(xy_resized, "XY (Axial)", f"z={volume.shape[0]//2}")
        xz_resized = add_label(xz_resized, "XZ (Coronal)", f"y={volume.shape[1]//2}")
        yz_resized = add_label(yz_resized, "YZ (Sagittal)", f"x={volume.shape[2]//2}")
    
    # Pad to same width
    max_width = max(xy_resized.shape[1], xz_resized.shape[1], yz_resized.shape[1])
    
    def pad_to_width(panel: np.ndarray, target_width: int) -> np.ndarray:
        h, w = panel.shape[:2]
        if w < target_width:
            pad = np.zeros((h, target_width - w, 3), dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=1)
        return panel
    
    xy_padded = pad_to_width(xy_resized, max_width)
    xz_padded = pad_to_width(xz_resized, max_width)
    yz_padded = pad_to_width(yz_resized, max_width)
    
    # Concatenate horizontally
    frame = np.concatenate([xy_padded, xz_padded, yz_padded], axis=1)
    
    # Add iteration and loss info at the top
    if PIL_AVAILABLE:
        # Create header
        header_height = 30
        header = np.zeros((header_height, frame.shape[1], 3), dtype=np.uint8)
        header[:] = [40, 40, 40]  # Dark gray background
        
        header_pil = Image.fromarray(header)
        draw = ImageDraw.Draw(header_pil)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        info_text = f"Iteration: {iteration}  |  TV Loss: {loss:.6e}"
        draw.text((10, 5), info_text, fill='white', font=font)
        header = np.array(header_pil)
        
        # Combine header and frame
        frame = np.concatenate([header, frame], axis=0)
    
    return frame


def generate_gif(
    volume_history: List[torch.Tensor],
    loss_history: List[float],
    output_path: str,
    fps: int = 5,
):
    """
    Generate GIF with three-panel cross-section views.
    
    Args:
        volume_history: List of rendered volumes
        loss_history: List of loss values
        output_path: Output GIF path
        fps: Frames per second
    """
    if not IMAGEIO_AVAILABLE:
        print(f"  ⚠ imageio not available, skipping GIF generation")
        return
    
    print(f"\n  Generating GIF: {output_path}")
    
    # Check if we have any frames
    if not volume_history or not loss_history:
        print(f"  ⚠ No volume history to generate GIF")
        return
    
    frames = []
    for i, (volume, loss) in enumerate(zip(volume_history, loss_history)):
        frame = create_three_panel_frame(volume, i, loss)
        frames.append(frame)
    
    # Add final frame repeated for better viewing
    if frames:  # Only if we have frames
        for _ in range(5):
            frames.append(frames[-1])
    
    # Write GIF
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"  ✓ GIF saved: {output_path}")
    print(f"    Total frames: {len(frames)}")
    print(f"    Frame rate: {fps} fps")


def optimize_gs3d_smoothing(
    dim0: int,
    dim1: int,
    dim2: int,
    num_gaussians: int,
    num_iterations: int,
    config: GS3DSmoothingOptimizerConfig,
    tile_size: int = 4,
    device: str = 'cuda',
    verbose: bool = True,
    save_every: int = 1,
) -> Tuple[List[float], List[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Perform damped second-order Newton optimization on 3D Gaussians with TV loss.
    
    Args:
        dim0, dim1, dim2: Volume dimensions
        num_gaussians: Number of Gaussians
        num_iterations: Number of optimization iterations
        config: Optimizer configuration
        tile_size: Tile size for voxelization
        device: Device to use
        verbose: Print progress
        save_every: Save every N iterations
        
    Returns:
        loss_history: List of loss values
        volume_history: List of rendered volumes
        final_params: Final parameter tensors
    """
    # Initialize Gaussians
    means3d, conics, opacities, radii = initialize_3d_gaussians(
        dim0, dim1, dim2, num_gaussians, device
    )
    
    # Wrap as parameters
    means3d_param = create_param_wrapper(means3d.clone())
    conics_param = create_param_wrapper(conics.clone())
    opacities_param = create_param_wrapper(opacities.clone())
    
    # Create GS3D Smoothing CUDA operator
    gs3d = GS3DSmoothingCUDA(tv_weight=1.0, tile_size=tile_size)
    
    # Create params container
    params = Params()
    params.add_param('means3d', means3d_param)
    params.add_param('conics', conics_param)
    params.add_param('opacities', opacities_param)
    
    # Create data structure
    data = GS3DSmoothingData(
        radii=radii,
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        tile_size=tile_size,
    )
    
    # Create config
    gs_config = GS3DSmoothingConfig(tile_size=tile_size, tv_weight=1.0)
    
    # Create optimizer
    optimizer = GS3DSmoothingSecondOrderOptimizer(
        params=params,
        config=config,
        opacity_clip=True,
        opacity_min=0.05,
        opacity_max=1.0,
        conic_constraint=True,
    )
    
    loss_history = []
    volume_history = []
    
    if verbose:
        print(f"\n  Optimization Progress:")
        print(f"  {'Iter':>5} | {'Loss':>12} | {'Δ Loss':>12} | {'Time':>8}")
        print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    
    import time
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Update params container
        params.param_map['means3d'] = means3d_param
        params.param_map['conics'] = conics_param
        params.param_map['opacities'] = opacities_param
        
        # Update radii based on current conics
        with torch.no_grad():
            # conics: [a, p, q, b, r, c]
            a = conics_param.data[:, 0].clamp(min=1e-6)
            b = conics_param.data[:, 3].clamp(min=1e-6)
            c = conics_param.data[:, 5].clamp(min=1e-6)
            sigma_max = torch.maximum(1.0 / torch.sqrt(a),
                                       torch.maximum(1.0 / torch.sqrt(b),
                                                     1.0 / torch.sqrt(c)))
            new_radii = (4.0 * sigma_max).int().clamp(min=2, max=max(dim0, dim1, dim2))
            new_radii = torch.stack([new_radii, new_radii, new_radii], dim=-1)
            data.radii = new_radii
        
        # Wrap params for gs3d
        gs3d_params = GS3DSmoothingParams(
            means3d=means3d_param.data,
            conics=conics_param.data,
            opacities=opacities_param.data,
        )
        
        # Forward + Backward2
        try:
            forward_result, grad_result, hess_result = gs3d.forward_backward2(
                gs3d_params, data, gs_config
            )
        except Exception as e:
            print(f"\n  Error at iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            break
        
        loss_val = forward_result.outputs['loss'].item()
        loss_history.append(loss_val)
        
        render_output = forward_result.outputs['render_output']
        
        # Save volume
        if iteration % save_every == 0 or iteration == num_iterations - 1:
            volume_history.append(render_output.clone().detach().cpu())
        
        # Optimization step
        optimizer.add_grad_hess(grad_result, hess_result)
        optimizer.step()
        optimizer.zero_grad()
        
        iter_time = time.time() - iter_start
        
        # Print progress
        if verbose and (iteration % max(1, num_iterations // 20) == 0 or iteration == num_iterations - 1):
            delta_loss = loss_history[-2] - loss_val if iteration > 0 else 0
            print(f"  {iteration:>5} | {loss_val:>12.6e} | {delta_loss:>+12.6e} | {iter_time*1000:>6.1f}ms")
    
    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.2f}s ({total_time/num_iterations*1000:.1f}ms/iter)")
    
    # Return final parameters
    final_params = {
        'means3d': means3d_param.data.clone(),
        'conics': conics_param.data.clone(),
        'opacities': opacities_param.data.clone(),
    }
    
    return loss_history, volume_history, final_params


def run_visualization_test(
    dim0: int = 32,
    dim1: int = 32,
    dim2: int = 32,
    num_gaussians: int = 50,
    num_iterations: int = 100,
    lr: float = 0.5,
    damping: float = 1e-2,
    tile_size: int = 4,
    output_dir: Optional[str] = None,
    device: str = 'cuda',
):
    """
    Run the complete visualization test.
    
    Args:
        dim0, dim1, dim2: Volume dimensions
        num_gaussians: Number of Gaussians
        num_iterations: Number of optimization iterations
        lr: Learning rate
        damping: Damping coefficient
        tile_size: Tile size for voxelization
        output_dir: Output directory for results
        device: Device to use
    """
    print_separator("3D Gaussian Splatting TV Smoothing Visualization Test")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/gs3d_vis_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Volume size: {dim0} × {dim1} × {dim2}")
    print(f"  Number of Gaussians: {num_gaussians}")
    print(f"  Number of iterations: {num_iterations}")
    print(f"  Learning rate: {lr}")
    print(f"  Damping: {damping}")
    print(f"  Tile size: {tile_size}")
    
    # Create optimizer config
    config = GS3DSmoothingOptimizerConfig(lr=lr, damping=damping)
    
    # Set per-parameter learning rates
    # Positions can move more freely, conics need more care
    config.set_per_param(
        means3d_lr=lr * 1.0,
        conics_lr=lr * 0.1,  # More conservative for shape parameters
        opacities_lr=lr * 0.5,
    )
    
    # Run optimization
    print_separator("Optimization")
    loss_history, volume_history, final_params = optimize_gs3d_smoothing(
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        num_gaussians=num_gaussians,
        num_iterations=num_iterations,
        config=config,
        tile_size=tile_size,
        device=device,
        verbose=True,
        save_every=max(1, num_iterations // 50),  # Save ~50 frames
    )
    
    # Save loss history
    loss_path = os.path.join(output_dir, "loss_history.json")
    with open(loss_path, 'w') as f:
        json.dump({
            'loss_history': loss_history,
            'config': {
                'dim0': dim0,
                'dim1': dim1,
                'dim2': dim2,
                'num_gaussians': num_gaussians,
                'num_iterations': num_iterations,
                'lr': lr,
                'damping': damping,
            }
        }, f, indent=2)
    print(f"\n  ✓ Loss history saved: {loss_path}")
    
    # Generate GIF
    print_separator("Generating GIF")
    gif_path = os.path.join(output_dir, "optimization.gif")
    
    # Create matching loss values for volume_history
    save_every = max(1, num_iterations // 50)
    loss_values = [loss_history[min(i * save_every, len(loss_history) - 1)] 
                   for i in range(len(volume_history))]
    
    generate_gif(volume_history, loss_values, gif_path, fps=10)
    
    # Save final frame as PNG
    if len(volume_history) > 0:
        final_frame = create_three_panel_frame(
            volume_history[-1], num_iterations - 1, loss_history[-1]
        )
        final_frame_path = os.path.join(output_dir, "final_frame.png")
        if PIL_AVAILABLE:
            Image.fromarray(final_frame).save(final_frame_path)
            print(f"  ✓ Final frame saved: {final_frame_path}")
    
    # Print summary
    print_separator("Summary")
    print(f"  Initial loss: {loss_history[0]:.6e}")
    print(f"  Final loss: {loss_history[-1]:.6e}")
    print(f"  Loss reduction: {(1 - loss_history[-1]/loss_history[0])*100:.2f}%")
    print(f"  Output files saved to: {output_dir}")
    
    # Generate loss curve plot
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_history, 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('TV Loss', fontsize=12)
        ax.set_title('3D Gaussian TV Smoothing Optimization', fontsize=14)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plot_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Loss curve saved: {plot_path}")
    
    print_separator()
    
    return loss_history, volume_history, final_params


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='3D Gaussian TV Smoothing Visualization')
    parser.add_argument('--dim', type=int, default=32, help='Volume dimension (cubic)')
    parser.add_argument('--num-gaussians', type=int, default=50, help='Number of Gaussians')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate')
    parser.add_argument('--damping', type=float, default=1e-2, help='Damping coefficient')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    run_visualization_test(
        dim0=args.dim,
        dim1=args.dim,
        dim2=args.dim,
        num_gaussians=args.num_gaussians,
        num_iterations=args.iterations,
        lr=args.lr,
        damping=args.damping,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
