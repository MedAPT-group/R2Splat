"""
2D Gaussian Splatting Optimization Visualization Test

This test loads a grayscale target image, initializes 1000 random 2D Gaussians,
and performs damped second-order Newton optimization to fit the image.

Features:
- Load grayscale image and resample to target size
- Initialize 1000 Gaussians with random positions covering the full image
- Perform second-order optimization with damping
- Save rendered images at each iteration
- Generate PDF output with optimization progress

Author: Dy164
Date: 2026-01-30
"""

import torch
import numpy as np
import sys
import os
import math
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available, will use PIL instead")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

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
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, PDF generation will be skipped")

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops import (
    GS2DCUDA,
    Params,
    Data,
)
from gsplat2ct.optim.gs2d import GS2DOptimizerConfig, GS2DSecondOrderOptimizer


def print_separator(title="", char="=", width=80):
    """Print a formatted separator."""
    if title:
        side_len = (width - len(title) - 2) // 2
        print(f"{char * side_len} {title} {char * side_len}")
    else:
        print(char * width)


def load_image(image_path: str) -> torch.Tensor:
    """
    Load grayscale image and convert to normalized float tensor.
    
    Args:
        image_path: Path to image file
        
    Returns:
        torch.Tensor: (H, W) tensor with values in [0, 1]
    """
    print(f"\nLoading image: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if OPENCV_AVAILABLE:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to load image with OpenCV: {image_path}")
    elif PIL_AVAILABLE:
        img = Image.open(image_path).convert('L')
        img = np.array(img, dtype=np.uint8)
    else:
        raise RuntimeError("Neither OpenCV nor PIL available for image loading")
    
    print(f"  Original size: {img.shape}")
    print(f"  Data type: {img.dtype}, Range: [{img.min()}, {img.max()}]")
    
    # Convert to float [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    # Convert to torch tensor (H, W)
    tensor = torch.from_numpy(img_float)
    
    return tensor


def resample_image(image: torch.Tensor, target_size: int = 128) -> torch.Tensor:
    """
    Resample image to target_size (longer side) while maintaining aspect ratio.
    
    Args:
        image: (H, W) tensor
        target_size: Target size for longer side
        
    Returns:
        torch.Tensor: (H', W') resampled tensor
    """
    H, W = image.shape
    
    # Calculate new size maintaining aspect ratio
    if H > W:
        new_h = target_size
        new_w = int(W * target_size / H)
    else:
        new_w = target_size
        new_h = int(H * target_size / W)
    
    print(f"\nResampling to {target_size}px (longer side):")
    print(f"  Original: {H}×{W}")
    print(f"  Resampled: {new_h}×{new_w}")
    
    # Use bilinear interpolation
    resampled = torch.nn.functional.interpolate(
        image.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    
    return resampled


def initialize_gaussians(
    H: int,
    W: int,
    num_gaussians: int,
    target: torch.Tensor,
    device: str = 'cuda',
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize random 2D Gaussians with adaptive parameters based on target image.
    
    Args:
        H: Image height (dim0)
        W: Image width (dim1)
        num_gaussians: Number of Gaussians
        target: (H, W) Target image for adaptive initialization
        device: Device to use
        seed: Random seed
        
    Returns:
        means2d: (1, N, 2) Gaussian centers (x, y) format
        conics: (1, N, 3) Inverse covariance [a, b, c]
        opacities: (1, N) Gaussian opacities
        mus: (1, N) Gaussian mu values
        radii: (1, N) Gaussian radii for tile intersection
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    N = num_gaussians
    
    # Move target to device if needed
    if not target.is_cuda:
        target = target.to(device)
    
    # Random positions: uniformly sample pixel coordinates
    # Sample integer pixel coordinates, then add random offset within pixel
    x_pixels = torch.randint(0, W, (N,), device=device).float()
    y_pixels = torch.randint(0, H, (N,), device=device).float()
    
    # Add random offset [0, 1) for sub-pixel positioning, then +0.5 for pixel center
    x_coords = x_pixels + torch.rand(N, device=device)
    y_coords = y_pixels + torch.rand(N, device=device)
    means2d = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0)  # (1, N, 2)
    
    # Initialize conics: circular Gaussians with random sizes
    # For isotropic Gaussian: conic = [1/σ², 0, 1/σ²]
    # sigma range: 3 to 15 pixels (smaller for better detail)
    sigma_min, sigma_max = 10.0, 30.0
    sigmas = torch.rand(N, device=device) * (sigma_max - sigma_min) + sigma_min
    inv_sigma2 = 1.0 / (sigmas ** 2)
    
    conics = torch.zeros(1, N, 3, device=device)
    conics[0, :, 0] = inv_sigma2  # a
    conics[0, :, 1] = 0.0         # b (no rotation)
    conics[0, :, 2] = inv_sigma2  # c
    
    # Initialize opacities: based on target image intensity at sampled positions
    # opacity = 0.5 * target_intensity (higher initial opacity for better starting point)
    x_idx = x_pixels.long().clamp(0, W - 1)
    y_idx = y_pixels.long().clamp(0, H - 1)
    target_intensities = target[y_idx, x_idx]  # Sample target at Gaussian centers
    opacities = (0.5 * target_intensities + 0.1).unsqueeze(0)  # (1, N), range ~[0.1, 0.6]
    
    # Initialize mus: random values around 0.5
    # mu in range [0.3, 0.7]
    mus = torch.rand(1, N, device=device) * 0.4 + 0.3
        
    # Compute radii for tile intersection (3*sigma cutoff)
    radii = (3.0 * sigmas).int().unsqueeze(0)  # (1, N)
    radii = radii.clamp(min=1, max=max(H, W))
    
    print(f"\nInitialized {num_gaussians} Gaussians (adaptive):")
    print(f"  means2d: {means2d.shape}, range: x=[{x_coords.min():.1f}, {x_coords.max():.1f}], y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")
    print(f"  conics: {conics.shape}, sigma range: [{sigma_min:.1f}, {sigma_max:.1f}]")
    print(f"  opacities: {opacities.shape}, range: [{opacities.min():.3f}, {opacities.max():.3f}] (0.15 × target)")
    print(f"  mus: {mus.shape}, range: [{mus.min():.3f}, {mus.max():.3f}]")
    print(f"  radii: {radii.shape}, range: [{radii.min().item()}, {radii.max().item()}]")
    
    return means2d, conics, opacities, mus, radii


def save_image_tensor(tensor: torch.Tensor, path: str):
    """
    Save tensor as PNG image.
    
    Args:
        tensor: (H, W) or (1, H, W) tensor with values in [0, 1]
        path: Output path
    """
    # Handle different shapes
    if tensor.dim() == 3:
        tensor = tensor[0]
    
    # Convert to numpy
    img_np = tensor.detach().cpu().numpy()
    
    # Clip to [0, 1] and convert to uint8
    img_np = np.clip(img_np, 0, 1) * 255
    img_uint8 = img_np.astype(np.uint8)
    
    # Save using OpenCV (faster) or PIL
    if OPENCV_AVAILABLE:
        cv2.imwrite(path, img_uint8)
    elif PIL_AVAILABLE:
        Image.fromarray(img_uint8, mode='L').save(path)
    else:
        raise RuntimeError("No image saving library available")


def create_param_wrapper(data: torch.Tensor):
    """Create a simple parameter wrapper."""
    class ParamWrapper:
        def __init__(self, data):
            self.data = data
    return ParamWrapper(data)


def optimize_2dgs(
    target: torch.Tensor,
    num_gaussians: int,
    num_iterations: int,
    config: GS2DOptimizerConfig,
    damping_name: str = "default",
    device: str = 'cuda',
    save_dir: Optional[str] = None,
    verbose: bool = True,
    save_every: int = 1,
    param_log_every: int = 10,  # NEW: Log parameter stats every N iterations
) -> Tuple[List[float], List[torch.Tensor], Dict[str, torch.Tensor], List[Dict]]:
    """
    Perform damped second-order Newton optimization on 2D Gaussians.
    
    Args:
        target: (H, W) target image
        num_gaussians: Number of Gaussians
        num_iterations: Number of optimization iterations
        config: GS2DOptimizerConfig
        device: Device to use
        save_dir: Directory to save predictions (optional)
        verbose: Print progress
        save_every: Save every N iterations
        
    Returns:
        Tuple[loss_history, render_history, final_params]
    """
    # Move target to device and add camera dimension
    target = target.to(device)
    target_3d = target.unsqueeze(0)  # (1, H, W)
    H, W = target.shape
    
    # Initialize Gaussians with adaptive parameters based on target
    means2d, conics, opacities, mus, radii = initialize_gaussians(
        H, W, num_gaussians, target, device
    )
    
    # Wrap as parameters
    means2d_param = create_param_wrapper(means2d.clone())
    conics_param = create_param_wrapper(conics.clone())
    opacities_param = create_param_wrapper(opacities.clone())
    mus_param = create_param_wrapper(mus.clone())
    
    # Create GS2D CUDA operator
    gs2d = GS2DCUDA(
        lambda_l1=1.0,
        lambda_ssim=0.25,
        window_size=11,
        tile_size=16,
    )
    
    # Create params and data containers
    params = Params()
    params.add_param('means2d', means2d_param)
    params.add_param('conics', conics_param)
    params.add_param('opacities', opacities_param)
    params.add_param('mus', mus_param)
    
    data = Data()
    data.set('target', target_3d)
    data.set('radii', radii)
    
    # ==== Compute per-parameter learning rates ====
    # Use FIXED learning rates for each parameter type based on their typical scales
    # means2d: positions change by ~1-10 pixels per step
    # conics: small values (~1e-3 to 1e-1), need moderate LR
    # opacities: range [0, 1], moderate LR
    # mus: range [0, 1], moderate LR
    
    # Get base_lr from config
    try:
        base_lr = config.get_lr('means2d', config.get_lr('means2d', 1.0))
    except Exception:
        base_lr = config.get_lr('means2d') if hasattr(config, 'get_lr') else 1.0

    # Fixed multipliers tuned for each parameter type:
    # means2d: positions in pixels, typically 0-200, need small relative LR
    # conics: very small values 1/σ², but CAREFUL - they can become singular easily
    # opacities: 0-1 range, moderate LR
    # mus: 0-1 range, moderate LR
    lr_means2d = base_lr * 1.0       # Base LR for positions
    lr_conics = base_lr * 0.1        # REDUCED: was 100, now 1.0 to prevent singularity
    lr_opacities = base_lr * 0.0     # Moderate LR for opacities
    lr_mus = base_lr * 0.2           # Moderate LR for mus

    # Update optimizer config per-parameter (keep damping as-is)
    try:
        config.set_per_param(
            means2d_lr=lr_means2d,
            conics_lr=lr_conics,
            opacities_lr=lr_opacities,
            mus_lr=lr_mus,
        )
    except Exception:
        # Fallback if method not present: set directly into maps if available
        if hasattr(config, 'lr_map'):
            config.lr_map['means2d'] = lr_means2d
            config.lr_map['conics'] = lr_conics
            config.lr_map['opacities'] = lr_opacities
            config.lr_map['mus'] = lr_mus

    # Create optimizer with conic constraint to prevent singularity
    optimizer = GS2DSecondOrderOptimizer(
        params, config,
        opacity_clip=True,
        mu_min=0.0,
        mu_max=None,
        conic_constraint=True,  # ENABLED: force positive definiteness of conics
    )
    
    loss_history = []
    render_history = []
    param_history = []  # NEW: Parameter statistics history
    
    if verbose:
        print(f"\n  Optimization Progress [{damping_name}]:")
        print(f"  {'Iter':>5} | {'Loss':>12} | {'Δ Loss':>12} | {'Time':>8}")
        print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}")
    
    import time
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Update params container (they may have been modified by optimizer)
        params.param_map['means2d'] = means2d_param
        params.param_map['conics'] = conics_param
        params.param_map['opacities'] = opacities_param
        params.param_map['mus'] = mus_param
        
        # Update radii based on current conics
        # For isotropic: sigma = 1/sqrt(a) where a is conics[..., 0]
        with torch.no_grad():
            a = conics_param.data[..., 0].clamp(min=1e-6)
            c = conics_param.data[..., 2].clamp(min=1e-6)
            sigma = 1.0 / torch.sqrt(torch.minimum(a, c))
            new_radii = (3.0 * sigma).int().clamp(min=1, max=max(H, W))
            data.set('radii', new_radii)
        
        # Forward + Backward2
        try:
            forward_result, grad_result, hess_result = gs2d.forward_backward2(
                params, data, config=None
            )
        except Exception as e:
            print(f"\n  Error at iteration {iteration}: {e}")
            break
        
        loss_val = forward_result.get_loss().item()
        loss_history.append(loss_val)
        
        render_output = forward_result.outputs.get('render_output')
        
        # Save render output
        if iteration % save_every == 0 or iteration == num_iterations - 1:
            render_history.append(render_output[0].clone().detach().cpu())
            
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"render_{iteration:04d}.png")
                save_image_tensor(render_output[0], save_path)
        
        # NEW: Log parameter statistics every param_log_every iterations
        if iteration % param_log_every == 0 or iteration == num_iterations - 1:
            with torch.no_grad():
                conics_data = conics_param.data
                a = conics_data[..., 0]
                b = conics_data[..., 1]
                c = conics_data[..., 2]
                det = a * c - b * b
                
                param_stats = {
                    'iteration': iteration,
                    'loss': loss_val,
                    'means2d': {
                        'min': means2d_param.data.min().item(),
                        'max': means2d_param.data.max().item(),
                        'mean': means2d_param.data.mean().item(),
                    },
                    'conics': {
                        'a_min': a.min().item(),
                        'a_max': a.max().item(),
                        'b_min': b.min().item(),
                        'b_max': b.max().item(),
                        'c_min': c.min().item(),
                        'c_max': c.max().item(),
                        'det_min': det.min().item(),
                        'det_max': det.max().item(),
                        'singular_count': (det <= 0).sum().item(),
                        'nearly_singular_count': (det < 1e-6).sum().item(),
                    },
                    'opacities': {
                        'min': opacities_param.data.min().item(),
                        'max': opacities_param.data.max().item(),
                        'mean': opacities_param.data.mean().item(),
                    },
                    'mus': {
                        'min': mus_param.data.min().item(),
                        'max': mus_param.data.max().item(),
                        'mean': mus_param.data.mean().item(),
                    },
                }
                param_history.append(param_stats)
        
        # Optimization step
        optimizer.add_grad_hess(grad_result, hess_result)
        optimizer.step()
        optimizer.zero_grad()
        
        iter_time = time.time() - iter_start
        
        # Print progress
        if verbose and (iteration % 5 == 0 or iteration == num_iterations - 1):
            delta_loss = loss_history[-2] - loss_val if iteration > 0 else 0
            print(f"  {iteration:>5} | {loss_val:>12.6e} | {delta_loss:>+12.6e} | {iter_time*1000:>6.1f}ms")
    
    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.2f}s ({total_time/num_iterations*1000:.1f}ms/iter)")
    
    # Return final parameters and param history
    final_params = {
        'means2d': means2d_param.data.clone(),
        'conics': conics_param.data.clone(),
        'opacities': opacities_param.data.clone(),
        'mus': mus_param.data.clone(),
    }
    
    return loss_history, render_history, final_params, param_history


def create_three_panel_frame(
    render: torch.Tensor,
    target: torch.Tensor,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """
    Create a three-panel comparison frame: Render | Target | Difference.
    
    Args:
        render: (H, W) rendered image tensor
        target: (H, W) target image tensor
        vmin: Minimum value for visualization range
        vmax: Maximum value for visualization range
        
    Returns:
        np.ndarray: RGB image frame with three panels
    """
    # Convert to numpy
    render_np = render.cpu().numpy() if render.is_cuda else render.numpy()
    target_np = target.cpu().numpy() if target.is_cuda else target.numpy()
    
    # Compute difference
    diff_np = render_np - target_np
    
    # Normalize to [0, 1] for display
    render_norm = np.clip((render_np - vmin) / (vmax - vmin), 0, 1)
    target_norm = np.clip((target_np - vmin) / (vmax - vmin), 0, 1)
    # Difference uses symmetric colormap, map [-range, +range] to [0, 1]
    diff_range = vmax - vmin
    diff_norm = np.clip((diff_np + diff_range) / (2 * diff_range), 0, 1)
    
    # Convert to uint8
    render_uint8 = (render_norm * 255).astype(np.uint8)
    target_uint8 = (target_norm * 255).astype(np.uint8)
    diff_uint8 = (diff_norm * 255).astype(np.uint8)
    
    # Apply colormap to difference (hot colormap)
    if MATPLOTLIB_AVAILABLE:
        import matplotlib.cm as cm
        diff_rgb = (cm.RdBu_r(diff_norm)[:, :, :3] * 255).astype(np.uint8)
    else:
        # Fallback: convert to RGB
        diff_rgb = np.stack([diff_uint8, diff_uint8, diff_uint8], axis=-1)
    
    # Convert grayscale to RGB
    render_rgb = np.stack([render_uint8, render_uint8, render_uint8], axis=-1)
    target_rgb = np.stack([target_uint8, target_uint8, target_uint8], axis=-1)
    
    # Add text labels using PIL or matplotlib
    if PIL_AVAILABLE:
        from PIL import ImageDraw, ImageFont
        
        def add_label(img_rgb, text):
            img_pil = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(img_pil)
            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            # Draw text at top-left with black background
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle([(5, 5), (15 + text_w, 15 + text_h)], fill='black')
            draw.text((10, 10), text, fill='white', font=font)
            return np.array(img_pil)
        
        render_rgb = add_label(render_rgb, "Render")
        target_rgb = add_label(target_rgb, "Target")
        diff_rgb = add_label(diff_rgb, "Difference")
    
    # Concatenate horizontally
    frame = np.concatenate([render_rgb, target_rgb, diff_rgb], axis=1)
    
    return frame


def generate_gif(
    target: torch.Tensor,
    render_history: List[torch.Tensor],
    output_path: str,
    fps: int = 5,
):
    """
    Generate GIF with three-panel comparison (Render | Target | Difference).
    
    Args:
        target: Target image tensor
        render_history: List of rendered images
        output_path: Output GIF path
        fps: Frames per second
    """
    if not IMAGEIO_AVAILABLE:
        print(f"  ⚠ imageio not available, skipping GIF generation")
        return
    
    print(f"\n  Generating GIF: {output_path}")
    
    # Determine value range (0 to 1 for normalized images)
    vmin, vmax = 0.0, 1.0
    
    frames = []
    for render in render_history:
        frame = create_three_panel_frame(render, target, vmin, vmax)
        frames.append(frame)
    
    # Add final frame repeated for better viewing
    for _ in range(3):
        frames.append(frames[-1])
    
    # Write GIF
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"  ✓ GIF saved: {output_path}")
    print(f"    Total frames: {len(frames)}")
    print(f"    Frame rate: {fps} fps")


def generate_pdf(
    target: torch.Tensor,
    render_history: List[torch.Tensor],
    loss_history: List[float],
    output_path: str,
    num_iterations: int,
    save_every: int = 1,
    config_info: Optional[Dict] = None,
):
    """
    Generate PDF with optimization progress.
    
    Args:
        target: Target image tensor
        render_history: List of rendered images
        loss_history: List of loss values
        output_path: Output PDF path
        num_iterations: Total number of iterations
        save_every: Save interval
        config_info: Optional config information
    """
    if not MATPLOTLIB_AVAILABLE:
        print(f"  ⚠ matplotlib not available, skipping PDF generation")
        return
    
    print(f"\n  Generating PDF: {output_path}")
    
    target_np = target.cpu().numpy() if target.dim() == 2 else target[0].cpu().numpy()
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title and summary
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('2D Gaussian Splatting Optimization Results', fontsize=16, fontweight='bold')
        
        # Target image
        axes[0, 0].imshow(target_np, cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title('Target Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # Final render
        final_render = render_history[-1].numpy() if len(render_history) > 0 else target_np
        axes[0, 1].imshow(final_render, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Final Render (Iter {num_iterations})', fontsize=12)
        axes[0, 1].axis('off')
        
        # Loss curve
        axes[1, 0].plot(loss_history, 'b-', linewidth=1.5)
        axes[1, 0].set_xlabel('Iteration', fontsize=10)
        axes[1, 0].set_ylabel('Loss', fontsize=10)
        axes[1, 0].set_title('Loss Curve', fontsize=12)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error image
        if len(render_history) > 0:
            error = np.abs(final_render - target_np)
            im = axes[1, 1].imshow(error, cmap='hot', vmin=0, vmax=0.5)
            axes[1, 1].set_title('Absolute Error', fontsize=12)
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Add config info text
        if config_info:
            info_text = f"Config: lr={config_info.get('lr', 'N/A')}, damping={config_info.get('damping', 'N/A')}\n"
            info_text += f"Gaussians: {config_info.get('num_gaussians', 'N/A')}, Iterations: {num_iterations}\n"
            info_text += f"Initial Loss: {loss_history[0]:.6e}, Final Loss: {loss_history[-1]:.6e}"
            fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        
        # Page 2+: Optimization progress grid
        num_renders = len(render_history)
        renders_per_page = 16  # 4x4 grid
        num_pages = (num_renders + renders_per_page - 1) // renders_per_page
        
        for page_idx in range(num_pages):
            start_idx = page_idx * renders_per_page
            end_idx = min(start_idx + renders_per_page, num_renders)
            renders_on_page = end_idx - start_idx
            
            rows = 4
            cols = 4
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
            fig.suptitle(f'Optimization Progress (Page {page_idx + 1}/{num_pages})', 
                         fontsize=14, fontweight='bold')
            
            for i in range(rows * cols):
                ax = axes[i // cols, i % cols]
                render_idx = start_idx + i
                
                if render_idx < num_renders:
                    render = render_history[render_idx].numpy()
                    ax.imshow(render, cmap='gray', vmin=0, vmax=1)
                    iter_num = render_idx * save_every
                    ax.set_title(f'Iter {iter_num}', fontsize=9)
                
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)
    
    print(f"  ✓ PDF saved: {output_path}")


def main():
    print("\n" * 2)
    print_separator("2D Gaussian Splatting Optimization Visualization", "=")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Please run on a GPU device.")
    
    device = 'cuda'
    print(f"\nDevice: {device}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(
        os.path.dirname(__file__),
        '..',
        'output',
        'test_output',
        f'2dgs_vis_{timestamp}'
    )
    os.makedirs(output_root, exist_ok=True)
    print(f"\nOutput directory: {output_root}")
    
    # ========================================================================
    # Step 1: Load and resample image
    # ========================================================================
    print_separator("Step 1: Load Image")
    
    image_path = os.path.join(
        os.path.dirname(__file__),
        'test_data',
        'test_image_target.jpg'
    )
    
    target = load_image(image_path)
    # Do NOT resample the target image here — keep original resolution so all
    # initialized Gaussians fall within the original image bounds.
    
    H, W = target.shape
    print(f"  Final target shape: ({H}, {W})")
    print(f"  Value range: [{target.min():.4f}, {target.max():.4f}]")
    
    # Save target
    target_path = os.path.join(output_root, 'target.png')
    save_image_tensor(target, target_path)
    print(f"  Saved target to: {target_path}")
    
    # ========================================================================
    # Step 2: Configure optimization
    # ========================================================================
    print_separator("Step 2: Configure Optimization")
    
    num_gaussians = 5000
    num_iterations = 500
    save_every = 5  # Save every 2 iterations
    
    # Test multiple damping configurations
    # Rule: Higher damping -> Higher learning rate (to maintain similar step size)
    # delta = -grad / (damping + |hess|)
    # If damping increases, denominator increases, so lr should increase proportionally
    # Significantly increased learning rates after fixing initialization and per-param LR
    damping_configs = [
        {
            "name": "low_damping",
            "display_name": "Low Damping",
            "damping": 1e-8,  # Very low damping
            "lr": 0.1,        # Higher base LR
            "description": "Low damping (1e-6), base lr (0.1)"
        },
        {
            "name": "medium_damping",
            "display_name": "Medium Damping",
            "damping": 1e-5,  
            "lr": 0.1,        
            "description": "Medium damping (1e-5), base lr (0.5)"
        },
        {
            "name": "high_damping",
            "display_name": "High Damping",
            "damping": 1e-2,  
            "lr": 1.0,        
            "description": "High damping (1e-4), base lr (1.0)"
        },
    ]
    
    print(f"\n  Configuration:")
    print(f"    Number of Gaussians: {num_gaussians}")
    print(f"    Number of iterations: {num_iterations}")
    print(f"    Save every: {save_every} iterations")
    print(f"\n  Testing {len(damping_configs)} damping configurations:")
    for cfg in damping_configs:
        print(f"    {cfg['display_name']}: damping={cfg['damping']:.1e}, lr={cfg['lr']:.1e}")
        print(f"      {cfg['description']}")
    
    all_results = {}
    
    # ========================================================================
    # Step 3: Run optimization for each configuration
    # ========================================================================
    print_separator("Step 3: Run Optimization")
    
    for config_info in damping_configs:
        print(f"\n{'='*80}")
        print(f"Running: {config_info['display_name']}")
        print(f"{'='*80}")
        
        # Create subdirectory for this configuration
        config_dir = os.path.join(output_root, config_info['name'])
        os.makedirs(config_dir, exist_ok=True)
        renders_dir = os.path.join(config_dir, 'renders')
        os.makedirs(renders_dir, exist_ok=True)
        
        # Create optimizer config with uniform parameters
        optim_config = GS2DOptimizerConfig(
            lr=config_info['lr'],
            damping=config_info['damping'],
        )
        
        # Run optimization
        loss_history, render_history, final_params, param_history = optimize_2dgs(
            target,
            num_gaussians=num_gaussians,
            num_iterations=num_iterations,
            config=optim_config,
            damping_name=config_info['display_name'],
            device=device,
            save_dir=renders_dir,
            verbose=True,
            save_every=save_every,
            param_log_every=10,  # Log parameter stats every 10 iterations
        )
        
        # Store results
        all_results[config_info['name']] = {
            'loss_history': loss_history,
            'render_history': render_history,
            'final_params': final_params,
            'param_history': param_history,  # NEW: include param history
            'config': config_info,
        }
    
    # ========================================================================
    # Step 4: Generate output
    # ========================================================================
    print_separator("Step 4: Generate Output")
    
    for config_name, result in all_results.items():
        config_dir = os.path.join(output_root, config_name)
        config_info = result['config']
        loss_history = result['loss_history']
        render_history = result['render_history']
        final_params = result['final_params']
        param_history = result['param_history']  # NEW
        
        print(f"\n  Processing: {config_info['display_name']}")
        
        # Save loss history
        loss_json_path = os.path.join(config_dir, 'loss_history.json')
        with open(loss_json_path, 'w') as f:
            json.dump({
                'loss_history': loss_history,
                'config': {
                    'num_gaussians': num_gaussians,
                    'num_iterations': num_iterations,
                    'damping': config_info['damping'],
                    'lr': config_info['lr'],
                    'image_size': f'{H}x{W}',
                },
            }, f, indent=2)
        print(f"    Saved loss history to: {loss_json_path}")
        
        # NEW: Save parameter history
        param_history_path = os.path.join(config_dir, 'param_history.json')
        with open(param_history_path, 'w') as f:
            json.dump(param_history, f, indent=2)
        print(f"    Saved parameter history to: {param_history_path}")
        
        # Save final parameters as JSON (convert tensors to lists)
        params_path = os.path.join(config_dir, 'final_params.json')
        serializable_params = {
            'means2d': final_params['means2d'].cpu().numpy().tolist(),
            'conics': final_params['conics'].cpu().numpy().tolist(),
            'opacities': final_params['opacities'].cpu().numpy().tolist(),
            'mus': final_params['mus'].cpu().numpy().tolist(),
        }
        with open(params_path, 'w') as f:
            json.dump(serializable_params, f)
        print(f"    Saved final parameters to: {params_path}")

        # Generate GIF
        gif_path = os.path.join(config_dir, 'optimization.gif')
        generate_gif(target, render_history, gif_path, fps=5)
        # PDF generation intentionally skipped per configuration
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("Summary")
    
    print("\n  Comparison Table:")
    print(f"  {'Configuration':<20} | {'Damping':<12} | {'LR':<12} | {'Initial Loss':<14} | {'Final Loss':<14} | {'Reduction':<12}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")
    
    for config_name, result in all_results.items():
        config_info = result['config']
        loss_history = result['loss_history']
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"  {config_info['display_name']:<20} | {config_info['damping']:<12.1e} | {config_info['lr']:<12.1e} | "
              f"{initial_loss:<14.6e} | {final_loss:<14.6e} | {loss_reduction:<12.2f}%")
    
    print(f"\n  Output structure:")
    print(f"    {output_root}/")
    for config_name in all_results.keys():
        print(f"      {config_name}/")
        print(f"        optimization.gif")
        print(f"        loss_history.json")
        print(f"        final_params.json")
        print(f"        renders/")
    
    print_separator()
    print(f"✓ All outputs saved to: {output_root}")
    print_separator()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
