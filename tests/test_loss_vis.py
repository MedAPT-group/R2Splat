"""
Loss Optimization Visualization Test

This test loads a grayscale target image, performs damped second-order Newton
optimization on predicted images with different damping levels, and generates
GIF animations showing the optimization process.

Features:
- Load grayscale image and resample to 256px (maintaining aspect ratio)
- Perform optimization with three damping configurations (low, medium, high)
- Save predictions at each iteration
- Generate side-by-side GIF (prediction | target) animation
- Automatically adjust learning rate based on damping level

Author: Dy164
Date: 2025-12-19
"""

import torch
import numpy as np
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict
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

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gsplat2ct.ops import (
    ImageLossCUDA,
    Params,
    Data,
    Config,
)
from gsplat2ct.optim import ImageLossOptimizerConfig, ImageLossSecondOrderOptimizer


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
        torch.Tensor: (1, H, W) tensor with values in [0, 1]
    """
    print(f"\nLoading image: {image_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Try OpenCV first (faster), fall back to PIL
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
    
    # Convert to torch tensor (1, H, W)
    tensor = torch.from_numpy(img_float).unsqueeze(0)
    
    return tensor


def resample_image(image: torch.Tensor, target_size: int = 256) -> torch.Tensor:
    """
    Resample image to target_size (longer side) while maintaining aspect ratio.
    
    Args:
        image: (1, H, W) tensor
        target_size: Target size for longer side
        
    Returns:
        torch.Tensor: (1, H', W') resampled tensor
    """
    B, H, W = image.shape
    
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
        image.unsqueeze(0),  # Add batch dimension for interpolate
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension
    
    return resampled


def save_image_tensor(tensor: torch.Tensor, path: str):
    """
    Save tensor as PNG image.
    
    Args:
        tensor: (1, H, W) tensor with values in [0, 1]
        path: Output path
    """
    # Convert to numpy
    img_np = tensor[0].detach().cpu().numpy()
    
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


def create_four_subplot_frame(
    target: torch.Tensor,
    pred_low: torch.Tensor,
    pred_med: torch.Tensor,
    pred_high: torch.Tensor,
    iteration: int,
    num_iterations: int
) -> np.ndarray:
    """
    Create a 4-subplot figure: Target | Low | Medium | High damping with iteration counter.
    
    Args:
        target: (1, H, W) target tensor
        pred_low, pred_med, pred_high: (1, H, W) prediction tensors
        iteration: Current iteration number
        num_iterations: Total iterations
        
    Returns:
        np.ndarray: RGB image frame with 4 subplots
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠ matplotlib not available, using simple concatenation")
        # Fallback: simple horizontal concatenation
        target_np = np.clip(target[0].cpu().numpy(), 0, 1) * 255
        pred_low_np = np.clip(pred_low[0].cpu().numpy(), 0, 1) * 255
        pred_med_np = np.clip(pred_med[0].cpu().numpy(), 0, 1) * 255
        pred_high_np = np.clip(pred_high[0].cpu().numpy(), 0, 1) * 255
        
        # Stack horizontally with labels
        combined = np.concatenate([target_np, pred_low_np, pred_med_np, pred_high_np], axis=1)
        combined_rgb = np.stack([combined, combined, combined], axis=-1).astype(np.uint8)
        return combined_rgb
    
    # Use matplotlib for better formatting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=100)
    
    # Convert tensors to numpy
    target_np = np.clip(target[0].cpu().numpy(), 0, 1)
    pred_low_np = np.clip(pred_low[0].cpu().numpy(), 0, 1)
    pred_med_np = np.clip(pred_med[0].cpu().numpy(), 0, 1)
    pred_high_np = np.clip(pred_high[0].cpu().numpy(), 0, 1)
    
    # Plot each subplot
    axes[0].imshow(target_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Target', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(pred_low_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Low Damping\n(lr=0.1)', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(pred_med_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Medium Damping\n(lr=0.5)', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(pred_high_np, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('High Damping\n(lr=2.0)', fontsize=11, fontweight='bold')
    axes[3].axis('off')
    
    # Add iteration counter at the top
    fig.suptitle(f'Optimization Progress - Iteration {iteration}/{num_iterations}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Remove spacing between subplots
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.1)
    
    # Convert figure to numpy array
    fig.canvas.draw()
    
    # Use the newer matplotlib API
    buffer_rgba = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    image_rgba = np.frombuffer(buffer_rgba, dtype=np.uint8).reshape((h, w, 4))
    image_rgb = image_rgba[:, :, :3]  # Drop alpha channel
    plt.close(fig)
    
    return image_rgb


def create_side_by_side_image(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """
    Create side-by-side image for GIF frame (prediction | target).
    
    Args:
        pred: (1, H, W) prediction tensor
        target: (1, H, W) target tensor
        
    Returns:
        np.ndarray: (H, 2*W, 3) RGB image for GIF
    """
    # Convert to numpy and scale to [0, 255]
    pred_np = np.clip(pred[0].detach().cpu().numpy(), 0, 1) * 255
    target_np = np.clip(target[0].detach().cpu().numpy(), 0, 1) * 255
    
    # Convert to uint8
    pred_uint8 = pred_np.astype(np.uint8)
    target_uint8 = target_np.astype(np.uint8)
    
    # Create side-by-side: make both RGB for consistency
    pred_rgb = np.stack([pred_uint8, pred_uint8, pred_uint8], axis=-1)
    target_rgb = np.stack([target_uint8, target_uint8, target_uint8], axis=-1)
    
    # Concatenate horizontally
    side_by_side = np.concatenate([pred_rgb, target_rgb], axis=1)
    
    return side_by_side


def optimize_image(
    target: torch.Tensor,
    num_iterations: int,
    config: ImageLossOptimizerConfig,
    lr: float,
    damping: float,
    device: str = 'cuda',
    save_dir: str = None,
    verbose: bool = True
) -> Tuple[List[float], List[torch.Tensor]]:
    """
    Perform damped second-order Newton optimization on image prediction.
    
    Args:
        target: (1, H, W) target image
        num_iterations: Number of optimization iterations
        config: ImageLossOptimizerConfig
        lr: Learning rate
        damping: Damping coefficient
        device: Device to use ('cuda' or 'cpu')
        save_dir: Directory to save predictions (optional)
        verbose: Print progress
        
    Returns:
        Tuple[loss_history, prediction_history]
    """
    # Move target to device
    target = target.to(device)
    
    # Initialize prediction as random [0, 1]
    torch.manual_seed(42)
    prediction_data = torch.rand_like(target).to(device)
    
    class Param:
        def __init__(self, data):
            self.data = data
    
    prediction = Param(prediction_data)
    
    # Setup loss function and optimizer
    loss_fn = ImageLossCUDA(
        lambda_l2=1.0,
        lambda_ssim=0.25,
        window_size=11
    )
    
    # Update config with current lr and damping
    config.lr = lr
    config.damping = damping
    
    optimizer = ImageLossSecondOrderOptimizer(Params(), config)
    optimizer.params.add_param('prediction', prediction)
    
    params = optimizer.params
    data = Data()
    data.set('target', target)
    
    loss_history = []
    pred_history = []
    
    if verbose:
        print(f"\n  Optimization Progress (lr={lr:.4f}, damping={damping:.4f}):")
        print(f"  {'Iter':>5} | {'Loss':>12} | {'Δ Loss':>12} | {'Grad Norm':>12}")
        print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    
    for iteration in range(num_iterations):
        # Forward + gradient + Hessian
        params.param_map['prediction'] = prediction
        forward_result, grad_result, hess_result = loss_fn.forward_backward2(
            params, data, config=None
        )
        
        loss_val = forward_result.get_loss().item()
        loss_history.append(loss_val)
        
        grad_norm = torch.norm(grad_result.get('prediction')).item()
        
        # Optimization step
        optimizer.add_grad_hess(grad_result, hess_result)
        optimizer.step()
        optimizer.zero_grad()
        
        # Clamp to [0, 1]
        prediction.data = prediction.data.clamp(0.0, 1.0)
        
        # Save prediction
        pred_history.append(prediction.data.clone().detach().cpu())
        
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"pred_{iteration:04d}.png")
            save_image_tensor(pred_history[-1], save_path)
        
        # Print progress
        if verbose and (iteration % 5 == 0 or iteration == num_iterations - 1):
            delta_loss = loss_history[-2] - loss_val if iteration > 0 else 0
            print(f"  {iteration:>5} | {loss_val:>12.6e} | {delta_loss:>12.6e} | {grad_norm:>12.6e}")
    
    return loss_history, pred_history


def generate_comparison_gif(
    target: torch.Tensor,
    pred_history_low: List[torch.Tensor],
    pred_history_med: List[torch.Tensor],
    pred_history_high: List[torch.Tensor],
    output_path: str,
    num_iterations: int,
    fps: int = 5
):
    """
    Generate 4-subplot comparison GIF (Target | Low | Medium | High damping).
    Shows all three configurations side-by-side with iteration counter.
    
    Args:
        target: (1, H, W) target tensor
        pred_history_*: Lists of (1, H, W) prediction tensors
        output_path: Output GIF path
        num_iterations: Total number of iterations
        fps: Frames per second
    """
    if not IMAGEIO_AVAILABLE:
        print(f"  ⚠ imageio not available, skipping comparison GIF generation")
        return
    
    print(f"\n  Generating comparison GIF: {output_path}")
    frames = []
    
    # Create frames for each iteration (subsample to keep file size reasonable)
    subsample_rate = 2  # Every 2 iterations
    for i in range(0, len(pred_history_low), subsample_rate):
        frame = create_four_subplot_frame(
            target,
            pred_history_low[i],
            pred_history_med[i],
            pred_history_high[i],
            i,
            num_iterations
        )
        frames.append(frame)
    
    # Add final frame repeated a few times for better viewing
    for _ in range(3):
        frames.append(frames[-1])
    
    # Write GIF
    imageio.mimsave(output_path, frames, fps=fps, format='GIF')
    print(f"  ✓ Comparison GIF saved: {output_path}")
    print(f"    Total frames: {len(frames)}")
    print(f"    Frame rate: {fps} fps")


def generate_gif(
    pred_history: List[torch.Tensor],
    target: torch.Tensor,
    output_path: str,
    fps: int = 5
):
    """
    Generate side-by-side GIF animation (prediction | target).
    DEPRECATED: Use generate_comparison_gif for multi-config comparison.
    
    Args:
        pred_history: List of (1, H, W) prediction tensors
        target: (1, H, W) target tensor
        output_path: Output GIF path
        fps: Frames per second
    """
    if not IMAGEIO_AVAILABLE:
        print(f"  ⚠ imageio not available, skipping GIF generation")
        return
    
    print(f"\n  Generating GIF: {output_path}")
    frames = []
    
    # Add each prediction-target pair as a frame
    for pred in pred_history:
        frame = create_side_by_side_image(pred, target)
        frames.append(frame)
    
    # Add final frame repeated a few times for better viewing
    for _ in range(3):
        frames.append(frames[-1])
    
    # Write GIF
    imageio.mimsave(output_path, frames, fps=fps, format='GIF')
    print(f"  ✓ GIF saved: {output_path}")
    print(f"    Total frames: {len(frames)}")
    print(f"    Frame rate: {fps} fps")


def main():
    print("\n" * 2)
    print_separator("Loss Optimization Visualization Test", "=")
    
    # Check CUDA availability (required for ImageLossCUDA)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ImageLossCUDA. Please run on a GPU device.")
    
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
        f'loss_vis_{timestamp}'
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
    target = resample_image(target, target_size=512)  # Increased to 512px for better visualization
    
    B, H, W = target.shape
    print(f"  Final target shape: {target.shape}")
    print(f"  Value range: [{target.min():.4f}, {target.max():.4f}]")
    
    # Move to device
    target_device = target.to(device)
    
    # ========================================================================
    # Step 2: Optimization with different damping levels
    # ========================================================================
    print_separator("Step 2: Optimization with Different Damping Levels", "=")
    
    # Define damping configurations
    # Note: Higher damping → larger denominator → need higher learning rate
    # Lower damping → smaller denominator → need lower learning rate
    damping_configs = [
        {
            "name": "Low Damping",
            "damping": 1e-4,
            "lr": 0.5,
            "description": "Conservative with low damping, small denominator needs small lr"
        },
        {
            "name": "Medium Damping",
            "damping": 1e-3,
            "lr": 2.5,
            "description": "Balanced optimization"
        },
        {
            "name": "High Damping",
            "damping": 1e-2,
            "lr": 10.0,
            "description": "Large denominator from high damping, needs larger lr"
        },
    ]
    
    num_iterations = 200  # Increased for better visualization
    results_summary = {}
    all_pred_histories = {}  # Store all predictions for comparison GIF
    
    for config_idx, config_info in enumerate(damping_configs):
        print_separator(f"{config_info['name']}", "-")
        print(f"\n  Configuration:")
        print(f"    Damping: {config_info['damping']:.2e}")
        print(f"    Learning rate: {config_info['lr']:.4f}")
        print(f"    Description: {config_info['description']}")
        print(f"    Iterations: {num_iterations}")
        
        # Create subdirectory for this configuration
        config_dir = os.path.join(output_root, config_info['name'].lower().replace(' ', '_'))
        os.makedirs(config_dir, exist_ok=True)
        os.makedirs(os.path.join(config_dir, 'predictions'), exist_ok=True)
        
        # Create optimizer config
        optim_config = ImageLossOptimizerConfig(
            lr=config_info['lr'],
            damping=config_info['damping']
        )
        
        # Run optimization
        loss_history, pred_history = optimize_image(
            target_device,
            num_iterations=num_iterations,
            config=optim_config,
            lr=config_info['lr'],
            damping=config_info['damping'],
            device=device,
            save_dir=os.path.join(config_dir, 'predictions'),
            verbose=True
        )
        
        # Store predictions for comparison GIF
        all_pred_histories[config_info['name'].lower().replace(' ', '_')] = pred_history
        
        # Analyze results
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\n  Results:")
        print(f"    Initial loss: {initial_loss:.6e}")
        print(f"    Final loss: {final_loss:.6e}")
        print(f"    Loss reduction: {loss_reduction:.2f}%")
        print(f"    Decreasing steps: {sum(1 for i in range(1, len(loss_history)) if loss_history[i] < loss_history[i-1])}")
        
        results_summary[config_info['name']] = {
            'damping': config_info['damping'],
            'lr': config_info['lr'],
            'initial_loss': float(initial_loss),
            'final_loss': float(final_loss),
            'loss_reduction': float(loss_reduction),
            'num_iterations': num_iterations,
            'predictions_dir': os.path.join(config_dir, 'predictions')
        }
        
        # ====================================================================
        # Generate GIF
        # ====================================================================
        gif_path = os.path.join(config_dir, f'{config_info["name"].lower().replace(" ", "_")}_optimization.gif')
        
        # Subsample predictions for GIF (every 2 iterations to keep file size reasonable)
        pred_history_subsampled = pred_history[::2]
        
        generate_gif(pred_history_subsampled, target.cpu(), gif_path, fps=5)
        
        # Save loss history as JSON
        loss_json_path = os.path.join(config_dir, 'loss_history.json')
        with open(loss_json_path, 'w') as f:
            json.dump({
                'loss_history': loss_history,
                'damping': config_info['damping'],
                'lr': config_info['lr'],
                'num_iterations': num_iterations
            }, f, indent=2)
        print(f"  Saved loss history to: {loss_json_path}")
        
        # Save target image
        target_img_path = os.path.join(config_dir, 'target.png')
        save_image_tensor(target.cpu(), target_img_path)
        print(f"  Saved target image to: {target_img_path}")
        
        # Save initial and final predictions
        save_image_tensor(pred_history[0].to(device), os.path.join(config_dir, 'pred_initial.png'))
        save_image_tensor(pred_history[-1].to(device), os.path.join(config_dir, 'pred_final.png'))
        print(f"  Saved initial and final predictions")
    
    # ========================================================================
    # Generate comparison GIF (all 3 configurations side-by-side)
    # ========================================================================
    print_separator("Generating Comparison GIF")
    
    comparison_gif_path = os.path.join(output_root, 'all_configs_comparison.gif')
    generate_comparison_gif(
        target.cpu(),
        all_pred_histories['low_damping'],
        all_pred_histories['medium_damping'],
        all_pred_histories['high_damping'],
        comparison_gif_path,
        num_iterations,
        fps=5
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" * 2)
    print_separator("Optimization Summary", "=")
    
    # Save results summary
    summary_path = os.path.join(output_root, 'optimization_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'device': device,
            'target_shape': list(target.shape),
            'num_iterations': num_iterations,
            'configurations': results_summary
        }, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")
    
    # Print comparison table
    print("\nComparison Table:")
    print(f"{'Configuration':<20} | {'Damping':<12} | {'LR':<8} | {'Initial Loss':<14} | {'Final Loss':<14} | {'Reduction':<12}")
    print(f"{'-'*20}-+-{'-'*12}-+-{'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*12}")
    
    for config_name, results in results_summary.items():
        print(f"{config_name:<20} | {results['damping']:<12.2e} | {results['lr']:<8.4f} | "
              f"{results['initial_loss']:<14.6e} | {results['final_loss']:<14.6e} | {results['loss_reduction']:<12.2f}%")
    
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
