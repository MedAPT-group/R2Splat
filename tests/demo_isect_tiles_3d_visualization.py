"""
Demo: Isect Tiles 3D Visualization

This demo visualizes how the isect_tiles_3d function computes 3D tile-Gaussian intersections.

Features:
- 128x128x96 volume with 8x8x6 tile grid (tile_size=16)
- Three 3D anisotropic Gaussians with different orientations
- Visualization of Gaussian density on X, Y, Z slices
- Color-coded tile borders based on which Gaussians cover them

Author: Dy164
Date: 2026-02-02
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import math


def create_anisotropic_gaussian_3d(
    center: tuple,
    sigmas: tuple,
    angles_deg: tuple = (0, 0, 0),
    size: tuple = (96, 128, 128)
) -> np.ndarray:
    """
    Create a 3D anisotropic Gaussian.
    
    Args:
        center: (x, y, z) center of the Gaussian
        sigmas: (sigma_x, sigma_y, sigma_z) standard deviations
        angles_deg: (angle_xy, angle_xz, angle_yz) rotation angles in degrees
                   (for simplicity, we use axis-aligned Gaussians here)
        size: (depth, height, width) of the output volume (z, y, x)
    
    Returns:
        3D numpy array with the Gaussian
    """
    d, h, w = size
    z, y, x = np.ogrid[:d, :h, :w]
    
    # Shift to center
    x = x - center[0]
    y = y - center[1]
    z = z - center[2]
    
    sigma_x, sigma_y, sigma_z = sigmas
    
    # Compute Gaussian (axis-aligned for simplicity)
    gaussian = np.exp(-(x**2 / (2 * sigma_x**2) + 
                        y**2 / (2 * sigma_y**2) + 
                        z**2 / (2 * sigma_z**2)))
    
    return gaussian.astype(np.float32)


def compute_gaussian_radius_3d(sigmas: tuple, threshold: float = 0.01) -> tuple:
    """
    Compute the effective radius of an anisotropic 3D Gaussian in each dimension.
    
    Args:
        sigmas: (sigma_x, sigma_y, sigma_z) standard deviations
        threshold: Threshold for considering the Gaussian negligible
    
    Returns:
        (radius_x, radius_y, radius_z) in voxels
    """
    # For a Gaussian, value = exp(-r^2 / (2*sigma^2))
    # Solving for r when value = threshold:
    # r = sigma * sqrt(-2 * ln(threshold))
    factor = math.sqrt(-2 * math.log(threshold))
    return tuple(int(math.ceil(s * factor)) for s in sigmas)


def run_isect_tiles_3d_demo():
    """Run the isect_tiles_3d visualization demo."""
    
    # =========================================================================
    # Setup parameters
    # =========================================================================
    
    volume_size_x = 128
    volume_size_y = 128
    volume_size_z = 96
    tile_size = 16
    tile_width = volume_size_x // tile_size   # 8 (x dimension)
    tile_depth = volume_size_y // tile_size   # 8 (y dimension)
    tile_height = volume_size_z // tile_size  # 6 (z dimension)
    n_tiles = tile_width * tile_depth * tile_height  # 384
    
    # Define three 3D anisotropic Gaussians with different properties
    gaussians_params = [
        # Gaussian 1: Front-left area, elongated in X direction
        {
            'center': (40, 40, 30),  # (x, y, z)
            'sigmas': (25, 12, 15),  # (sigma_x, sigma_y, sigma_z)
            'color': (1.0, 0.3, 0.3),  # Red
            'name': 'Gaussian 1 (Red)'
        },
        # Gaussian 2: Center-right area, elongated in Y direction
        {
            'center': (90, 80, 50),
            'sigmas': (15, 30, 12),
            'color': (0.3, 0.8, 0.3),  # Green
            'name': 'Gaussian 2 (Green)'
        },
        # Gaussian 3: Back area, elongated in Z direction
        {
            'center': (60, 60, 75),
            'sigmas': (12, 15, 28),
            'color': (0.3, 0.5, 1.0),  # Blue
            'name': 'Gaussian 3 (Blue)'
        },
    ]
    
    print("=" * 60)
    print("Isect Tiles 3D Visualization Demo")
    print("=" * 60)
    print(f"Volume size: {volume_size_x}x{volume_size_y}x{volume_size_z}")
    print(f"Tile size: {tile_size}x{tile_size}x{tile_size}")
    print(f"Tile grid: {tile_width}x{tile_depth}x{tile_height} = {n_tiles} tiles")
    print()
    
    # =========================================================================
    # Create Gaussian volumes and compute radii
    # =========================================================================
    
    gaussian_volumes = []
    means3d_list = []
    radii_list = []
    
    for i, params in enumerate(gaussians_params):
        # Create 3D Gaussian volume
        g_vol = create_anisotropic_gaussian_3d(
            center=params['center'],
            sigmas=params['sigmas'],
            size=(volume_size_z, volume_size_y, volume_size_x)  # (z, y, x)
        )
        gaussian_volumes.append(g_vol)
        
        # Store center for isect_tiles_3d (x, y, z order)
        means3d_list.append([params['center'][0], params['center'][1], params['center'][2]])
        
        # Compute effective radius in each dimension
        radius = compute_gaussian_radius_3d(params['sigmas'])
        radii_list.append(list(radius))
        
        print(f"{params['name']}:")
        print(f"  Center: {params['center']}")
        print(f"  Sigmas: {params['sigmas']}")
        print(f"  Effective radii: {radius}")
        print()
    
    # =========================================================================
    # Prepare tensors for isect_tiles_3d
    # =========================================================================
    
    N = len(gaussians_params)
    
    means3d = torch.tensor(means3d_list, dtype=torch.float32)  # [N, 3]
    radii = torch.tensor(radii_list, dtype=torch.int32)  # [N, 3]
    
    # Move to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tile_gaussian_map = {}  # tile_id -> list of gaussian indices
    
    if device.type != 'cuda':
        print("WARNING: CUDA not available. Skipping isect_tiles_3d computation.")
        print("The visualization will still show the Gaussians but not the intersection results.")
        tiles_per_gauss = None
        isect_ids = None
        flatten_ids = None
    else:
        means3d = means3d.to(device)
        radii = radii.to(device)
        
        # =====================================================================
        # Call isect_tiles_3d
        # =====================================================================
        
        from gsplat2ct.cuda import isect_tiles_3d, isect_offset_encode_3d
        
        print("Calling isect_tiles_3d...")
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
            means3d, radii, tile_size, tile_width, tile_depth, tile_height
        )
        
        # Also compute offsets for per-tile access
        offsets = isect_offset_encode_3d(isect_ids, tile_width, tile_depth, tile_height)
        
        # Move results to CPU for visualization
        tiles_per_gauss = tiles_per_gauss.cpu()
        isect_ids = isect_ids.cpu()
        flatten_ids = flatten_ids.cpu()
        offsets = offsets.cpu()
        
        print(f"\nResults:")
        print(f"  tiles_per_gauss shape: {tiles_per_gauss.shape}")
        print(f"  tiles_per_gauss values: {tiles_per_gauss.tolist()}")
        print(f"  Total intersections: {isect_ids.shape[0]}")
        print()
        
        # =====================================================================
        # Detailed analysis of results
        # =====================================================================
        
        print("=" * 60)
        print("DETAILED ANALYSIS")
        print("=" * 60)
        
        # Build tile -> Gaussian mapping
        for i in range(len(flatten_ids)):
            gid = flatten_ids[i].item()
            tile_id = isect_ids[i].item()
            
            if tile_id not in tile_gaussian_map:
                tile_gaussian_map[tile_id] = []
            tile_gaussian_map[tile_id].append(gid)
        
        # Summary statistics
        total_tiles = tile_width * tile_depth * tile_height
        non_empty_tiles = len(tile_gaussian_map)
        
        print(f"\nTotal tiles: {total_tiles}")
        print(f"Non-empty tiles: {non_empty_tiles} ({100*non_empty_tiles/total_tiles:.1f}%)")
        
        # Count tiles per Gaussian
        print(f"\nTiles covered by each Gaussian:")
        for gid, params in enumerate(gaussians_params):
            count = tiles_per_gauss[gid].item()
            print(f"  {params['name']}: {count} tiles")
        
        # Count tiles by number of Gaussians
        coverage_histogram = {}
        for tile_id, gaussians in tile_gaussian_map.items():
            n_gauss = len(gaussians)
            coverage_histogram[n_gauss] = coverage_histogram.get(n_gauss, 0) + 1
        
        print(f"\nTile coverage histogram:")
        for n_gauss in sorted(coverage_histogram.keys()):
            count = coverage_histogram[n_gauss]
            print(f"  Tiles covered by {n_gauss} Gaussian(s): {count}")
    
    # =========================================================================
    # Visualization: 3 slices (X, Y, Z planes)
    # =========================================================================
    
    print("\nCreating visualization...")
    
    # Choose slice positions (at Gaussian centers for interesting views)
    slice_x = 60  # X slice (YZ plane)
    slice_y = 60  # Y slice (XZ plane)
    slice_z = 50  # Z slice (XY plane)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Function to get tile coverage for a slice
    def get_tile_coverage_2d(axis: str, slice_idx: int):
        """Get tiles and their Gaussian coverage for a 2D slice."""
        coverage = {}
        
        if axis == 'z':  # XY plane at z=slice_idx
            slice_tile_z = slice_idx // tile_size
            for tile_id, gids in tile_gaussian_map.items():
                tz = tile_id // (tile_depth * tile_width)
                ty = (tile_id % (tile_depth * tile_width)) // tile_width
                tx = tile_id % tile_width
                if tz == slice_tile_z:
                    key = (tx, ty)  # (tile_x, tile_y)
                    coverage[key] = gids
                    
        elif axis == 'y':  # XZ plane at y=slice_idx
            slice_tile_y = slice_idx // tile_size
            for tile_id, gids in tile_gaussian_map.items():
                tz = tile_id // (tile_depth * tile_width)
                ty = (tile_id % (tile_depth * tile_width)) // tile_width
                tx = tile_id % tile_width
                if ty == slice_tile_y:
                    key = (tx, tz)  # (tile_x, tile_z)
                    coverage[key] = gids
                    
        elif axis == 'x':  # YZ plane at x=slice_idx
            slice_tile_x = slice_idx // tile_size
            for tile_id, gids in tile_gaussian_map.items():
                tz = tile_id // (tile_depth * tile_width)
                ty = (tile_id % (tile_depth * tile_width)) // tile_width
                tx = tile_id % tile_width
                if tx == slice_tile_x:
                    key = (ty, tz)  # (tile_y, tile_z)
                    coverage[key] = gids
        
        return coverage
    
    # -------------------------------------------------------------------------
    # Row 1: Gaussian density slices
    # -------------------------------------------------------------------------
    
    slice_configs = [
        ('Z', slice_z, volume_size_x, volume_size_y, tile_width, tile_depth, 'X', 'Y'),
        ('Y', slice_y, volume_size_x, volume_size_z, tile_width, tile_height, 'X', 'Z'),
        ('X', slice_x, volume_size_y, volume_size_z, tile_depth, tile_height, 'Y', 'Z'),
    ]
    
    for col, (axis, slice_idx, dim1, dim2, tw, th, label1, label2) in enumerate(slice_configs):
        ax = axes[0, col]
        
        # Create combined RGB slice
        combined_slice = np.zeros((dim2, dim1, 3), dtype=np.float32)
        
        for i, (g_vol, params) in enumerate(zip(gaussian_volumes, gaussians_params)):
            if axis == 'Z':
                g_slice = g_vol[slice_idx, :, :]  # [y, x]
            elif axis == 'Y':
                g_slice = g_vol[:, slice_idx, :]  # [z, x]
            else:  # X
                g_slice = g_vol[:, :, slice_idx]  # [z, y]
            
            for c in range(3):
                combined_slice[:, :, c] += g_slice * params['color'][c]
        
        combined_slice = np.clip(combined_slice, 0, 1)
        
        ax.imshow(combined_slice, origin='lower', aspect='auto')
        ax.set_title(f'{axis}-Slice at {axis.lower()}={slice_idx}\nGaussian Density', fontsize=11)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        
        # Draw tile grid
        for i in range(th + 1):
            y = i * tile_size
            ax.axhline(y=y, color='white', linewidth=0.5, alpha=0.5)
        for j in range(tw + 1):
            x = j * tile_size
            ax.axvline(x=x, color='white', linewidth=0.5, alpha=0.5)
        
        # Mark Gaussian centers (if in slice)
        for i, params in enumerate(gaussians_params):
            cx, cy, cz = params['center']
            if axis == 'Z':
                if abs(cz - slice_idx) < tile_size:
                    ax.scatter(cx, cy, c=[params['color']], s=100, marker='x', linewidths=2)
                    ax.annotate(f"G{i+1}", (cx, cy), xytext=(5, 5), 
                               textcoords='offset points', color='white', fontsize=9)
            elif axis == 'Y':
                if abs(cy - slice_idx) < tile_size:
                    ax.scatter(cx, cz, c=[params['color']], s=100, marker='x', linewidths=2)
                    ax.annotate(f"G{i+1}", (cx, cz), xytext=(5, 5),
                               textcoords='offset points', color='white', fontsize=9)
            else:  # X
                if abs(cx - slice_idx) < tile_size:
                    ax.scatter(cy, cz, c=[params['color']], s=100, marker='x', linewidths=2)
                    ax.annotate(f"G{i+1}", (cy, cz), xytext=(5, 5),
                               textcoords='offset points', color='white', fontsize=9)
    
    # -------------------------------------------------------------------------
    # Row 2: Tile intersection visualization
    # -------------------------------------------------------------------------
    
    for col, (axis, slice_idx, dim1, dim2, tw, th, label1, label2) in enumerate(slice_configs):
        ax = axes[1, col]
        
        # Create faded Gaussian slice
        combined_slice = np.zeros((dim2, dim1, 3), dtype=np.float32)
        
        for i, (g_vol, params) in enumerate(zip(gaussian_volumes, gaussians_params)):
            if axis == 'Z':
                g_slice = g_vol[slice_idx, :, :]
            elif axis == 'Y':
                g_slice = g_vol[:, slice_idx, :]
            else:
                g_slice = g_vol[:, :, slice_idx]
            
            for c in range(3):
                combined_slice[:, :, c] += g_slice * params['color'][c]
        
        combined_slice = np.clip(combined_slice * 0.3, 0, 1)
        
        ax.imshow(combined_slice, origin='lower', aspect='auto')
        ax.set_title(f'{axis}-Slice at {axis.lower()}={slice_idx}\nTile-Gaussian Intersections', fontsize=11)
        ax.set_xlabel(label1)
        ax.set_ylabel(label2)
        
        # Get tile coverage for this slice
        if device.type == 'cuda' and tile_gaussian_map:
            coverage = get_tile_coverage_2d(axis.lower(), slice_idx)
            
            # Draw tiles with coverage
            for ty_idx in range(th):
                for tx_idx in range(tw):
                    x_start = tx_idx * tile_size
                    y_start = ty_idx * tile_size
                    
                    tile_key = (tx_idx, ty_idx)
                    
                    if tile_key in coverage:
                        gaussians_in_tile = coverage[tile_key]
                        
                        # Draw colored borders for each Gaussian
                        for offset_idx, gid in enumerate(gaussians_in_tile):
                            color = gaussians_params[gid]['color']
                            offset = offset_idx * 2
                            
                            rect = patches.Rectangle(
                                (x_start + offset, y_start + offset),
                                tile_size - 2 * offset,
                                tile_size - 2 * offset,
                                linewidth=2,
                                edgecolor=color,
                                facecolor='none',
                                alpha=0.9
                            )
                            ax.add_patch(rect)
                        
                        # Fill with blended color
                        avg_color = np.mean([gaussians_params[g]['color'] for g in gaussians_in_tile], axis=0)
                        fill_rect = patches.Rectangle(
                            (x_start, y_start),
                            tile_size, tile_size,
                            linewidth=0,
                            facecolor=avg_color,
                            alpha=0.2
                        )
                        ax.add_patch(fill_rect)
                    else:
                        # Empty tile
                        rect = patches.Rectangle(
                            (x_start, y_start),
                            tile_size, tile_size,
                            linewidth=0.5,
                            edgecolor='gray',
                            facecolor='none',
                            alpha=0.3
                        )
                        ax.add_patch(rect)
        else:
            # Draw simple grid if CUDA not available
            for i in range(th + 1):
                y = i * tile_size
                ax.axhline(y=y, color='gray', linewidth=0.5, alpha=0.5)
            for j in range(tw + 1):
                x = j * tile_size
                ax.axvline(x=x, color='gray', linewidth=0.5, alpha=0.5)
        
        ax.set_xlim(0, dim1)
        ax.set_ylim(0, dim2)
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=params['color'], alpha=0.7, label=params['name'])
        for params in gaussians_params
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10, 
               bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save figure
    output_path = 'tests/isect_tiles_3d_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Show the figure
    plt.show()
    
    # =========================================================================
    # Print tile coverage matrix for each slice
    # =========================================================================
    
    if device.type == 'cuda':
        print("\n" + "=" * 60)
        print("Tile Coverage on Slices")
        print("=" * 60)
        
        for axis, slice_idx, _, _, tw, th, _, _ in slice_configs:
            coverage = get_tile_coverage_2d(axis.lower(), slice_idx)
            
            print(f"\n{axis}-Slice at {axis.lower()}={slice_idx}:")
            print("  - R: Gaussian 1, G: Gaussian 2, B: Gaussian 3")
            
            # Print header
            print("    ", end="")
            for j in range(tw):
                print(f" {j} ", end="")
            print()
            print("   +" + "---" * tw + "+")
            
            for i in range(th):
                print(f" {i} |", end="")
                for j in range(tw):
                    tile_key = (j, i)
                    if tile_key in coverage:
                        gaussians = coverage[tile_key]
                        chars = ""
                        if 0 in gaussians:
                            chars += "R"
                        if 1 in gaussians:
                            chars += "G"
                        if 2 in gaussians:
                            chars += "B"
                        print(f"{chars:^3}", end="")
                    else:
                        print(" . ", end="")
                print("|")
            
            print("   +" + "---" * tw + "+")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_isect_tiles_3d_demo()
