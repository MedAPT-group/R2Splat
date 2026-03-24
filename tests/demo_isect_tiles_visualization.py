"""
Demo: Isect Tiles Visualization

This demo visualizes how the isect_tiles function computes tile-Gaussian intersections.

Features:
- 256x256 image with 8x8 tile grid (tile_size=32)
- Three anisotropic 2D Gaussians with different orientations
- Visualization of Gaussian coverage and tile intersections
- Color-coded tile borders based on which Gaussians cover them

Author: Dy164
Date: 2025-01-12
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import math


def create_anisotropic_gaussian_2d(
    center: tuple,
    sigma_major: float,
    sigma_minor: float,
    angle_deg: float,
    size: tuple = (256, 256)
) -> np.ndarray:
    """
    Create a 2D anisotropic Gaussian.
    
    Args:
        center: (x, y) center of the Gaussian
        sigma_major: Standard deviation along major axis
        sigma_minor: Standard deviation along minor axis
        angle_deg: Rotation angle in degrees (counter-clockwise)
        size: (height, width) of the output image
    
    Returns:
        2D numpy array with the Gaussian
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    
    # Shift to center
    x = x - center[0]
    y = y - center[1]
    
    # Rotation matrix
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotate coordinates
    x_rot = cos_a * x + sin_a * y
    y_rot = -sin_a * x + cos_a * y
    
    # Compute Gaussian
    gaussian = np.exp(-(x_rot**2 / (2 * sigma_major**2) + y_rot**2 / (2 * sigma_minor**2)))
    
    return gaussian.astype(np.float32)


def compute_gaussian_radius(sigma_major: float, sigma_minor: float, threshold: float = 0.01) -> int:
    """
    Compute the effective radius of an anisotropic Gaussian.
    
    The radius is based on where the Gaussian falls below a threshold.
    For an anisotropic Gaussian, we use the major axis.
    
    Args:
        sigma_major: Standard deviation along major axis
        sigma_minor: Standard deviation along minor axis
        threshold: Threshold for considering the Gaussian negligible
    
    Returns:
        Radius in pixels
    """
    # For a Gaussian, value = exp(-r^2 / (2*sigma^2))
    # Solving for r when value = threshold:
    # r = sigma * sqrt(-2 * ln(threshold))
    radius = sigma_major * math.sqrt(-2 * math.log(threshold))
    return int(math.ceil(radius))


def run_isect_tiles_demo():
    """Run the isect_tiles visualization demo."""
    
    # =========================================================================
    # Setup parameters
    # =========================================================================
    
    image_size = 256
    tile_size = 32
    tile_width = image_size // tile_size  # 8
    tile_height = image_size // tile_size  # 8
    n_tiles = tile_width * tile_height  # 64
    
    # Define three anisotropic Gaussians with different properties
    # Each: (center_x, center_y, sigma_major, sigma_minor, angle_deg, color)
    gaussians_params = [
        # Gaussian 1: Top-left area, elongated horizontally
        {
            'center': (70, 60),
            'sigma_major': 35,
            'sigma_minor': 15,
            'angle': 20,
            'color': (1.0, 0.3, 0.3),  # Red
            'name': 'Gaussian 1 (Red)'
        },
        # Gaussian 2: Center-right area, elongated diagonally
        {
            'center': (180, 130),
            'sigma_major': 40,
            'sigma_minor': 12,
            'angle': -45,
            'color': (0.3, 0.8, 0.3),  # Green
            'name': 'Gaussian 2 (Green)'
        },
        # Gaussian 3: Bottom-left area, elongated vertically
        {
            'center': (80, 200),
            'sigma_major': 30,
            'sigma_minor': 18,
            'angle': 80,
            'color': (0.3, 0.5, 1.0),  # Blue
            'name': 'Gaussian 3 (Blue)'
        },
    ]
    
    print("=" * 60)
    print("Isect Tiles Visualization Demo")
    print("=" * 60)
    print(f"Image size: {image_size}x{image_size}")
    print(f"Tile size: {tile_size}x{tile_size}")
    print(f"Tile grid: {tile_width}x{tile_height} = {n_tiles} tiles")
    print()
    
    # =========================================================================
    # Create Gaussian images and compute radii
    # =========================================================================
    
    gaussian_images = []
    means2d_list = []
    radii_list = []
    
    for i, params in enumerate(gaussians_params):
        # Create 2D Gaussian image
        g_img = create_anisotropic_gaussian_2d(
            center=params['center'],
            sigma_major=params['sigma_major'],
            sigma_minor=params['sigma_minor'],
            angle_deg=params['angle'],
            size=(image_size, image_size)
        )
        gaussian_images.append(g_img)
        
        # Store center for isect_tiles
        means2d_list.append([params['center'][0], params['center'][1]])
        
        # Compute effective radius
        radius = compute_gaussian_radius(params['sigma_major'], params['sigma_minor'])
        radii_list.append(radius)
        
        print(f"{params['name']}:")
        print(f"  Center: {params['center']}")
        print(f"  Sigma: major={params['sigma_major']}, minor={params['sigma_minor']}")
        print(f"  Angle: {params['angle']}°")
        print(f"  Effective radius: {radius} pixels")
        print()
    
    # =========================================================================
    # Prepare tensors for isect_tiles
    # =========================================================================
    
    # We have C=1 camera and N=3 Gaussians
    C = 1
    N = len(gaussians_params)
    
    means2d = torch.tensor(means2d_list, dtype=torch.float32).unsqueeze(0)  # [1, 3, 2]
    radii = torch.tensor(radii_list, dtype=torch.int32).unsqueeze(0)  # [1, 3]
    
    # Move to CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type != 'cuda':
        print("WARNING: CUDA not available. Skipping isect_tiles computation.")
        print("The visualization will still show the Gaussians but not the intersection results.")
        tiles_per_gauss = None
        isect_ids = None
        flatten_ids = None
    else:
        means2d = means2d.to(device)
        radii = radii.to(device)
        
        # =====================================================================
        # Call isect_tiles
        # =====================================================================
        
        from gsplat2ct.cuda import isect_tiles, isect_offset_encode
        
        print("Calling isect_tiles...")
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d, radii, tile_size, tile_width, tile_height
        )
        
        # Also compute offsets for per-tile access
        offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
        
        # Move results to CPU for visualization
        tiles_per_gauss = tiles_per_gauss.cpu()
        isect_ids = isect_ids.cpu()
        flatten_ids = flatten_ids.cpu()
        offsets = offsets.cpu()
        
        print(f"\nResults:")
        print(f"  tiles_per_gauss shape: {tiles_per_gauss.shape}")
        print(f"  tiles_per_gauss values: {tiles_per_gauss.squeeze().tolist()}")
        print(f"  Total intersections: {isect_ids.shape[0]}")
        print()
        
        # =====================================================================
        # Detailed analysis and verification of results
        # =====================================================================
        
        print("=" * 60)
        print("DETAILED ANALYSIS OF COMPUTED DATA")
        print("=" * 60)
        
        # ---------------------------------------------------------------------
        # 1. Analysis of isect_ids
        # ---------------------------------------------------------------------
        print("\n1. ISECT_IDS (Intersection ID Encoding)")
        print("-" * 60)
        print(f"Shape: {isect_ids.shape}")
        print(f"Dtype: {isect_ids.dtype}")
        print(f"Encoding: isect_id = camera_id * n_tiles + tile_id")
        print(f"Purpose: Sort key for (camera, tile) ordering")
        print()
        
        # Show first 10 entries
        n_show = min(10, len(isect_ids))
        print(f"First {n_show} entries:")
        print(f"{'Index':<8} {'isect_id':<12} {'camera_id':<12} {'tile_id':<10} {'tile(y,x)':<12}")
        print("-" * 60)
        for i in range(n_show):
            isect_id = isect_ids[i].item()
            camera_id = isect_id // n_tiles
            tile_id = isect_id % n_tiles
            tile_y = tile_id // tile_width
            tile_x = tile_id % tile_width
            print(f"{i:<8} {isect_id:<12} {camera_id:<12} {tile_id:<10} ({tile_y},{tile_x})")
        
        # Verify sorting
        is_sorted = torch.all(isect_ids[1:] >= isect_ids[:-1]).item()
        print(f"\n✓ Sorting verification: {'PASS' if is_sorted else 'FAIL'}")
        
        # ---------------------------------------------------------------------
        # 2. Analysis of flatten_ids
        # ---------------------------------------------------------------------
        print("\n2. FLATTEN_IDS (Flattened Gaussian Index)")
        print("-" * 60)
        print(f"Shape: {flatten_ids.shape}")
        print(f"Dtype: {flatten_ids.dtype}")
        print(f"Encoding: flatten_id = camera_id * N + gaussian_id")
        print(f"Purpose: Track which Gaussian created each intersection")
        print()
        
        # Show first 10 entries with corresponding isect_ids
        print(f"First {n_show} entries (paired with isect_ids):")
        print(f"{'Index':<8} {'flatten_id':<12} {'camera_id':<12} {'gauss_id':<10} {'tile(y,x)':<12} {'Gaussian':<10}")
        print("-" * 60)
        for i in range(n_show):
            flatten_id = flatten_ids[i].item()
            camera_id_f = flatten_id // N
            gauss_id = flatten_id % N
            
            isect_id = isect_ids[i].item()
            tile_id = isect_id % n_tiles
            tile_y = tile_id // tile_width
            tile_x = tile_id % tile_width
            
            print(f"{i:<8} {flatten_id:<12} {camera_id_f:<12} {gauss_id:<10} ({tile_y},{tile_x}){' ':<7} G{gauss_id+1}")
        
        # Verify flatten_ids are within valid range
        min_fid = flatten_ids.min().item()
        max_fid = flatten_ids.max().item()
        expected_max = C * N - 1
        print(f"\n✓ Range verification: min={min_fid}, max={max_fid}, expected_max={expected_max}")
        print(f"  {'PASS' if max_fid <= expected_max and min_fid >= 0 else 'FAIL'}")
        
        # Count intersections per Gaussian
        print(f"\nIntersections per Gaussian:")
        for gid in range(N):
            count = (flatten_ids == gid).sum().item()
            expected = tiles_per_gauss[0, gid].item()
            match = "✓" if count == expected else "✗"
            print(f"  Gaussian {gid+1}: {count} intersections (expected: {expected}) {match}")
        
        # ---------------------------------------------------------------------
        # 3. Analysis of offsets
        # ---------------------------------------------------------------------
        print("\n3. OFFSETS (Per-Tile Starting Indices)")
        print("-" * 60)
        print(f"Shape: {offsets.shape}")
        print(f"Dtype: {offsets.dtype}")
        print(f"Encoding: offsets[c, ty, tx] = starting index in sorted arrays")
        print(f"Purpose: Fast lookup of Gaussians covering a specific tile")
        print()
        
        # Show offset values for some tiles
        print("Sample offset values (camera 0):")
        print(f"{'Tile(y,x)':<12} {'tile_id':<10} {'Offset':<10} {'#Gaussians':<12} {'Gaussians':<20}")
        print("-" * 60)
        
        sample_tiles = [(ty, tx) for ty in range(tile_height) for tx in range(tile_width)]
        shown_count = 0
        for ty, tx in sample_tiles:
            tile_id = ty * tile_width + tx
            start_idx = offsets[0, ty, tx].item()
            
            # Find end index (next tile or end of array)
            if tx + 1 < tile_width:
                end_idx = offsets[0, ty, tx + 1].item()
            elif ty + 1 < tile_height:
                end_idx = offsets[0, ty + 1, 0].item()
            else:
                end_idx = len(flatten_ids)
            
            n_gaussians = end_idx - start_idx
            
            if n_gaussians > 0:
                # Get Gaussian IDs for this tile
                gauss_ids_in_tile = [flatten_ids[j].item() for j in range(start_idx, end_idx)]
                gauss_names = [f"G{gid+1}" for gid in gauss_ids_in_tile]
                
                print(f"({ty},{tx}){' ':<8} {tile_id:<10} {start_idx:<10} {n_gaussians:<12} {', '.join(gauss_names)}")
                shown_count += 1
                
                if shown_count >= 15:  # Limit display
                    remaining = sum(1 for t_y, t_x in sample_tiles 
                                  if offsets[0, t_y, t_x].item() < len(flatten_ids) - 1
                                  and (t_y * tile_width + t_x) > tile_id)
                    if remaining > 0:
                        print(f"... ({remaining} more non-empty tiles not shown)")
                    break
        
        # Verify offsets are non-decreasing
        offsets_flat = offsets[0].flatten()
        is_nondecreasing = torch.all(offsets_flat[1:] >= offsets_flat[:-1]).item()
        print(f"\n✓ Monotonicity verification: {'PASS' if is_nondecreasing else 'FAIL'}")
        
        # Verify first and last offsets
        first_offset = offsets_flat[0].item()
        last_offset = offsets_flat[-1].item()
        print(f"✓ Boundary verification:")
        print(f"  First offset: {first_offset} (expected: 0) {'✓' if first_offset == 0 else '✗'}")
        print(f"  Last offset: {last_offset} (expected: {len(flatten_ids)}) {'✓' if last_offset == len(flatten_ids) else '✗'}")
        
        # ---------------------------------------------------------------------
        # 4. Cross-verification: Reconstruct tile_gaussian_map from offsets
        # ---------------------------------------------------------------------
        print("\n4. CROSS-VERIFICATION")
        print("-" * 60)
        print("Reconstructing tile-Gaussian mapping from offsets...")
        
        tile_gaussian_map_from_offsets = {}
        
        for ty in range(tile_height):
            for tx in range(tile_width):
                tile_id = ty * tile_width + tx
                start_idx = offsets[0, ty, tx].item()
                
                # Find end index
                if tx + 1 < tile_width:
                    end_idx = offsets[0, ty, tx + 1].item()
                elif ty + 1 < tile_height:
                    end_idx = offsets[0, ty + 1, 0].item()
                else:
                    end_idx = len(flatten_ids)
                
                if start_idx < end_idx:
                    gauss_ids = [flatten_ids[j].item() for j in range(start_idx, end_idx)]
                    tile_gaussian_map_from_offsets[tile_id] = gauss_ids
        
        print(f"Reconstructed {len(tile_gaussian_map_from_offsets)} non-empty tiles")
        
        # Decode which Gaussians cover each tile (original method)
        # Decode which Gaussians cover each tile (original method from isect_ids/flatten_ids)
        tile_gaussian_map = {}  # tile_id -> list of gaussian indices
        
        for i in range(len(flatten_ids)):
            # flatten_ids[i] gives the flattened Gaussian index (cid * N + gid)
            # Since C=1, this is just the Gaussian index
            gid = flatten_ids[i].item()
            
            # isect_ids[i] encodes camera_id * n_tiles + tile_id
            # Since C=1, this is just tile_id
            tile_id = isect_ids[i].item()
            
            if tile_id not in tile_gaussian_map:
                tile_gaussian_map[tile_id] = []
            tile_gaussian_map[tile_id].append(gid)
        
        # Compare the two methods
        print("\nComparing two reconstruction methods:")
        all_match = True
        mismatch_count = 0
        
        # Check all tiles
        all_tile_ids = set(list(tile_gaussian_map.keys()) + list(tile_gaussian_map_from_offsets.keys()))
        for tile_id in sorted(all_tile_ids):
            map1 = sorted(tile_gaussian_map.get(tile_id, []))
            map2 = sorted(tile_gaussian_map_from_offsets.get(tile_id, []))
            
            if map1 != map2:
                all_match = False
                mismatch_count += 1
                tile_y = tile_id // tile_width
                tile_x = tile_id % tile_width
                print(f"  ✗ MISMATCH at tile ({tile_y},{tile_x}): isect_ids={map1}, offsets={map2}")
        
        if all_match:
            print(f"  ✓ PASS: All {len(all_tile_ids)} tiles match between both methods")
        else:
            print(f"  ✗ FAIL: {mismatch_count} tiles have mismatches")
        
        # ---------------------------------------------------------------------
        # 5. Summary Statistics
        # ---------------------------------------------------------------------
        print("\n5. SUMMARY STATISTICS")
        print("-" * 60)
        
        total_tiles = tile_width * tile_height
        non_empty_tiles = len(tile_gaussian_map)
        empty_tiles = total_tiles - non_empty_tiles
        
        print(f"Total tiles: {total_tiles}")
        print(f"Non-empty tiles (covered by at least 1 Gaussian): {non_empty_tiles} ({100*non_empty_tiles/total_tiles:.1f}%)")
        print(f"Empty tiles: {empty_tiles} ({100*empty_tiles/total_tiles:.1f}%)")
        print()
        
        # Count tiles by number of Gaussians
        coverage_histogram = {}
        for tile_id, gaussians in tile_gaussian_map.items():
            n_gauss = len(gaussians)
            coverage_histogram[n_gauss] = coverage_histogram.get(n_gauss, 0) + 1
        
        print("Tile coverage histogram:")
        for n_gauss in sorted(coverage_histogram.keys()):
            count = coverage_histogram[n_gauss]
            print(f"  Tiles covered by {n_gauss} Gaussian(s): {count}")
        
        # Show overlapping tiles
        overlapping_tiles = {tid: gids for tid, gids in tile_gaussian_map.items() if len(gids) > 1}
        if overlapping_tiles:
            print(f"\nOverlapping tiles ({len(overlapping_tiles)} tiles covered by multiple Gaussians):")
            for tile_id in sorted(overlapping_tiles.keys())[:10]:  # Show first 10
                tile_y = tile_id // tile_width
                tile_x = tile_id % tile_width
                gauss_ids = overlapping_tiles[tile_id]
                gauss_names = [f"G{gid+1}" for gid in gauss_ids]
                print(f"  Tile ({tile_y},{tile_x}): {', '.join(gauss_names)}")
            
            if len(overlapping_tiles) > 10:
                print(f"  ... ({len(overlapping_tiles) - 10} more)")
        
        print("\n" + "=" * 60)
        print()
        
        print("Tile -> Gaussian coverage:")
        for tile_id in sorted(tile_gaussian_map.keys()):
            tile_y = tile_id // tile_width
            tile_x = tile_id % tile_width
            gaussians = tile_gaussian_map[tile_id]
            gaussian_names = [f"G{g+1}" for g in gaussians]
            print(f"  Tile ({tile_x}, {tile_y}): {gaussian_names}")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # -------------------------------------------------------------------------
    # Left plot: Combined Gaussian image with tile grid
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    
    # Create combined RGB image from Gaussians
    combined_img = np.zeros((image_size, image_size, 3), dtype=np.float32)
    for i, (g_img, params) in enumerate(zip(gaussian_images, gaussians_params)):
        for c in range(3):
            combined_img[:, :, c] += g_img * params['color'][c]
    
    # Normalize to [0, 1]
    combined_img = np.clip(combined_img, 0, 1)
    
    ax1.imshow(combined_img, origin='upper')
    ax1.set_title('Gaussian Kernels with Tile Grid', fontsize=12)
    
    # Draw tile grid
    for i in range(tile_height + 1):
        y = i * tile_size
        ax1.axhline(y=y, color='white', linewidth=0.5, alpha=0.5)
    for j in range(tile_width + 1):
        x = j * tile_size
        ax1.axvline(x=x, color='white', linewidth=0.5, alpha=0.5)
    
    # Mark Gaussian centers
    for i, params in enumerate(gaussians_params):
        ax1.scatter(params['center'][0], params['center'][1], 
                   c=[params['color']], s=100, marker='x', linewidths=2)
        ax1.annotate(f"G{i+1}", params['center'], 
                    xytext=(5, 5), textcoords='offset points',
                    color='white', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(0, image_size)
    ax1.set_ylim(image_size, 0)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # -------------------------------------------------------------------------
    # Right plot: Tile intersection visualization
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    # Background: faded Gaussians
    faded_img = combined_img * 0.3
    ax2.imshow(faded_img, origin='upper')
    ax2.set_title('Tile-Gaussian Intersections', fontsize=12)
    
    # Draw tile borders with colors based on coverage
    if device.type == 'cuda' and tile_gaussian_map:
        for tile_id in range(n_tiles):
            tile_y = tile_id // tile_width
            tile_x = tile_id % tile_width
            
            x_start = tile_x * tile_size
            y_start = tile_y * tile_size
            
            if tile_id in tile_gaussian_map:
                gaussians_in_tile = tile_gaussian_map[tile_id]
                
                # Draw a colored border for each Gaussian that covers this tile
                # Stack the borders with slight offsets
                for offset_idx, gid in enumerate(gaussians_in_tile):
                    color = gaussians_params[gid]['color']
                    offset = offset_idx * 2  # Offset each border slightly
                    
                    rect = patches.Rectangle(
                        (x_start + offset, y_start + offset),
                        tile_size - 2 * offset,
                        tile_size - 2 * offset,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none',
                        alpha=0.9
                    )
                    ax2.add_patch(rect)
                
                # Fill tile with blended color
                avg_color = np.mean([gaussians_params[g]['color'] for g in gaussians_in_tile], axis=0)
                fill_rect = patches.Rectangle(
                    (x_start, y_start),
                    tile_size, tile_size,
                    linewidth=0,
                    facecolor=avg_color,
                    alpha=0.2
                )
                ax2.add_patch(fill_rect)
            else:
                # No Gaussian covers this tile - draw gray border
                rect = patches.Rectangle(
                    (x_start, y_start),
                    tile_size, tile_size,
                    linewidth=0.5,
                    edgecolor='gray',
                    facecolor='none',
                    alpha=0.3
                )
                ax2.add_patch(rect)
    else:
        # Draw simple grid if CUDA not available
        for i in range(tile_height + 1):
            y = i * tile_size
            ax2.axhline(y=y, color='gray', linewidth=0.5, alpha=0.5)
        for j in range(tile_width + 1):
            x = j * tile_size
            ax2.axvline(x=x, color='gray', linewidth=0.5, alpha=0.5)
    
    ax2.set_xlim(0, image_size)
    ax2.set_ylim(image_size, 0)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=params['color'], alpha=0.7, label=params['name'])
        for params in gaussians_params
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'tests/isect_tiles_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Also show the figure
    plt.show()
    
    # =========================================================================
    # Create detailed tile coverage table
    # =========================================================================
    
    if device.type == 'cuda':
        print("\n" + "=" * 60)
        print("Detailed Tile Coverage Summary")
        print("=" * 60)
        
        # Count tiles per Gaussian
        for gid, params in enumerate(gaussians_params):
            count = tiles_per_gauss[0, gid].item()
            print(f"{params['name']}: covers {count} tiles")
        
        # Create coverage matrix visualization
        print("\nCoverage Matrix (tile grid):")
        print("  - R: Covered by Gaussian 1 (Red)")
        print("  - G: Covered by Gaussian 2 (Green)")  
        print("  - B: Covered by Gaussian 3 (Blue)")
        print("  - .: Not covered")
        print()
        
        # Print header
        print("    ", end="")
        for j in range(tile_width):
            print(f" {j} ", end="")
        print()
        print("   +" + "---" * tile_width + "+")
        
        for i in range(tile_height):
            print(f" {i} |", end="")
            for j in range(tile_width):
                tile_id = i * tile_width + j
                if tile_id in tile_gaussian_map:
                    gaussians = tile_gaussian_map[tile_id]
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
        
        print("   +" + "---" * tile_width + "+")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    run_isect_tiles_demo()
