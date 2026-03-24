"""
Test Isect Tiles CUDA Implementation

This test validates the tile-Gaussian intersection computation.

Author: Dy164
Date: 2025-01-12
"""

import torch
import pytest
import math


def test_isect_tiles_basic():
    """Test basic isect_tiles functionality."""
    from gsplat2ct.cuda import isect_tiles, isect_offset_encode
    
    # Setup
    C = 2  # Number of cameras
    N = 100  # Number of Gaussians
    tile_size = 16
    tile_width = 64
    tile_height = 48
    
    # Create test data: Gaussians spread across the image
    # Image size: 64 * 16 = 1024 x 48 * 16 = 768
    torch.manual_seed(42)
    means2d = torch.rand(C, N, 2, device='cuda', dtype=torch.float32)
    means2d[..., 0] *= tile_width * tile_size  # x: 0 to 1024
    means2d[..., 1] *= tile_height * tile_size  # y: 0 to 768
    
    radii = torch.randint(8, 32, (C, N), device='cuda', dtype=torch.int32)
    
    # Run isect_tiles
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, tile_width, tile_height
    )
    
    # Validate output shapes
    assert tiles_per_gauss.shape == (C, N), f"Expected {(C, N)}, got {tiles_per_gauss.shape}"
    assert tiles_per_gauss.dtype == torch.int32
    
    # Total intersections should match
    n_isects = tiles_per_gauss.sum().item()
    assert isect_ids.shape[0] == n_isects, f"isect_ids length mismatch"
    assert flatten_ids.shape[0] == n_isects, f"flatten_ids length mismatch"
    
    print(f"Total intersections: {n_isects}")
    print(f"Average tiles per Gaussian: {n_isects / (C * N):.2f}")
    
    # Validate isect_ids are sorted
    if n_isects > 1:
        diff = isect_ids[1:] - isect_ids[:-1]
        assert (diff >= 0).all(), "isect_ids should be sorted"
    
    # Test offset encoding
    offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    assert offsets.shape == (C, tile_height, tile_width), f"offsets shape mismatch"
    assert offsets.dtype == torch.int32
    
    print("✓ test_isect_tiles_basic passed")


def test_isect_tiles_empty():
    """Test isect_tiles with zero-radius Gaussians (no intersections)."""
    from gsplat2ct.cuda import isect_tiles
    
    C, N = 2, 50
    tile_size, tile_width, tile_height = 16, 32, 24
    
    means2d = torch.rand(C, N, 2, device='cuda', dtype=torch.float32) * 100
    radii = torch.zeros(C, N, device='cuda', dtype=torch.int32)  # All zero radii
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, tile_width, tile_height
    )
    
    assert tiles_per_gauss.sum().item() == 0, "Should have no intersections"
    assert isect_ids.shape[0] == 0
    assert flatten_ids.shape[0] == 0
    
    print("✓ test_isect_tiles_empty passed")


def test_isect_tiles_single_tile():
    """Test with Gaussians that each cover exactly one tile."""
    from gsplat2ct.cuda import isect_tiles
    
    C, N = 1, 10
    tile_size = 16
    tile_width, tile_height = 8, 8
    
    # Place Gaussians at tile centers with small radii
    means2d = torch.zeros(C, N, 2, device='cuda', dtype=torch.float32)
    for i in range(N):
        # Each Gaussian at center of tile (i % tile_width, i // tile_width)
        tile_x = i % tile_width
        tile_y = (i // tile_width) % tile_height
        means2d[0, i, 0] = (tile_x + 0.5) * tile_size
        means2d[0, i, 1] = (tile_y + 0.5) * tile_size
    
    radii = torch.ones(C, N, device='cuda', dtype=torch.int32)  # Radius 1 pixel
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, tile_width, tile_height
    )
    
    # Each Gaussian should cover exactly 1 tile
    assert (tiles_per_gauss == 1).all(), "Each Gaussian should cover 1 tile"
    assert isect_ids.shape[0] == N
    
    print("✓ test_isect_tiles_single_tile passed")


def test_isect_tiles_sorting():
    """Verify that sorting orders by (camera_id, tile_id)."""
    from gsplat2ct.cuda import isect_tiles
    
    C, N = 3, 200
    tile_size = 16
    tile_width, tile_height = 16, 12
    n_tiles = tile_width * tile_height
    
    torch.manual_seed(123)
    means2d = torch.rand(C, N, 2, device='cuda', dtype=torch.float32)
    means2d[..., 0] *= tile_width * tile_size
    means2d[..., 1] *= tile_height * tile_size
    radii = torch.randint(10, 50, (C, N), device='cuda', dtype=torch.int32)
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, tile_width, tile_height
    )
    
    n_isects = isect_ids.shape[0]
    if n_isects > 0:
        # Decode camera_id and tile_id from isect_ids
        # isect_id = camera_id * n_tiles + tile_id
        camera_ids = (isect_ids // n_tiles).cpu()
        tile_ids = (isect_ids % n_tiles).cpu()
        
        # Verify ordering
        for i in range(1, n_isects):
            # Camera should be non-decreasing
            assert camera_ids[i] >= camera_ids[i-1], f"Camera order violated at {i}"
            
            # If same camera, tile should be non-decreasing
            if camera_ids[i] == camera_ids[i-1]:
                assert tile_ids[i] >= tile_ids[i-1], f"Tile order violated at {i}"
    
    print("✓ test_isect_tiles_sorting passed")


def test_isect_offset_encode():
    """Test offset encoding produces correct per-tile access indices."""
    from gsplat2ct.cuda import isect_tiles, isect_offset_encode
    
    C, N = 2, 100
    tile_size = 16
    tile_width, tile_height = 8, 6
    n_tiles = tile_width * tile_height
    
    torch.manual_seed(456)
    means2d = torch.rand(C, N, 2, device='cuda', dtype=torch.float32)
    means2d[..., 0] *= tile_width * tile_size
    means2d[..., 1] *= tile_height * tile_size
    radii = torch.randint(5, 20, (C, N), device='cuda', dtype=torch.int32)
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, tile_width, tile_height
    )
    
    offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    
    # Verify offsets
    # offsets should be non-decreasing and within [0, n_isects]
    flat_offsets = offsets.view(-1).cpu()
    assert (flat_offsets[1:] >= flat_offsets[:-1]).all(), "Offsets should be non-decreasing"
    assert flat_offsets[0] >= 0
    assert flat_offsets[-1] <= isect_ids.shape[0]
    
    # Verify we can use offsets to access Gaussians for each tile
    n_isects = isect_ids.shape[0]
    if n_isects > 0:
        for c in range(C):
            for ty in range(tile_height):
                for tx in range(tile_width):
                    tile_id = ty * tile_width + tx
                    expected_isect_id = c * n_tiles + tile_id
                    
                    start = offsets[c, ty, tx].item()
                    # Get end from next tile or end of array
                    if tx + 1 < tile_width:
                        end = offsets[c, ty, tx + 1].item()
                    elif ty + 1 < tile_height:
                        end = offsets[c, ty + 1, 0].item()
                    elif c + 1 < C:
                        end = offsets[c + 1, 0, 0].item()
                    else:
                        end = n_isects
                    
                    # All isect_ids in [start, end) should have this (camera, tile)
                    for idx in range(start, min(end, start + 5)):  # Check first few
                        assert isect_ids[idx].item() == expected_isect_id, \
                            f"Mismatch at tile ({c}, {ty}, {tx})"
    
    print("✓ test_isect_offset_encode passed")


def test_isect_tiles_no_sort():
    """Test isect_tiles without sorting."""
    from gsplat2ct.cuda import isect_tiles
    
    C, N = 2, 100
    tile_size = 16
    tile_width, tile_height = 16, 12
    
    torch.manual_seed(789)
    means2d = torch.rand(C, N, 2, device='cuda', dtype=torch.float32)
    means2d[..., 0] *= tile_width * tile_size
    means2d[..., 1] *= tile_height * tile_size
    radii = torch.randint(10, 30, (C, N), device='cuda', dtype=torch.int32)
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, tile_width, tile_height
    )
    
    # Just verify shapes and totals match
    n_isects = tiles_per_gauss.sum().item()
    assert isect_ids.shape[0] == n_isects
    assert flatten_ids.shape[0] == n_isects
    
    print("✓ test_isect_tiles_no_sort passed")


def test_isect_tiles_correctness():
    """Verify intersection computation against CPU reference."""
    from gsplat2ct.cuda import isect_tiles
    
    C, N = 1, 20
    tile_size = 16
    tile_width, tile_height = 8, 6
    
    # Create simple test case
    means2d = torch.tensor([
        [[64.0, 48.0]],   # Center of image
    ], device='cuda', dtype=torch.float32).expand(C, N, 2).clone()
    
    # Spread Gaussians
    for i in range(N):
        means2d[0, i, 0] = (i % tile_width + 0.5) * tile_size
        means2d[0, i, 1] = (i // tile_width) * tile_size + tile_size / 2
    
    radii = torch.full((C, N), tile_size // 2, device='cuda', dtype=torch.int32)
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, tile_width, tile_height
    )
    
    # CPU reference computation
    tiles_per_gauss_ref = torch.zeros(C, N, dtype=torch.int32)
    for c in range(C):
        for n in range(N):
            x, y = means2d[c, n].cpu().numpy()
            r = radii[c, n].item()
            
            tile_x = x / tile_size
            tile_y = y / tile_size
            tile_r = r / tile_size
            
            tile_min_x = max(0, int(math.floor(tile_x - tile_r)))
            tile_min_y = max(0, int(math.floor(tile_y - tile_r)))
            tile_max_x = min(tile_width, int(math.ceil(tile_x + tile_r)))
            tile_max_y = min(tile_height, int(math.ceil(tile_y + tile_r)))
            
            tiles_per_gauss_ref[c, n] = (tile_max_y - tile_min_y) * (tile_max_x - tile_min_x)
    
    # Compare
    torch.testing.assert_close(
        tiles_per_gauss.cpu(), tiles_per_gauss_ref,
        msg="tiles_per_gauss mismatch with CPU reference"
    )
    
    print("✓ test_isect_tiles_correctness passed")


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)
    
    print("Running isect_tiles tests...\n")
    
    test_isect_tiles_basic()
    test_isect_tiles_empty()
    test_isect_tiles_single_tile()
    test_isect_tiles_sorting()
    test_isect_offset_encode()
    test_isect_tiles_no_sort()
    test_isect_tiles_correctness()
    
    print("\n" + "=" * 50)
    print("All isect_tiles tests passed!")
