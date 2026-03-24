"""
Test Isect Tiles 3D CUDA Implementation

This test validates the 3D tile-Gaussian intersection computation.

Author: Dy164
Date: 2026-02-02
"""

import torch
import pytest
import math


def test_isect_tiles_3d_basic():
    """Test basic isect_tiles_3d functionality."""
    from gsplat2ct.cuda import isect_tiles_3d, isect_offset_encode_3d
    
    # Setup
    N = 100  # Number of Gaussians
    tile_size = 16
    tile_width = 8   # x dimension
    tile_depth = 8   # y dimension
    tile_height = 6  # z dimension
    
    # Create test data: Gaussians spread across the volume
    # Volume size: 8 * 16 = 128 x 8 * 16 = 128 x 6 * 16 = 96
    torch.manual_seed(42)
    means3d = torch.rand(N, 3, device='cuda', dtype=torch.float32)
    means3d[:, 0] *= tile_width * tile_size   # x: 0 to 128
    means3d[:, 1] *= tile_depth * tile_size   # y: 0 to 128
    means3d[:, 2] *= tile_height * tile_size  # z: 0 to 96
    
    radii = torch.randint(8, 32, (N, 3), device='cuda', dtype=torch.int32)
    
    # Run isect_tiles_3d
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        means3d, radii, tile_size, tile_width, tile_depth, tile_height
    )
    
    # Validate output shapes
    assert tiles_per_gauss.shape == (N,), f"Expected ({N},), got {tiles_per_gauss.shape}"
    assert tiles_per_gauss.dtype == torch.int32
    
    # Total intersections should match
    n_isects = tiles_per_gauss.sum().item()
    assert isect_ids.shape[0] == n_isects, f"isect_ids length mismatch"
    assert flatten_ids.shape[0] == n_isects, f"flatten_ids length mismatch"
    
    print(f"Total intersections: {n_isects}")
    print(f"Average tiles per Gaussian: {n_isects / N:.2f}")
    
    # Validate isect_ids are sorted
    if n_isects > 1:
        diff = isect_ids[1:] - isect_ids[:-1]
        assert (diff >= 0).all(), "isect_ids should be sorted"
    
    # Test offset encoding
    offsets = isect_offset_encode_3d(isect_ids, tile_width, tile_depth, tile_height)
    assert offsets.shape == (tile_height, tile_depth, tile_width), f"offsets shape mismatch"
    assert offsets.dtype == torch.int32
    
    print("✓ test_isect_tiles_3d_basic passed")


def test_isect_tiles_3d_empty():
    """Test isect_tiles_3d with zero-radius Gaussians (no intersections)."""
    from gsplat2ct.cuda import isect_tiles_3d
    
    N = 50
    tile_size, tile_width, tile_depth, tile_height = 16, 8, 8, 6
    
    means3d = torch.rand(N, 3, device='cuda', dtype=torch.float32) * 100
    radii = torch.zeros(N, 3, device='cuda', dtype=torch.int32)  # All zero radii
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        means3d, radii, tile_size, tile_width, tile_depth, tile_height
    )
    
    assert tiles_per_gauss.sum().item() == 0, "Should have no intersections"
    assert isect_ids.shape[0] == 0
    assert flatten_ids.shape[0] == 0
    
    print("✓ test_isect_tiles_3d_empty passed")


def test_isect_tiles_3d_single_tile():
    """Test with Gaussians that each cover exactly one tile."""
    from gsplat2ct.cuda import isect_tiles_3d
    
    N = 10
    tile_size = 16
    tile_width, tile_depth, tile_height = 4, 4, 4
    
    # Place Gaussians at tile centers with small radii
    means3d = torch.zeros(N, 3, device='cuda', dtype=torch.float32)
    for i in range(N):
        # Each Gaussian at center of a different tile
        tile_x = i % tile_width
        tile_y = (i // tile_width) % tile_depth
        tile_z = (i // (tile_width * tile_depth)) % tile_height
        means3d[i, 0] = (tile_x + 0.5) * tile_size
        means3d[i, 1] = (tile_y + 0.5) * tile_size
        means3d[i, 2] = (tile_z + 0.5) * tile_size
    
    radii = torch.ones(N, 3, device='cuda', dtype=torch.int32)  # Radius 1 voxel
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        means3d, radii, tile_size, tile_width, tile_depth, tile_height
    )
    
    # Each Gaussian should cover exactly 1 tile
    assert (tiles_per_gauss == 1).all(), "Each Gaussian should cover 1 tile"
    assert isect_ids.shape[0] == N
    
    print("✓ test_isect_tiles_3d_single_tile passed")


def test_isect_tiles_3d_sorting():
    """Verify that sorting orders by tile_id."""
    from gsplat2ct.cuda import isect_tiles_3d
    
    N = 200
    tile_size = 16
    tile_width, tile_depth, tile_height = 8, 8, 6
    n_tiles = tile_width * tile_depth * tile_height
    
    torch.manual_seed(123)
    means3d = torch.rand(N, 3, device='cuda', dtype=torch.float32)
    means3d[:, 0] *= tile_width * tile_size
    means3d[:, 1] *= tile_depth * tile_size
    means3d[:, 2] *= tile_height * tile_size
    radii = torch.randint(10, 50, (N, 3), device='cuda', dtype=torch.int32)
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        means3d, radii, tile_size, tile_width, tile_depth, tile_height
    )
    
    n_isects = isect_ids.shape[0]
    if n_isects > 0:
        # Verify ordering: isect_ids should be sorted
        for i in range(1, n_isects):
            assert isect_ids[i] >= isect_ids[i-1], f"Tile order violated at {i}"
    
    print("✓ test_isect_tiles_3d_sorting passed")


def test_isect_offset_encode_3d():
    """Test offset encoding produces correct per-tile access indices."""
    from gsplat2ct.cuda import isect_tiles_3d, isect_offset_encode_3d
    
    N = 100
    tile_size = 16
    tile_width, tile_depth, tile_height = 4, 4, 3
    n_tiles = tile_width * tile_depth * tile_height
    
    torch.manual_seed(456)
    means3d = torch.rand(N, 3, device='cuda', dtype=torch.float32)
    means3d[:, 0] *= tile_width * tile_size
    means3d[:, 1] *= tile_depth * tile_size
    means3d[:, 2] *= tile_height * tile_size
    radii = torch.randint(5, 20, (N, 3), device='cuda', dtype=torch.int32)
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        means3d, radii, tile_size, tile_width, tile_depth, tile_height
    )
    
    offsets = isect_offset_encode_3d(isect_ids, tile_width, tile_depth, tile_height)
    
    # Verify offsets
    # offsets should be non-decreasing and within [0, n_isects]
    flat_offsets = offsets.view(-1).cpu()
    assert (flat_offsets[1:] >= flat_offsets[:-1]).all(), "Offsets should be non-decreasing"
    assert flat_offsets[0] >= 0
    assert flat_offsets[-1] <= isect_ids.shape[0]
    
    # Verify we can use offsets to access Gaussians for each tile
    n_isects = isect_ids.shape[0]
    if n_isects > 0:
        for tz in range(tile_height):
            for ty in range(tile_depth):
                for tx in range(tile_width):
                    tile_id = tz * tile_depth * tile_width + ty * tile_width + tx
                    expected_isect_id = tile_id
                    
                    start = offsets[tz, ty, tx].item()
                    # Get end from next tile or end of array
                    if tx + 1 < tile_width:
                        end = offsets[tz, ty, tx + 1].item()
                    elif ty + 1 < tile_depth:
                        end = offsets[tz, ty + 1, 0].item()
                    elif tz + 1 < tile_height:
                        end = offsets[tz + 1, 0, 0].item()
                    else:
                        end = n_isects
                    
                    # All isect_ids in [start, end) should have this tile_id
                    for idx in range(start, min(end, start + 5)):  # Check first few
                        assert isect_ids[idx].item() == expected_isect_id, \
                            f"Mismatch at tile ({tz}, {ty}, {tx})"
    
    print("✓ test_isect_offset_encode_3d passed")


def test_isect_tiles_3d_correctness():
    """Verify intersection computation against CPU reference."""
    from gsplat2ct.cuda import isect_tiles_3d
    
    N = 20
    tile_size = 16
    tile_width, tile_depth, tile_height = 4, 4, 3
    
    # Create simple test case
    means3d = torch.zeros(N, 3, device='cuda', dtype=torch.float32)
    
    # Spread Gaussians
    for i in range(N):
        means3d[i, 0] = (i % tile_width + 0.5) * tile_size
        means3d[i, 1] = ((i // tile_width) % tile_depth + 0.5) * tile_size
        means3d[i, 2] = ((i // (tile_width * tile_depth)) % tile_height + 0.5) * tile_size
    
    radii = torch.full((N, 3), tile_size // 2, device='cuda', dtype=torch.int32)
    
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        means3d, radii, tile_size, tile_width, tile_depth, tile_height
    )
    
    # CPU reference computation
    tiles_per_gauss_ref = torch.zeros(N, dtype=torch.int32)
    for n in range(N):
        x, y, z = means3d[n].cpu().numpy()
        rx, ry, rz = radii[n].cpu().numpy()
        
        tile_x = x / tile_size
        tile_y = y / tile_size
        tile_z = z / tile_size
        tile_r_x = rx / tile_size
        tile_r_y = ry / tile_size
        tile_r_z = rz / tile_size
        
        tile_min_x = max(0, int(math.floor(tile_x - tile_r_x)))
        tile_min_y = max(0, int(math.floor(tile_y - tile_r_y)))
        tile_min_z = max(0, int(math.floor(tile_z - tile_r_z)))
        tile_max_x = min(tile_width, int(math.ceil(tile_x + tile_r_x)))
        tile_max_y = min(tile_depth, int(math.ceil(tile_y + tile_r_y)))
        tile_max_z = min(tile_height, int(math.ceil(tile_z + tile_r_z)))
        
        tiles_per_gauss_ref[n] = (
            (tile_max_z - tile_min_z) * 
            (tile_max_y - tile_min_y) * 
            (tile_max_x - tile_min_x)
        )
    
    # Compare
    torch.testing.assert_close(
        tiles_per_gauss.cpu(), tiles_per_gauss_ref,
        msg="tiles_per_gauss mismatch with CPU reference"
    )
    
    print("✓ test_isect_tiles_3d_correctness passed")


def test_isect_tiles_3d_large():
    """Test with a larger number of Gaussians for performance."""
    from gsplat2ct.cuda import isect_tiles_3d, isect_offset_encode_3d
    
    N = 10000
    tile_size = 8
    tile_width, tile_depth, tile_height = 32, 32, 24
    
    torch.manual_seed(999)
    means3d = torch.rand(N, 3, device='cuda', dtype=torch.float32)
    means3d[:, 0] *= tile_width * tile_size
    means3d[:, 1] *= tile_depth * tile_size
    means3d[:, 2] *= tile_height * tile_size
    radii = torch.randint(4, 16, (N, 3), device='cuda', dtype=torch.int32)
    
    # Warm up
    for _ in range(3):
        isect_tiles_3d(means3d, radii, tile_size, tile_width, tile_depth, tile_height)
    
    torch.cuda.synchronize()
    
    import time
    start = time.time()
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_3d(
        means3d, radii, tile_size, tile_width, tile_depth, tile_height
    )
    offsets = isect_offset_encode_3d(isect_ids, tile_width, tile_depth, tile_height)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    n_isects = tiles_per_gauss.sum().item()
    print(f"N={N} Gaussians, {n_isects} intersections, time={elapsed*1000:.2f}ms")
    
    # Basic validation
    assert tiles_per_gauss.shape == (N,)
    assert isect_ids.shape[0] == n_isects
    assert offsets.shape == (tile_height, tile_depth, tile_width)
    
    print("✓ test_isect_tiles_3d_large passed")


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)
    
    print("Running isect_tiles_3d tests...\n")
    
    test_isect_tiles_3d_basic()
    test_isect_tiles_3d_empty()
    test_isect_tiles_3d_single_tile()
    test_isect_tiles_3d_sorting()
    test_isect_offset_encode_3d()
    test_isect_tiles_3d_correctness()
    test_isect_tiles_3d_large()
    
    print("\n" + "=" * 50)
    print("All isect_tiles_3d tests passed!")
