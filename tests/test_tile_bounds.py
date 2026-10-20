"""3D tile intersections must clip primitives that cross volume boundaries."""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "mean,radius,expected_ids",
    [
        pytest.param((0.5, 0.5, 0.5), 2, [0], id="negative-corner"),
        pytest.param((-1.0, 3.0, 5.0), 2, [0, 3, 6, 9], id="cross-negative-x"),
        pytest.param((5.0, -1.0, 5.0), 2, [0, 1, 6, 7], id="cross-negative-y"),
        pytest.param((5.0, 3.0, -1.0), 2, [0, 1, 3, 4], id="cross-negative-z"),
        pytest.param((-3.0, 2.0, 2.0), 2, [], id="outside-negative-x"),
        pytest.param((2.0, -3.0, 2.0), 2, [], id="outside-negative-y"),
        pytest.param((2.0, 2.0, -3.0), 2, [], id="outside-negative-z"),
        pytest.param((20.0, 2.0, 2.0), 2, [], id="outside-positive-x"),
        pytest.param((2.0, 20.0, 2.0), 2, [], id="outside-positive-y"),
        pytest.param((2.0, 2.0, 20.0), 2, [], id="outside-positive-z"),
        pytest.param((11.0, 7.0, 15.0), 2, [23], id="positive-corner"),
        pytest.param((6.0, 4.0, 8.0), 24, list(range(24)), id="covers-entire-grid"),
    ],
)
def test_3d_tile_bounds_clip_before_integer_conversion(dtype, mean, radius, expected_ids):
    from r2splat.cuda._wrapper import isect_offset_encode_3d, isect_tiles_3d

    means = torch.tensor([mean], dtype=dtype, device="cuda")
    radii = torch.full((1, 3), radius, dtype=torch.int32, device="cuda")
    # The x/y/z grid has 3/2/4 tiles and covers 12/8/16 voxels.
    counts, ids, primitive_ids = isect_tiles_3d(means, radii, 4, 3, 2, 4)
    assert counts.tolist() == [len(expected_ids)]
    assert ids.tolist() == expected_ids
    assert primitive_ids.tolist() == [0] * len(expected_ids)

    offsets = isect_offset_encode_3d(ids, 3, 2, 4)
    assert offsets.shape == (4, 2, 3)  # Tensor order is z, y, x.
    expected_offsets = torch.searchsorted(
        torch.tensor(expected_ids, device="cuda", dtype=torch.int64),
        torch.arange(24, device="cuda"),
    ).to(torch.int32).reshape(4, 2, 3)
    torch.testing.assert_close(offsets, expected_offsets)
