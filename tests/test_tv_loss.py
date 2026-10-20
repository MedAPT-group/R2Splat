import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_volume_tv_matches_edge_mean_reference():
    from r2splat.cuda._wrapper import volume_loss_bwd, volume_loss_fwd

    torch.manual_seed(0)
    volume = torch.randn(1, 4, 5, 6, device="cuda", requires_grad=True)
    lambda_tv = 0.05

    loss_cuda, tv_cuda, _tv_map = volume_loss_fwd(volume.detach(), lambda_tv)

    diffs = [
        volume[:, 1:] - volume[:, :-1],
        volume[:, :, 1:] - volume[:, :, :-1],
        volume[:, :, :, 1:] - volume[:, :, :, :-1],
    ]
    edge_count = sum(diff.numel() for diff in diffs)
    tv_ref = sum(diff.abs().sum() for diff in diffs) / edge_count
    loss_ref = lambda_tv * tv_ref
    loss_ref.backward()

    grad_cuda = volume_loss_bwd(
        volume.detach(), torch.ones(1, device="cuda"), lambda_tv
    )

    assert torch.allclose(tv_cuda, tv_ref.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(loss_cuda, loss_ref.detach(), atol=1e-6, rtol=1e-6)
    assert torch.allclose(grad_cuda, volume.grad, atol=1e-6, rtol=1e-5)
