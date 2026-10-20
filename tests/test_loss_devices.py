"""Loss kernels must use the input device, then restore the caller's device."""
import pytest
import torch

from r2splat.cuda._wrapper import (
    image_loss_fwd, image_loss_bwd, volume_loss_fwd, volume_loss_bwd,
)


pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA devices required')


@pytest.mark.parametrize('kind', ['image', 'volume'])
def test_loss_uses_input_device_and_stream(kind):
    # CUDA_VISIBLE_DEVICES can select any two physical GPUs for this test.
    input_device = torch.device('cuda:1')
    stream = torch.cuda.Stream(device=input_device)
    with torch.cuda.device(input_device), torch.cuda.stream(stream):
        if kind == 'image':
            value = torch.tensor([[[-2., 1.], [3., -4.]]], device=input_device, requires_grad=True)
            target = torch.zeros_like(value)
            reference = 0.25 * (value - target).abs().mean()
        else:
            value = torch.arange(8., device=input_device).reshape(1, 2, 2, 2).requires_grad_()
            differences = [value[:, 1:] - value[:, :-1],
                           value[:, :, 1:] - value[:, :, :-1],
                           value[:, :, :, 1:] - value[:, :, :, :-1]]
            reference = 0.25 * sum(d.abs().sum() for d in differences) / sum(d.numel() for d in differences)
        upstream = torch.ones(1, device=input_device)
        expected_gradient = torch.autograd.grad(reference, value)[0]
        with torch.cuda.device(0):
            assert torch.cuda.current_device() == 0
            if kind == 'image':
                actual, _, _ = image_loss_fwd(value.detach(), target, 0.25)
                gradient = image_loss_bwd(value.detach(), target, upstream, 0.25)
            else:
                actual, _, _ = volume_loss_fwd(value.detach(), 0.25)
                gradient = volume_loss_bwd(value.detach(), upstream, 0.25)
            assert torch.cuda.current_device() == 0
        torch.testing.assert_close(actual, reference.detach())
        torch.testing.assert_close(gradient, expected_gradient)
    stream.synchronize()


@pytest.mark.parametrize('kind', ['image_target', 'image_gradient', 'volume_gradient'])
def test_loss_rejects_tensors_on_different_devices(kind):
    value = torch.ones((1, 2, 2), device='cuda:1')
    wrong_device = torch.zeros_like(value, device='cuda:0')
    with pytest.raises(RuntimeError, match='same CUDA device'):
        if kind == 'image_target':
            image_loss_fwd(value, wrong_device, 1.)
        elif kind == 'image_gradient':
            image_loss_bwd(value, torch.zeros_like(value), wrong_device.reshape(-1)[:1], 1.)
        else:
            volume_loss_bwd(value[None], wrong_device.reshape(-1)[:1], 1.)
