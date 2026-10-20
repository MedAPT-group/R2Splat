"""Explicit Gaussian/Beta forward and VJP: python examples/basic.py."""

import torch

from r2splat import rasterize_bwd, rasterize_fwd, voxelize_bwd, voxelize_fwd


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-enabled PyTorch installation and GPU.")

    for primitive in ("gaussian", "beta"):
        means = torch.tensor([[0.05, 0.0, 2.0]], device="cuda")
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
        scales = torch.full((1, 3), 0.2, device="cuda")
        densities = torch.ones(1, device="cuda")
        viewmats = torch.eye(4, device="cuda")[None]
        Ks = torch.tensor([[[40.0, 0.0, 16.0], [0.0, 40.0, 12.0],
                            [0.0, 0.0, 1.0]]], device="cuda")
        beta = None if primitive == "gaussian" else torch.full((1,), 3.5, device="cuda")
        uncertainty = torch.full((1,), 0.1, device="cuda")

        (image, image_variance), image_ctx = rasterize_fwd(
            means, quats, scales, densities, viewmats, Ks,
            image_size=(24, 32), beta=beta, uncertainty=uncertainty,
        )
        (volume, volume_variance), volume_ctx = voxelize_fwd(
            means, quats, scales, densities, volume_shape=(16, 20, 24),
            volume_extent=(1.2, 1.0, 0.8), volume_center=(0.0, 0.0, 2.0),
            beta=beta, uncertainty=uncertainty,
        )
        # MSE against zero, plus mean variance. Supply both upstream gradients.
        loss = image.square().mean() + volume.square().mean()
        loss = loss + image_variance.mean() + volume_variance.mean()
        image_grads = rasterize_bwd(
            image_ctx, 2 * image / image.numel(),
            torch.full_like(image_variance, 1 / image_variance.numel()),
        )
        volume_grads = voxelize_bwd(
            volume_ctx, 2 * volume / volume.numel(),
            torch.full_like(volume_variance, 1 / volume_variance.numel()),
        )
        # Shared parameters receive the sum of the two operator VJPs.
        names = ["means", "quats", "scales", "densities", "uncertainty"]
        if beta is not None:
            names.append("beta")
        for name in names:
            gradient = getattr(image_grads, name) + getattr(volume_grads, name)
            assert bool(torch.isfinite(gradient).all()), name
        assert all(value.grad_fn is None for value in (image, volume, loss))
        print(f"{primitive}: image={tuple(image.shape)}, volume={tuple(volume.shape)}, "
              f"loss={loss.item():.6f}; explicit parameter gradients are finite")


if __name__ == "__main__":
    main()
