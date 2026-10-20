"""Fit Gaussian/Beta parameters with explicit VJPs: python examples/optimize.py."""

import torch
from torch.nn import functional as F

from r2splat import rasterize_bwd, rasterize_fwd, voxelize_bwd, voxelize_fwd


@torch.no_grad()
def fit(primitive):
    def tensor(values):
        return torch.tensor(values, device="cuda")

    viewmats = torch.eye(4, device="cuda")[None]
    Ks = tensor([[[25.0, 0.0, 10.0], [0.0, 25.0, 8.0], [0.0, 0.0, 1.0]]])

    def render(means, quats, scales, densities, beta):
        image, image_ctx = rasterize_fwd(
            means, quats, scales, densities, viewmats, Ks,
            image_size=(16, 20), beta=beta,
        )
        volume, volume_ctx = voxelize_fwd(
            means, quats, scales, densities, volume_shape=(10, 12, 14),
            volume_extent=(1.4, 1.2, 1.0), volume_center=(0.0, 0.0, 2.0),
            beta=beta,
        )
        return (image, volume), (image_ctx, volume_ctx)

    # Fixed synthetic targets; their contexts are not needed.
    targets, _ = render(
        tensor([[0.05, 0.02, 2.0]]), tensor([[1.3, 0.2, -0.15, 0.25]]),
        tensor([[0.24, 0.31, 0.39]]), tensor([1.3]),
        tensor([3.5]) if primitive == "beta" else None,
    )
    params = {
        "means": torch.nn.Parameter(tensor([[0.10, 0.08, 2.04]])),
        "quats": torch.nn.Parameter(tensor([[1.2, -0.1, 0.2, 0.1]])),
        "raw_scales": torch.nn.Parameter(tensor([[0.29, 0.36, 0.32]]).log()),
        "raw_densities": torch.nn.Parameter(tensor([0.4])),
    }
    if primitive == "beta":
        params["raw_beta"] = torch.nn.Parameter(tensor([2.8]).log())
    before = {name: value.clone() for name, value in params.items()}
    optimizer = torch.optim.Adam(params.values(), lr=0.02)

    def predict():
        scales = params["raw_scales"].exp()
        densities = F.softplus(params["raw_densities"])
        beta = params["raw_beta"].exp() if primitive == "beta" else None
        outputs, contexts = render(params["means"], params["quats"], scales, densities, beta)
        return outputs, contexts, scales, beta

    def loss_fn(outputs):
        return sum((output - target).square().mean() for output, target in zip(outputs, targets))

    initial_loss = None
    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        outputs, contexts, scales, beta = predict()
        loss = loss_fn(outputs)
        assert bool(torch.isfinite(loss))
        assert all(output.grad_fn is None for output in outputs)
        if step == 0:
            initial_loss = loss.item()
        # d mean((prediction - target)**2) / d prediction, without autograd.
        upstream = [2 * (output - target) / output.numel()
                    for output, target in zip(outputs, targets)]
        image_grads = rasterize_bwd(contexts[0], upstream[0])
        volume_grads = voxelize_bwd(contexts[1], upstream[1])
        params["means"].grad = image_grads.means + volume_grads.means
        params["quats"].grad = image_grads.quats + volume_grads.quats
        # Sum physical VJPs, then apply the caller-owned activation derivatives.
        params["raw_scales"].grad = (image_grads.scales + volume_grads.scales) * scales
        params["raw_densities"].grad = (
            image_grads.densities + volume_grads.densities
        ) * params["raw_densities"].sigmoid()
        if beta is not None:
            params["raw_beta"].grad = (image_grads.beta + volume_grads.beta) * beta
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in params.values())
        if step == 0:
            assert bool(params["quats"].grad.abs().sum() > 0)
        # Both contexts have been consumed; parameters may now be changed.
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in params.values())

    final_outputs, _, _, _ = predict()
    final_loss = loss_fn(final_outputs).item()
    changes = {name: (value - before[name]).norm().item() for name, value in params.items()}
    assert final_loss < initial_loss, "Optimization did not reduce the loss"
    assert all(change > 0 for change in changes.values()), "A parameter did not update"
    print(f"{primitive}: loss {initial_loss:.6g} -> {final_loss:.6g}")
    print("  parameter changes (L2): " + ", ".join(f"{name}={delta:.4g}" for name, delta in changes.items()))


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This example requires a CUDA-enabled PyTorch installation and GPU.")
    for primitive in ("gaussian", "beta"):
        fit(primitive)


if __name__ == "__main__":
    main()
