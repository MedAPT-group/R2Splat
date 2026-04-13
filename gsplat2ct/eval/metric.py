"""Image and volume quality metrics for CT reconstruction.

All functions are pure torch operations with no side effects.
"""

from math import exp as _exp
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------

def _gaussian_kernel_1d(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    g = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_window(
    window_size: int, channel: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(window_size).unsqueeze(1)
    k2d = (k1d @ k1d.t()).unsqueeze(0).unsqueeze(0)
    return k2d.expand(channel, 1, window_size, window_size).contiguous().to(
        device=device, dtype=dtype
    )


# ------------------------------------------------------------------
# Core metrics
# ------------------------------------------------------------------

def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Peak signal-to-noise ratio from prediction and target tensors."""
    mse = torch.mean((pred - target) ** 2).clamp(min=eps)
    return 10.0 * torch.log10(torch.tensor(data_range ** 2, device=mse.device) / mse)


def ssim2d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Mean SSIM for a batch of images shaped ``(N, C, H, W)``."""
    assert img1.shape == img2.shape
    _, channel, _, _ = img1.shape
    window = _create_window(window_size, channel, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu12

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def ssim3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """Approximate 3-D SSIM by averaging 2-D SSIM over slices from all axes.

    Inputs: ``(W, D, H)`` or ``(N, W, D, H)``.  Empty GT slices are skipped.
    """
    assert pred.shape == target.shape
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    sample_scores = []
    for b in range(pred.shape[0]):
        pv, tv = pred[b], target[b]
        axis_scores = []
        for axis in range(3):
            slice_scores = []
            for i in range(tv.shape[axis]):
                ts = tv.select(axis, i)
                ps = pv.select(axis, i)
                if ts.max() > 0:
                    s = ssim2d(
                        ps.unsqueeze(0).unsqueeze(0),
                        ts.unsqueeze(0).unsqueeze(0),
                        window_size=window_size,
                    )
                    slice_scores.append(s)
            if slice_scores:
                axis_scores.append(torch.stack(slice_scores).mean())
        if axis_scores:
            sample_scores.append(torch.stack(axis_scores).mean())
        else:
            sample_scores.append(torch.zeros((), device=pv.device))
    return torch.stack(sample_scores).mean()


# ------------------------------------------------------------------
# Aggregate metrics
# ------------------------------------------------------------------

def metric_proj(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Per-slice-normalised 2-D metrics for projection images.

    Inputs: ``(N, W, H)`` — batch of single-channel projections.
    """
    assert pred.shape == target.shape and pred.ndim == 3
    psnr_list, ssim_list = [], []
    for i in range(target.shape[0]):
        ts, ps = target[i], pred[i]
        if ts.max() <= 0:
            continue
        t_max = ts.max().clamp_min(1e-12)
        p_max = ps.max().clamp_min(1e-12)
        tn = (ts / t_max).unsqueeze(0).unsqueeze(0)
        pn = (ps / p_max).unsqueeze(0).unsqueeze(0)
        psnr_list.append(psnr(pn, tn, data_range=data_range, eps=0.0))
        ssim_list.append(ssim2d(pn, tn, window_size=window_size))

    zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
    return {
        "l1": (pred - target).abs().mean(),
        "mse": ((pred - target) ** 2).mean(),
        "psnr": torch.stack(psnr_list).mean() if psnr_list else zero,
        "ssim": torch.stack(ssim_list).mean() if ssim_list else zero,
    }


def metric_vol(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: Optional[float] = 1.0,
) -> Dict[str, torch.Tensor]:
    """Volume reconstruction quality metrics.

    Inputs: ``(W, D, H)`` or ``(N, W, D, H)``.
    """
    assert pred.shape == target.shape
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    l1v = (pred - target).abs().mean()
    mse = ((pred - target) ** 2).mean()
    dr = float(target.max().clamp_min(1e-12).item()) if data_range is None else data_range
    psnr_val = psnr(pred, target, data_range=dr)
    ssim_val = ssim3d(pred, target, window_size=window_size)
    return {"l1": l1v, "mse": mse, "psnr": psnr_val, "ssim": ssim_val}
