"""Video visualization helpers for training-time projection tracking."""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from gsplat2ct.data.ctproj import CTImageDataset
from gsplat2ct.ops.data import Config, Data, Params
from gsplat2ct.ops.gs3d_ct_rasterize import GS3DCTRasterizeCUDA
from gsplat2ct.vis.plotting import save_projection_triplet


class _ParamWrapper:
    def __init__(self, data: torch.Tensor):
        self.data = data


@dataclass
class VideoVisConfig:
    enabled: bool = False
    interval: int = 5
    view_index: int = 0
    fps: int = 10
    value_min: float = 0.0
    value_max: float = 1.0
    frame_subdir: str = "video_frames"
    output_name: str = "projection_progress.mp4"


class ProjectionVideoRecorder:
    """Record a fixed-view projection during training and export mp4."""

    def __init__(
        self,
        *,
        vis_cfg: VideoVisConfig,
        dataset: CTImageDataset,
        device: torch.device,
        out_dir: str,
        scale_min: float,
        scale_max: float,
        eps_blur: float,
    ):
        self.cfg = vis_cfg
        self.dataset = dataset
        self.device = device
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.eps_blur = float(eps_blur)
        self._enabled = bool(vis_cfg.enabled) and len(dataset) > 0
        self._op = GS3DCTRasterizeCUDA()
        self._params: Optional[Params] = None

        self.frames_dir = os.path.join(out_dir, vis_cfg.frame_subdir)
        self.video_path = os.path.join(out_dir, vis_cfg.output_name)
        if self._enabled:
            os.makedirs(self.frames_dir, exist_ok=True)
            self._cached_view = self._load_view_sample(vis_cfg.view_index)
        else:
            self._cached_view = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_params(self, params: Params) -> None:
        if not self._enabled:
            return
        self._params = params
        self._op.initialize(params)

    def maybe_record_step(self, step: int) -> None:
        if not self._enabled or self._params is None:
            return
        if step % max(int(self.cfg.interval), 1) != 0:
            return
        pred, gt = self._render_cached_view()
        frame_path = os.path.join(self.frames_dir, f"frame_{step:06d}.png")
        save_projection_triplet(
            pred=pred,
            gt=gt,
            path=frame_path,
            value_range=(self.cfg.value_min, self.cfg.value_max),
            title=f"Step {step}",
        )

    def finalize(self) -> Optional[str]:
        if not self._enabled:
            return None
        frame_files = sorted(
            f for f in os.listdir(self.frames_dir)
            if f.endswith(".png")
        )
        if len(frame_files) == 0:
            return None
        try:
            import imageio.v2 as imageio
        except Exception:
            print("   [VideoVis] imageio is unavailable, skip mp4 export.")
            return None

        with imageio.get_writer(self.video_path, fps=int(self.cfg.fps)) as writer:
            for name in frame_files:
                frame_path = os.path.join(self.frames_dir, name)
                writer.append_data(imageio.imread(frame_path))
        return self.video_path

    def _load_view_sample(self, view_index: int):
        index = int(np.clip(view_index, 0, len(self.dataset) - 1))
        sample = self.dataset[index]
        c2w = sample["camtoworld"].unsqueeze(0).to(self.device).contiguous()
        K = sample["K"].unsqueeze(0).to(self.device).contiguous()
        gt = sample["image"].squeeze(-1).to(self.device).contiguous()
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        return {"c2w": c2w, "K": K, "gt": gt}

    @torch.no_grad()
    def _render_cached_view(self):
        assert self._cached_view is not None
        c2w = self._cached_view["c2w"]
        K = self._cached_view["K"]
        gt = self._cached_view["gt"]
        viewmats = torch.linalg.inv(c2w).contiguous()
        C = viewmats.shape[0]
        opacity_raw = self._params.get("opacity").data
        opacity_2d = opacity_raw.unsqueeze(0).expand(C, -1).contiguous()

        rast_params = Params()
        rast_params.add_param("means", self._params.get("means"))
        rast_params.add_param("quats", self._params.get("quats"))
        rast_params.add_param("scale_ori", self._params.get("scale_ori"))
        rast_params.add_param("opacity", _ParamWrapper(opacity_2d))

        data_obj = Data()
        data_obj.set(viewmats=viewmats, Ks=K)

        img_h, img_w = int(gt.shape[1]), int(gt.shape[2])
        op_cfg = Config()
        op_cfg.set(
            img_dim1=img_w,
            img_dim0=img_h,
            tile_size=16,
            eps_blur=self.eps_blur,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )
        fwd = self._op.forward_only(rast_params, data_obj, op_cfg)
        pred = fwd.outputs["render_output"][0].detach().cpu()
        gt_img = gt[0].detach().cpu()
        return pred, gt_img
