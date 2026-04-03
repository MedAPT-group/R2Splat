"""CT projection dataset loading.

Reads the ``meta_data.json`` file format shared with r2splat projects.  The
dataset returns per-view camera parameters and ground-truth projection images
that can be converted into the ``Data`` / ``Config`` containers expected by
``GS3DCTLossCUDA``.

Dataset directory layout::

    <data_dir>/
        meta_data.json          # scanner geometry + projection list
        <vol_file>.npy          # ground-truth 3-D volume  (W, D, H)
        init_<scene>.npy        # initial point cloud  (N, >=4)
        projections/
            proj_000.npy  ...   # per-angle detector images
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class CTImageParser:
    """Parse ``meta_data.json`` and expose scanner geometry, projections,
    bounding box, and ground-truth volume path."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        meta_path = os.path.join(data_dir, "meta_data.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta_data.json not found in {data_dir}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.vol_gt_path = os.path.join(data_dir, self.meta["vol"])
        self.scanner = self.meta["scanner"]
        self.bbox: Optional[List[List[float]]] = self.meta.get("bbox")
        self.proj_train: List[Dict[str, Any]] = self.meta["proj_train"]
        self.proj_test: List[Dict[str, Any]] = self.meta["proj_test"]

    def get_proj_info(self, split: str = "train") -> List[Dict[str, Any]]:
        if split == "train":
            return self.proj_train
        if split == "test":
            return self.proj_test
        raise ValueError(f"Unknown split: {split}")

    def get_vol_gt(self) -> torch.Tensor:
        """Return ground-truth volume in original stored order (W, D, H)."""
        vol = np.load(self.vol_gt_path)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3-D volume, got shape {vol.shape}")
        return torch.from_numpy(vol).float()

    def get_scanner_params(self) -> Dict[str, Any]:
        return self.scanner

    def get_bbox(self) -> Optional[List[List[float]]]:
        return self.bbox


class CTImageDataset(torch.utils.data.Dataset):
    """PyTorch dataset that pre-loads all projections and camera matrices.

    Each sample is a dict with keys ``K``, ``camtoworld``, ``image``, and
    ``image_id``.  Images are returned as ``(W, H, 1)`` float32 tensors.
    """

    def __init__(self, parser: CTImageParser, split: str = "train"):
        self.parser = parser
        self.split = split
        self.proj_info = parser.get_proj_info(split)
        self.data_dir = parser.data_dir

        self.images: List[torch.Tensor] = []
        self.camtoworlds: List[torch.Tensor] = []

        for info in self.proj_info:
            proj_path = os.path.join(self.data_dir, info["file_path"])
            angle = info["angle"]

            image = np.load(proj_path)
            if image.ndim == 2:
                image = image[..., None]
            elif image.ndim == 3 and image.shape[-1] == 1:
                pass
            else:
                raise ValueError(
                    f"Unexpected projection shape {image.shape}; "
                    "expected 2-D or (W, H, 1)"
                )
            self.images.append(torch.from_numpy(image).float())
            self.camtoworlds.append(self._get_camtoworld(angle))

        self.K = self._get_K()
        self._images_tensor: Optional[torch.Tensor] = None
        self._camtoworlds_tensor: Optional[torch.Tensor] = None
        self._K_tensor: Optional[torch.Tensor] = None
        self._preload_device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.proj_info)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "K": self.K,
            "camtoworld": self.camtoworlds[idx],
            "image": self.images[idx],
            "image_id": idx,
        }

    def get_vol_gt(self) -> torch.Tensor:
        return self.parser.get_vol_gt()

    def get_scanner_params(self) -> Dict[str, Any]:
        return self.parser.get_scanner_params()

    def get_bbox(self) -> Optional[List[List[float]]]:
        return self.parser.get_bbox()

    def preload_to(self, device: Union[str, torch.device]) -> None:
        """Preload all per-view tensors onto a target device."""
        dev = torch.device(device)
        if self._preload_device == dev:
            return
        self._images_tensor = torch.stack(self.images, dim=0).to(dev).contiguous()
        self._camtoworlds_tensor = torch.stack(self.camtoworlds, dim=0).to(dev).contiguous()
        self._K_tensor = self.K.to(dev).contiguous()
        self._preload_device = dev

    def get_batch(self, indices: Union[int, List[int], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fetch a batched sample dict from preloaded tensors."""
        if self._images_tensor is None or self._camtoworlds_tensor is None or self._K_tensor is None:
            raise RuntimeError("Dataset is not preloaded. Call preload_to(device) first.")

        if isinstance(indices, int):
            idx = torch.tensor([indices], device=self._images_tensor.device, dtype=torch.long)
        elif isinstance(indices, list):
            idx = torch.tensor(indices, device=self._images_tensor.device, dtype=torch.long)
        elif isinstance(indices, torch.Tensor):
            idx = indices.to(self._images_tensor.device, dtype=torch.long).reshape(-1)
        else:
            raise TypeError(f"Unsupported indices type: {type(indices)}")

        c2w = self._camtoworlds_tensor.index_select(0, idx).contiguous()
        image = self._images_tensor.index_select(0, idx).contiguous()
        K = self._K_tensor.unsqueeze(0).expand(idx.shape[0], -1, -1).contiguous()
        return {
            "K": K,
            "camtoworld": c2w,
            "image": image,
            "image_id": idx,
        }

    # ------------------------------------------------------------------
    # Camera geometry
    # ------------------------------------------------------------------

    def _get_K(self) -> torch.Tensor:
        """Build 3x3 intrinsic matrix from scanner metadata."""
        scanner = self.parser.scanner
        n_detector = scanner["nDetector"]
        s_detector = scanner["sDetector"]
        dsd = scanner["DSD"]

        width, height = n_detector[1], n_detector[0]
        pixel_size_x = s_detector[1] / n_detector[1]
        pixel_size_y = s_detector[0] / n_detector[0]

        K = torch.tensor(
            [
                [dsd / pixel_size_x, 0.0, width / 2.0],
                [0.0, dsd / pixel_size_y, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return K

    def _get_camtoworld(self, angle: float) -> torch.Tensor:
        """Compute a camera-to-world matrix from the scanner rotation angle.

        Convention (matching r2splat1):
            1. Rotate -90 deg around x-axis (fixed).
            2. Rotate +90 deg around z-axis (fixed).
            3. Rotate ``angle`` around z-axis (fixed).
            Translation = DSO * [cos(angle), sin(angle), 0].
        """
        dso = self.parser.scanner["DSO"]

        c1, s1 = np.cos(-np.pi / 2), np.sin(-np.pi / 2)
        R1 = np.array([[1, 0, 0], [0, c1, -s1], [0, s1, c1]])

        c2, s2 = np.cos(np.pi / 2), np.sin(np.pi / 2)
        R2 = np.array([[c2, -s2, 0], [s2, c2, 0], [0, 0, 1]])

        ca, sa = np.cos(angle), np.sin(angle)
        R3 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])

        rot = R3 @ R2 @ R1
        trans = np.array([dso * ca, dso * sa, 0.0])

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot
        c2w[:3, 3] = trans
        return torch.from_numpy(c2w)
