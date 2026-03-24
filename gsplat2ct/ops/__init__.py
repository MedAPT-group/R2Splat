"""
Operators module for gsplat2ct.

This module contains operator implementations with support for
second-order derivatives.

Author: Dy164
Date: 2025-12-02
"""

# Import base class
from .base import SecondForwardBackwardBase

# Import generic data structures
from .data import (
    ParameterLike,
    Params,
    Data,
    Config,
    ForwardResult,
    GradResult,
    HessResult,
)

# Import MLP classification operators
from .mlp_classification import (
    MLPClassificationCUDA,
    MLPClassificationTorch,
    MLPClassificationAutograd,
    create_mlp_classification,
)

# Import loss functions
from .loss import (
    ImageLossAutograd,
    VolumeLossAutograd,
    ImageLossTorch,
    VolumeLossTorch,
    ImageLossCUDA,
    L2LossAutograd,
    L2LossCUDA,
    create_image_loss,
    create_image_loss_torch,
    create_image_loss_cuda,
    create_volume_loss,
    create_volume_loss_torch,
    create_l2_loss,
    create_l2_loss_cuda,
    l1_loss,
    ssim,
    tv_3d_loss,
)

# Import rasterize to pixels operators
from .rasterize_to_pixels import (
    RasterizeToPixelsCUDA,
    RasterizeToPixelsTorch,
    create_rasterize_to_pixels,
)

# Import 2D Gaussian Splatting operators
from .gs2d import (
    GS2DParams,
    GS2DData,
    GS2DConfig,
    GS2DAutograd,
    GS2DCUDA,
    create_gs2d_autograd,
    create_gs2d_cuda,
)

# Import voxelize to voxels operators
from .voxelize_to_voxels import (
    VoxelizeToVoxelsCUDA,
    VoxelizeToVoxelsAutograd,
)

# Import 3D Gaussian Splatting with Smoothing operators
from .gs3d_smoothing import (
    GS3DSmoothingParams,
    GS3DSmoothingData,
    GS3DSmoothingConfig,
    GS3DSmoothingAutograd,
    GS3DSmoothingCUDA,
    create_gs3d_smoothing_autograd,
    create_gs3d_smoothing_cuda,
)

# Import ConicsMu operators
from .conics_mu import (
    ConicsMuCUDA,
    ConicsMuAutograd,
    create_conics_mu,
)

# Import 2D Gaussian Splatting with ConicsMu rectification operators
from .gs2d_rectifying import (
    GS2DRectifyingParams,
    GS2DRectifyingData,
    GS2DRectifyingConfig,
    GS2DRectifyingAutograd,
    GS2DRectifyingCUDA,
    create_gs2d_rectifying_autograd,
    create_gs2d_rectifying_cuda,
)

# Import WorldToImageProjection operators
from .world_to_image_projection import (
    WorldToImageProjectionCUDA,
    WorldToImageProjectionAutograd,
    create_world_to_image_projection,
)

# Import WorldToVolumeMapping operators
from .world_to_volume_mapping import (
    WorldToVolumeMappingCUDA,
    WorldToVolumeMappingAutograd,
    create_world_to_volume_mapping,
)

# Import ParameterTransform operators
from .parameter_transform import (
    ParameterTransformCUDA,
    ParameterTransformAutograd,
    create_parameter_transform,
)

# Import ProjectionAndRasterization pipeline
from .projection_and_rasterization import (
    ProjectionAndRasterizationCUDA,
    ProjectionAndRasterizationAutograd,
    create_projection_and_rasterization,
)

# Import GS3DProjectionOnly pipeline (ParameterTransform + Projection + Rasterization)
from .gs3d_projection_only import (
    GS3DProjectionOnlyCUDA,
    GS3DProjectionOnlyAutograd,
    create_gs3d_projection_only,
)

# Import GS3DVoxelizationOnly pipeline (ParameterTransform + WorldToVolumeMapping + VoxelizeToVoxels + L2Loss)
from .gs3d_voxelization_only import (
    GS3DVoxelizationOnlyCUDA,
    GS3DVoxelizationOnlyAutograd,
    create_gs3d_voxelization_only,
)


__all__ = [
    # Base class
    "SecondForwardBackwardBase",
    # Generic data structures
    "ParameterLike",
    "Params",
    "Data",
    "Config",
    "ForwardResult",
    "GradResult",
    "HessResult",
    # MLP classification
    "MLPClassificationCUDA",
    "MLPClassificationTorch",
    "MLPClassificationAutograd",
    "create_mlp_classification",
    # Loss functions (autograd)
    "ImageLossAutograd",
    "VolumeLossAutograd",
    # Loss functions (manual torch)
    "ImageLossTorch",
    "VolumeLossTorch",
    # Loss functions (CUDA)
    "ImageLossCUDA",
    "L2LossCUDA",
    # Loss functions (autograd/torch)
    "L2LossAutograd",
    # Factory functions
    "create_image_loss",
    "create_image_loss_torch",
    "create_image_loss_cuda",
    "create_volume_loss",
    "create_volume_loss_torch",
    "create_l2_loss",
    "create_l2_loss_cuda",
    # Rasterize to pixels
    "RasterizeToPixelsCUDA",
    "RasterizeToPixelsTorch",
    "create_rasterize_to_pixels",
    # 2D Gaussian Splatting
    "GS2DParams",
    "GS2DData",
    "GS2DConfig",
    "GS2DAutograd",
    "GS2DCUDA",
    "create_gs2d_autograd",
    "create_gs2d_cuda",
    # Voxelize to Voxels
    "VoxelizeToVoxelsCUDA",
    "VoxelizeToVoxelsAutograd",
    # 3D Gaussian Splatting with Smoothing
    "GS3DSmoothingParams",
    "GS3DSmoothingData",
    "GS3DSmoothingConfig",
    "GS3DSmoothingAutograd",
    "GS3DSmoothingCUDA",
    "create_gs3d_smoothing_autograd",
    "create_gs3d_smoothing_cuda",
    # ConicsMu
    "ConicsMuCUDA",
    "ConicsMuAutograd",
    "create_conics_mu",
    # 2D Gaussian Splatting with ConicsMu Rectification
    "GS2DRectifyingParams",
    "GS2DRectifyingData",
    "GS2DRectifyingConfig",
    "GS2DRectifyingAutograd",
    "GS2DRectifyingCUDA",
    "create_gs2d_rectifying_autograd",
    "create_gs2d_rectifying_cuda",
    # WorldToImageProjection
    "WorldToImageProjectionCUDA",
    "WorldToImageProjectionAutograd",
    "create_world_to_image_projection",
    # WorldToVolumeMapping
    "WorldToVolumeMappingCUDA",
    "WorldToVolumeMappingAutograd",
    "create_world_to_volume_mapping",
    # ParameterTransform
    "ParameterTransformCUDA",
    "ParameterTransformAutograd",
    "create_parameter_transform",
    # ProjectionAndRasterization pipeline
    "ProjectionAndRasterizationCUDA",
    "ProjectionAndRasterizationAutograd",
    "create_projection_and_rasterization",
    # GS3DProjectionOnly pipeline
    "GS3DProjectionOnlyCUDA",
    "GS3DProjectionOnlyAutograd",
    "create_gs3d_projection_only",
    # GS3DVoxelizationOnly pipeline
    "GS3DVoxelizationOnlyCUDA",
    "GS3DVoxelizationOnlyAutograd",
    "create_gs3d_voxelization_only",
    # Utility functions
    "l1_loss",
    "ssim",
    "tv_3d_loss",
]
