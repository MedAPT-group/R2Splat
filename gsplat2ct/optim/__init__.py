"""
Optimizer package for gsplat2ct.

This package provides second-order optimization methods based on damped Newton's method.

Submodules:
- base: Abstract base classes (OptimizerConfig, SecondOrderOptimizerBase)
- mlp: MLP-specific optimizer (MLPOptimizerConfig, MLPSecondOrderOptimizer)
- loss: ImageLoss-specific optimizer (ImageLossOptimizerConfig, ImageLossSecondOrderOptimizer)

Usage:
    # Import from specific submodule
    from gsplat2ct.optim.base import OptimizerConfig
    from gsplat2ct.optim.mlp import MLPSecondOrderOptimizer
    from gsplat2ct.optim.loss import ImageLossSecondOrderOptimizer
    
    # Or import directly from package (convenience)
    from gsplat2ct.optim import MLPSecondOrderOptimizer, MLPOptimizerConfig
    from gsplat2ct.optim import ImageLossSecondOrderOptimizer, ImageLossOptimizerConfig

Author: Dy164
Date: 2025-12-18
"""

from .base import OptimizerConfig, SecondOrderOptimizerBase
from .mlp import MLPOptimizerConfig, MLPSecondOrderOptimizer
from .loss import ImageLossOptimizerConfig, ImageLossSecondOrderOptimizer
from .gs2d import GS2DOptimizerConfig, GS2DSecondOrderOptimizer
from .gs3d_smoothing import (
    GS3DSmoothingOptimizerConfig,
    GS3DSmoothingSecondOrderOptimizer,
    create_gs3d_smoothing_optimizer,
)


__all__ = [
    # Base classes
    "OptimizerConfig",
    "SecondOrderOptimizerBase",
    
    # MLP-specific
    "MLPOptimizerConfig",
    "MLPSecondOrderOptimizer",
    
    # ImageLoss-specific
    "ImageLossOptimizerConfig",
    "ImageLossSecondOrderOptimizer",
    
    # 2DGS-specific
    "GS2DOptimizerConfig",
    "GS2DSecondOrderOptimizer",
    
    # 3DGS Smoothing-specific
    "GS3DSmoothingOptimizerConfig",
    "GS3DSmoothingSecondOrderOptimizer",
    "create_gs3d_smoothing_optimizer",
]
