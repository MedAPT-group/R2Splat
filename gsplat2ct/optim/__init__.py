"""
Optimizer package for gsplat2ct.

New optimizer hierarchy (for GS3DCT):
    base.OptimizerBase
    ├── base.FirstOrderOptimizerBase  → gs3d_ct.GS3DCTAdamOptimizer
    └── base.NewtonOptimizerBase      → gs3d_ct.GS3DCTNewtonOptimizer

Legacy optimizers (gs2d, gs3d_smoothing, mlp, loss) still use the old
SecondOrderOptimizerBase interface.

Author: Dy164
Date: 2026-03-25
"""

# --- New base classes and enums ---
from .base import (
    HessianMode,
    SignalStrategy,
    OptimizerConfig,
    FirstOrderConfig,
    SecondOrderConfig,
    OptimizerBase,
    FirstOrderOptimizerBase,
    NewtonOptimizerBase,
    SecondOrderOptimizerBase,  # legacy alias
)

# --- GS3DCT-specific optimizers ---
from .gs3d_ct import (
    GS3DCTAdamOptimizer,
    GS3DCTNewtonOptimizer,
    create_gs3d_ct_adam,
    create_gs3d_ct_newton,
)

# --- Legacy optimizers ---
from .mlp import MLPOptimizerConfig, MLPSecondOrderOptimizer
from .loss import ImageLossOptimizerConfig, ImageLossSecondOrderOptimizer
from .gs2d import GS2DOptimizerConfig, GS2DSecondOrderOptimizer
from .gs3d_smoothing import (
    GS3DSmoothingOptimizerConfig,
    GS3DSmoothingSecondOrderOptimizer,
    create_gs3d_smoothing_optimizer,
)


__all__ = [
    # Enums
    "HessianMode",
    "SignalStrategy",

    # Configs
    "OptimizerConfig",
    "FirstOrderConfig",
    "SecondOrderConfig",

    # New base classes
    "OptimizerBase",
    "FirstOrderOptimizerBase",
    "NewtonOptimizerBase",

    # GS3DCT optimizers
    "GS3DCTAdamOptimizer",
    "GS3DCTNewtonOptimizer",
    "create_gs3d_ct_adam",
    "create_gs3d_ct_newton",

    # Legacy
    "SecondOrderOptimizerBase",
    "MLPOptimizerConfig",
    "MLPSecondOrderOptimizer",
    "ImageLossOptimizerConfig",
    "ImageLossSecondOrderOptimizer",
    "GS2DOptimizerConfig",
    "GS2DSecondOrderOptimizer",
    "GS3DSmoothingOptimizerConfig",
    "GS3DSmoothingSecondOrderOptimizer",
    "create_gs3d_smoothing_optimizer",
]
