"""
gsplat2ct: 3D Gaussian Splatting to CT with Second-Order Optimization
"""

__version__ = "0.1.0"

# Import from ops module (new structure)
from .ops import (
    # Base class
    SecondForwardBackwardBase,
    # Generic data structures
    ParameterLike,
    Params,
    Data,
    Config,
    ForwardResult,
    GradResult,
    HessResult,
    # MLP classification
    MLPClassificationCUDA,
    MLPClassificationTorch,
    MLPClassificationAutograd,
    create_mlp_classification,
)

# Import optimizers
from .optim import (
    OptimizerConfig,
    MLPOptimizerConfig,
    SecondOrderOptimizerBase,
    MLPSecondOrderOptimizer,
)

__all__ = [
    "__version__",
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
    # Optimizers
    "OptimizerConfig",
    "MLPOptimizerConfig",
    "SecondOrderOptimizerBase",
    "MLPSecondOrderOptimizer",
]
