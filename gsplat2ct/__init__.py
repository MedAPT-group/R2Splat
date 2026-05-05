"""gsplat2ct: fast first-order 3D Gaussian Splatting for CT reconstruction."""

__version__ = "0.1.0"

from .ops import Config, Data, ForwardBackwardBase, ForwardResult, GradResult, Params
from .optim import FirstOrderConfig, GS3DCTAdamOptimizer, create_gs3d_ct_adam

__all__ = [
    "__version__",
    "ForwardBackwardBase",
    "Params",
    "Data",
    "Config",
    "ForwardResult",
    "GradResult",
    "FirstOrderConfig",
    "GS3DCTAdamOptimizer",
    "create_gs3d_ct_adam",
]
