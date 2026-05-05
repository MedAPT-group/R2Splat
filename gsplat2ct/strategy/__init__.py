from .default import DefaultStrategy
from .mcmc import KAMCMCStrategy, MCMCStrategy


class NoOpStrategy(DefaultStrategy):
    def __init__(self, cfg=None):
        super().__init__({"enabled": False, "absgrad": False})


def build_strategy(cfg):
    strategy_type = "grad2d_based_adc"
    enabled = True
    if cfg is None:
        enabled = False
    else:
        strategy_type = str(getattr(cfg, "type", getattr(cfg, "strategy_type", strategy_type))).lower()
        enabled = bool(getattr(cfg, "enabled", True))

    if not enabled or strategy_type in {"none", "off", "noop", "no_op"}:
        return NoOpStrategy(cfg)
    if strategy_type in {"grad2d_based_adc", "default", "adc"}:
        return DefaultStrategy(cfg)
    if strategy_type in {"ka_mcmc", "mcmc", "kernel_agnostic_mcmc"}:
        return KAMCMCStrategy(cfg)
    raise ValueError(f"Unknown strategy.type: {strategy_type}")


__all__ = [
    "DefaultStrategy",
    "KAMCMCStrategy",
    "MCMCStrategy",
    "NoOpStrategy",
    "build_strategy",
]
