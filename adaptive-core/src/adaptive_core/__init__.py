"""Guarded adaptive strategy core."""

from .core import AdaptiveCore, NoActiveStrategyError
from .evolution import EvolutionManager
from .models import EvaluationCase, PromotionDecision, StrategyConfig, StoredStrategy
from .store import Store

__all__ = [
    "AdaptiveCore",
    "EvaluationCase",
    "EvolutionManager",
    "NoActiveStrategyError",
    "PromotionDecision",
    "Store",
    "StrategyConfig",
    "StoredStrategy",
]

__version__ = "0.1.0"
