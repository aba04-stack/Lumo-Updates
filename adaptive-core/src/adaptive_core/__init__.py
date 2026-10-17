"""Guarded adaptive strategy and cognitive research core."""

from .cognition import CognitiveSnapshot, CognitiveSystem
from .core import AdaptiveCore, NoActiveStrategyError
from .episodic_memory import EpisodicMemory
from .evolution import EvolutionManager
from .internet_learning import InternetLearningError, WebKnowledgeStore, WikipediaLearner
from .models import EvaluationCase, PromotionDecision, StrategyConfig, StoredStrategy
from .runtime import AdaptiveRunner, Assessment, RunResult
from .self_model import SelfModel, SelfModelReport
from .store import Store
from .workspace import GlobalWorkspace, WorkspaceSignal

__all__ = [
    "AdaptiveCore", "AdaptiveRunner", "Assessment", "CognitiveSnapshot",
    "CognitiveSystem", "EvaluationCase", "EpisodicMemory", "EvolutionManager",
    "GlobalWorkspace", "InternetLearningError", "NoActiveStrategyError",
    "PromotionDecision", "RunResult", "SelfModel", "SelfModelReport", "Store",
    "StrategyConfig", "StoredStrategy", "WebKnowledgeStore", "WikipediaLearner",
    "WorkspaceSignal",
]

__version__ = "0.3.0"
