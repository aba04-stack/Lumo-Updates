"""Guarded adaptive strategy and cognitive research core."""
from .affect import AffectEngine, AffectState
from .autonomy import AutonomousLearningLoop
from .cognition import CognitiveSnapshot, CognitiveSystem
from .core import AdaptiveCore, NoActiveStrategyError
from .episodic_memory import EpisodicMemory
from .evolution import EvolutionManager
from .internet_learning import InternetLearningError, WebKnowledgeStore, WikipediaLearner
from .models import EvaluationCase, PromotionDecision, StrategyConfig, StoredStrategy
from .motivation import DriveState, MotivationEngine, MotivationalGoal
from .reasoning import EvidenceReview, InferenceResult, InferenceRule, KnowledgeFact, ReasoningEngine
from .runtime import AdaptiveRunner, Assessment, RunResult
from .self_model import SelfModel, SelfModelReport
from .store import Store
from .workspace import GlobalWorkspace, WorkspaceSignal

__all__ = [
    "AffectEngine", "AffectState", "AdaptiveCore", "AdaptiveRunner", "Assessment",
    "AutonomousLearningLoop", "CognitiveSnapshot", "CognitiveSystem", "DriveState",
    "EvaluationCase", "EpisodicMemory", "EvolutionManager", "EvidenceReview",
    "GlobalWorkspace", "InferenceResult", "InferenceRule", "InternetLearningError",
    "KnowledgeFact", "MotivationEngine", "MotivationalGoal", "NoActiveStrategyError",
    "PromotionDecision", "ReasoningEngine", "RunResult", "SelfModel", "SelfModelReport",
    "Store", "StrategyConfig", "StoredStrategy", "WebKnowledgeStore", "WikipediaLearner",
    "WorkspaceSignal",
]
__version__ = "0.9.0"
