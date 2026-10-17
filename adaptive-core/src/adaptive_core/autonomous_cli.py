"""Command-line entry point for a bounded online research cycle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .autonomy import AutonomousLearningLoop
from .cognition import CognitiveSystem
from .core import AdaptiveCore
from .internet_learning import WikipediaLearner
from .store import Store

def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded self-directed research; no LLM is required.")
    parser.add_argument("--steps", type=int, default=3, help="research cycles, 1-10")
    parser.add_argument("--page-budget", type=int, default=6, help="maximum requested pages, 1-30")
    parser.add_argument("--language", default="ru", help="Wikipedia language code")
    parser.add_argument("--topic", action="append", help="topic; may be repeated")
    parser.add_argument("--strategy-db", default="adaptive-core.sqlite3", help="path to SQLite state database")
    parser.add_argument("--web-db", default="adaptive-web-knowledge.sqlite3", help="path to web-knowledge database")
    args = parser.parse_args()
    Path(args.strategy_db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.web_db).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    with Store(args.strategy_db) as store:
        core = AdaptiveCore(store)
        if not store.list_strategies("autonomous-research", status="active"):
            core.create_baseline("autonomous-research")
        internet = WikipediaLearner(args.web_db)
        try:
            cognitive = CognitiveSystem(core, internet=internet)
            loop = AutonomousLearningLoop(cognitive, topics=args.topic, language=args.language, pages_per_topic=2)
            print(json.dumps(loop.run(max_steps=args.steps, page_budget=args.page_budget), ensure_ascii=False, indent=2))
        finally:
            internet.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
