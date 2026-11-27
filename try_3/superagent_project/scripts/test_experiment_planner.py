import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from superagent.data_inspector import DataInspector
from superagent.llm.budget_manager import BudgetManager
from superagent.llm.budgeted_client import BudgetedLLMClient
from superagent.llm.client import LLMClient
from superagent.llm.experiment_planner import LLMExperimentPlanner
from superagent.task_config import load_task_config


class OfflinePlannerClient:
    def __init__(self, content):
        self.content = content

    def chat(self, messages, purpose, temperature, max_tokens):
        return {
            "call_id": "offline-planner-test",
            "content": self.content,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "cost": {"total_cost_rmb": "0.00000000"},
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test one LLM Experiment Planner decision."
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "task1.json",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the real budgeted API. This consumes quota.",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def offline_response_for(modality):
    if modality == "tabular":
        response = {
            "action": "run_experiment",
            "adapter_name": "tabular_one_hot_logistic",
            "hyperparameters": {
                "max_iter": 2000,
                "C": 1.0,
            },
            "reason": "Establish a validated linear baseline.",
        }
    elif modality == "text_pair":
        response = {
            "action": "run_experiment",
            "adapter_name": "text_pair_word_tfidf_logistic",
            "hyperparameters": {
                "ngram_min": 1,
                "ngram_max": 2,
                "min_df": 2,
                "max_df": 0.98,
                "max_features": 50000,
                "sublinear_tf": True,
                "C": 1.0,
                "max_iter": 2000,
            },
            "reason": "Establish a validated pairwise text baseline.",
        }
    else:
        response = {
            "action": "stop",
            "reason": "No offline fixture supports this modality.",
        }
    return json.dumps(response)


def build_live_client(llm_config, smoke_dir):
    raw_client = LLMClient(
        endpoint=llm_config["endpoint"],
        model=llm_config["model"],
        log_path=smoke_dir / "llm_calls.jsonl",
        api_key_environment_variable=llm_config.get(
            "api_key_environment_variable",
            "AIGC_API_KEY",
        ),
        timeout_seconds=llm_config.get("timeout_seconds", 120),
        max_retries=llm_config.get("max_retries", 3),
    )
    budget_manager = BudgetManager(
        pricing_config_path=PROJECT_ROOT / "configs" / "model_pricing.json",
        ledger_path=smoke_dir / "budget_ledger.jsonl",
        model_name=llm_config["model"],
    )
    return BudgetedLLMClient(raw_client, budget_manager), budget_manager


def main():
    args = parse_args()
    task_config = load_task_config(args.task_config)
    profile = DataInspector().inspect(task_config)
    llm_config = load_json(PROJECT_ROOT / "configs" / "llm.json")
    smoke_dir = PROJECT_ROOT / "runs" / "planner_smoke_test"

    if args.live:
        client, budget_manager = build_live_client(llm_config, smoke_dir)
        remaining_budget = budget_manager.remaining_rmb()
        mode = "live-budgeted"
    else:
        client = OfflinePlannerClient(
            offline_response_for(profile["modality"])
        )
        remaining_budget = "20.00"
        mode = "offline"

    planner = LLMExperimentPlanner(
        llm_client=client,
        temperature=llm_config.get("planner_temperature", 0.1),
        max_tokens=llm_config.get("planner_max_tokens", 800),
    )
    decision = planner.plan_next(
        data_profile=profile,
        experiment_history=[],
        existing_plans=[],
        remaining_budget_rmb=remaining_budget,
        next_experiment_id="exp_001",
    )

    print(f"Mode: {mode}")
    print(f"Detected modality: {profile['modality']}")
    print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
    if args.live:
        print(f"Remaining budget: {budget_manager.remaining_rmb()} RMB")
        print(f"LLM log: {smoke_dir / 'llm_calls.jsonl'}")
        print(f"Budget ledger: {smoke_dir / 'budget_ledger.jsonl'}")


if __name__ == "__main__":
    main()
