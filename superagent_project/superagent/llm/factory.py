import json

from superagent.llm.budget_manager import (
    BudgetManager,
)
from superagent.llm.budgeted_client import (
    BudgetedLLMClient,
)
from superagent.llm.client import LLMClient
from superagent.llm.experiment_planner import (
    LLMExperimentPlanner,
)


def build_experiment_planner(
    project_root,
    run_directory,
):
    llm_config_path = (
        project_root
        / "configs"
        / "llm.json"
    )

    with llm_config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        llm_config = json.load(file)

    raw_client = LLMClient(
        endpoint=llm_config["endpoint"],
        model=llm_config["model"],
        log_path=(
            run_directory
            / "llm_calls.jsonl"
        ),
        api_key_environment_variable=(
            llm_config[
                "api_key_environment_variable"
            ]
        ),
        timeout_seconds=llm_config[
            "timeout_seconds"
        ],
        max_retries=llm_config[
            "max_retries"
        ],
        retry_base_seconds=llm_config.get("retry_base_seconds", 1),
        retry_max_seconds=llm_config.get("retry_max_seconds", 8),
    )

    budget_manager = BudgetManager(
        pricing_config_path=(
            project_root
            / "configs"
            / "model_pricing.json"
        ),
        ledger_path=(
            run_directory
            / "budget_ledger.jsonl"
        ),
        model_name=llm_config["model"],
    )

    budgeted_client = BudgetedLLMClient(
        llm_client=raw_client,
        budget_manager=budget_manager,
    )

    planner = LLMExperimentPlanner(
        llm_client=budgeted_client,
        temperature=llm_config.get(
            "planner_temperature",
            0.1,
        ),
        max_tokens=llm_config.get(
            "planner_max_tokens",
            2048,
        ),
        max_response_attempts=llm_config.get(
            "planner_response_attempts",
            3,
        ),
    )

    return planner, budget_manager
