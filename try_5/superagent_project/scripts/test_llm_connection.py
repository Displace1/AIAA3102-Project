import argparse
import json
import sys
from pathlib import Path


# Make both execution methods work:
# python -m scripts.test_llm_connection
# python scripts/test_llm_connection.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from superagent.llm.budget_manager import (
    BudgetError,
    BudgetManager,
)
from superagent.llm.budgeted_client import (
    BudgetedLLMClient,
)
from superagent.llm.client import (
    LLMClient,
    LLMClientError,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the HKUST(GZ) LLM API connection."
    )

    parser.add_argument(
        "--budgeted",
        action="store_true",
        help=(
            "Enable pricing validation, pre-call authorization, "
            "and cost recording."
        ),
    )

    return parser.parse_args()


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_messages():
    return [
        {
            "role": "system",
            "content": (
                "You are a machine-learning experiment planner. "
                "Follow the user's requested output format exactly."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly this text and nothing else: "
                "API connection successful"
            ),
        },
    ]


def print_usage(result):
    print("\nToken usage:")
    print(
        json.dumps(
            result.get("usage"),
            indent=2,
            ensure_ascii=False,
        )
    )


def print_cost(result):
    cost = result.get("cost")

    if cost is None:
        return

    print("\nCost record:")
    print(
        json.dumps(
            cost,
            indent=2,
            ensure_ascii=False,
        )
    )


def main():
    args = parse_args()

    llm_config_path = (
        PROJECT_ROOT
        / "configs"
        / "llm.json"
    )

    pricing_config_path = (
        PROJECT_ROOT
        / "configs"
        / "model_pricing.json"
    )

    smoke_test_dir = (
        PROJECT_ROOT
        / "runs"
        / "api_smoke_test"
    )

    llm_log_path = (
        smoke_test_dir
        / "llm_calls.jsonl"
    )

    budget_ledger_path = (
        smoke_test_dir
        / "budget_ledger.jsonl"
    )

    if not llm_config_path.exists():
        raise FileNotFoundError(
            f"LLM configuration not found: {llm_config_path}"
        )

    llm_config = load_json(llm_config_path)

    client = LLMClient(
        endpoint=llm_config["endpoint"],
        model=llm_config["model"],
        log_path=llm_log_path,
        api_key_environment_variable=(
            llm_config.get(
                "api_key_environment_variable",
                "AIGC_API_KEY",
            )
        ),
        timeout_seconds=llm_config.get(
            "timeout_seconds",
            120,
        ),
        max_retries=llm_config.get(
            "max_retries",
            3,
        ),
    )

    messages = build_messages()

    if args.budgeted:
        if not pricing_config_path.exists():
            raise FileNotFoundError(
                "Pricing configuration not found: "
                f"{pricing_config_path}"
            )

        budget_manager = BudgetManager(
            pricing_config_path=pricing_config_path,
            ledger_path=budget_ledger_path,
            model_name=llm_config["model"],
        )

        callable_client = BudgetedLLMClient(
            llm_client=client,
            budget_manager=budget_manager,
        )

        result = callable_client.chat(
            messages=messages,
            purpose="exploration",
            temperature=llm_config.get(
                "temperature",
                0.2,
            ),
            max_tokens=llm_config.get(
                "max_tokens",
                500,
            ),
        )

        mode = "budgeted"

    else:
        result = client.chat(
            messages=messages,
            purpose="api_connection_test",
            temperature=llm_config.get(
                "temperature",
                0.2,
            ),
            max_tokens=llm_config.get(
                "max_tokens",
                500,
            ),
        )

        mode = "connection-only"

    print(f"Mode: {mode}")
    print(f"Model: {llm_config['model']}")

    print("\nResponse:")
    print(result["content"])

    print_usage(result)
    print_cost(result)

    print("\nFiles:")
    print(f"LLM log: {llm_log_path}")

    if args.budgeted:
        print(f"Budget ledger: {budget_ledger_path}")
        print(
            "Remaining budget: "
            f"{budget_manager.remaining_rmb()} RMB"
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
        raise SystemExit(130)

    except (LLMClientError, BudgetError) as error:
        print(f"\nTest failed: {error}")
        raise SystemExit(1)

    except Exception as error:
        print(
            "\nUnexpected test failure: "
            f"{type(error).__name__}: {error}"
        )
        raise SystemExit(1)