import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from superagent.llm.budget_manager import (
    BudgetExceededError,
    BudgetManager,
)


class BudgetManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()

        root = Path(__file__).resolve().parents[1]

        self.manager = BudgetManager(
            pricing_config_path=(
                root
                / "tests"
                / "fixtures"
                / "test_pricing.json"
            ),
            ledger_path=(
                Path(self.temp_directory.name)
                / "budget.jsonl"
            ),
            model_name="test-model",
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_calculates_cost(self):
        result = self.manager.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
        )

        # 1000 / 1M * 2 = 0.002
        self.assertEqual(
            result["input_cost_rmb"],
            Decimal("0.00200000"),
        )

        # 500 / 1M * 8 = 0.004
        self.assertEqual(
            result["output_cost_rmb"],
            Decimal("0.00400000"),
        )

        self.assertEqual(
            result["total_cost_rmb"],
            Decimal("0.00600000"),
        )

    def test_records_cost(self):
        record = self.manager.record_successful_call(
            call_id="test-call",
            purpose="exploration",
            input_tokens=1000,
            output_tokens=500,
        )

        self.assertEqual(
            record["total_cost_rmb"],
            "0.00600000",
        )

    def test_rejects_call_above_hard_limit(self):
        with self.assertRaises(BudgetExceededError):
            self.manager.authorize_call(
                estimated_input_tokens=10000000,
                maximum_output_tokens=10000000,
                purpose="finalization",
            )


if __name__ == "__main__":
    unittest.main()