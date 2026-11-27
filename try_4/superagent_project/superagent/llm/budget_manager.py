import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ONE_MILLION = Decimal("1000000")


class BudgetError(RuntimeError):
    pass


class PricingConfigurationError(BudgetError):
    pass


class BudgetExceededError(BudgetError):
    pass


class BudgetManager:
    def __init__(
        self,
        pricing_config_path,
        ledger_path,
        model_name,
    ):
        self.pricing_config_path = Path(
            pricing_config_path
        )
        self.ledger_path = Path(ledger_path)
        self.model_name = model_name

        with self.pricing_config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        models = self.config.get("models", {})

        if model_name not in models:
            raise PricingConfigurationError(
                f"No pricing configuration for model: "
                f"{model_name}"
            )

        self.model_pricing = models[model_name]

        if not self.model_pricing.get("verified", False):
            raise PricingConfigurationError(
                f"Pricing for {model_name} has not been verified."
            )

        input_price = self.model_pricing.get(
            "input_rmb_per_million_tokens"
        )
        output_price = self.model_pricing.get(
            "output_rmb_per_million_tokens"
        )

        if input_price is None or output_price is None:
            raise PricingConfigurationError(
                "Verified pricing cannot contain null prices."
            )

        self.input_price = Decimal(str(input_price))
        self.output_price = Decimal(str(output_price))

        self.hard_limit = Decimal(
            str(self.config["task_budget_rmb"])
        )
        self.exploration_limit = Decimal(
            str(self.config["exploration_limit_rmb"])
        )

        if self.exploration_limit > self.hard_limit:
            raise PricingConfigurationError(
                "Exploration limit cannot exceed hard limit."
            )

        self.ledger_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def calculate_cost(
        self,
        input_tokens,
        output_tokens,
    ):
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative.")

        input_cost = (
            Decimal(input_tokens)
            / ONE_MILLION
            * self.input_price
        )

        output_cost = (
            Decimal(output_tokens)
            / ONE_MILLION
            * self.output_price
        )

        total_cost = input_cost + output_cost

        return {
            "input_cost_rmb": self._round(input_cost),
            "output_cost_rmb": self._round(output_cost),
            "total_cost_rmb": self._round(total_cost),
        }

    def spent_rmb(self):
        if not self.ledger_path.exists():
            return Decimal("0")

        total = Decimal("0")

        with self.ledger_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)
                total += Decimal(
                    str(record["total_cost_rmb"])
                )

        return self._round(total)

    def remaining_rmb(self):
        return self._round(
            self.hard_limit - self.spent_rmb()
        )

    def authorize_call(
        self,
        estimated_input_tokens,
        maximum_output_tokens,
        purpose,
    ):
        estimated = self.calculate_cost(
            input_tokens=estimated_input_tokens,
            output_tokens=maximum_output_tokens,
        )

        estimated_cost = estimated["total_cost_rmb"]
        spent = self.spent_rmb()
        projected = spent + estimated_cost

        if projected > self.hard_limit:
            raise BudgetExceededError(
                "Call rejected: projected cost "
                f"{projected} RMB exceeds hard limit "
                f"{self.hard_limit} RMB."
            )

        if (
            purpose == "exploration"
            and projected > self.exploration_limit
        ):
            raise BudgetExceededError(
                "Exploration call rejected: projected cost "
                f"{projected} RMB exceeds exploration limit "
                f"{self.exploration_limit} RMB."
            )

        return {
            "authorized": True,
            "spent_before_rmb": str(spent),
            "estimated_call_cost_rmb": str(
                estimated_cost
            ),
            "projected_cost_rmb": str(projected),
            "remaining_before_rmb": str(
                self.hard_limit - spent
            ),
        }

    def record_successful_call(
        self,
        call_id,
        purpose,
        input_tokens,
        output_tokens,
    ):
        costs = self.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        spent_before = self.spent_rmb()
        spent_after = (
            spent_before + costs["total_cost_rmb"]
        )

        if spent_after > self.hard_limit:
            raise BudgetExceededError(
                "Actual usage exceeded the hard budget limit."
            )

        record = {
            "call_id": call_id,
            "model": self.model_name,
            "purpose": purpose,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "input_price_rmb_per_million": str(
                self.input_price
            ),
            "output_price_rmb_per_million": str(
                self.output_price
            ),
            "input_cost_rmb": str(
                costs["input_cost_rmb"]
            ),
            "output_cost_rmb": str(
                costs["output_cost_rmb"]
            ),
            "total_cost_rmb": str(
                costs["total_cost_rmb"]
            ),
            "cumulative_cost_rmb": str(
                self._round(spent_after)
            ),
            "remaining_budget_rmb": str(
                self._round(
                    self.hard_limit - spent_after
                )
            ),
            "pricing_snapshot_date": self.config[
                "pricing_snapshot_date"
            ],
            "official_source": self.model_pricing[
                "official_source"
            ],
        }

        with self.ledger_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        return record

    @staticmethod
    def _round(value):
        return value.quantize(
            Decimal("0.00000001"),
            rounding=ROUND_HALF_UP,
        )