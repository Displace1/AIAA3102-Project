import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from superagent.experiment_plan import ExperimentPlan
from superagent.plan_validator import PlanValidator


class PlannerError(RuntimeError):
    pass


class PlannerResponseError(PlannerError):
    pass


@dataclass(frozen=True)
class PlannerDecision:
    action: str
    reason: str
    plan: Optional[ExperimentPlan]
    call_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    cost: Optional[Dict[str, Any]] = None

    def to_dict(self):
        data = asdict(self)
        if self.plan is not None:
            data["plan"] = self.plan.to_dict()
        return data


class LLMExperimentPlanner:
    """Uses an LLM to propose one validated experiment or stop."""

    def __init__(
        self,
        llm_client,
        temperature=0.1,
        max_tokens=2048,
        max_response_attempts=3,
    ):
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_response_attempts = max(1, int(max_response_attempts))
        self.validator = PlanValidator()

    def plan_next(
        self,
        data_profile,
        experiment_history,
        existing_plans,
        remaining_budget_rmb,
        next_experiment_id,
    ):
        if not re.fullmatch(r"exp_\d{3,}", next_experiment_id):
            raise ValueError(
                "next_experiment_id must look like exp_003."
            )

        modality = data_profile["modality"]
        adapter_specs = self.validator.allowed_adapter_specs(modality)
        if not adapter_specs:
            raise PlannerError(
                f"No allow-listed adapters support modality: {modality}"
            )

        messages = self._build_messages(
            data_profile=data_profile,
            experiment_history=experiment_history,
            existing_plans=existing_plans,
            remaining_budget_rmb=remaining_budget_rmb,
            next_experiment_id=next_experiment_id,
            adapter_specs=adapter_specs,
        )

        request_messages = list(messages)
        token_limit = self.max_tokens
        response = None
        result = None
        last_error = None
        for response_attempt in range(1, self.max_response_attempts + 1):
            result = self.llm_client.chat(
                messages=request_messages,
                purpose="exploration",
                temperature=self.temperature,
                max_tokens=token_limit,
            )
            content = result.get("content")
            try:
                if not isinstance(content, str) or not content.strip():
                    raise PlannerResponseError(
                        "The LLM returned an empty planner response."
                    )
                response = self._parse_json_object(content)
                break
            except PlannerResponseError as error:
                last_error = error
                if response_attempt >= self.max_response_attempts:
                    raise PlannerResponseError(
                        "Planner response remained unusable after "
                        f"{self.max_response_attempts} audited API calls: "
                        f"{error}"
                    ) from error
                if result.get("finish_reason") == "length" or not (
                    isinstance(content, str) and content.strip()
                ):
                    token_limit = min(max(token_limit * 2, 2048), 8192)
                request_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "The previous response was empty, truncated, or "
                            "invalid JSON. Return only one short JSON object "
                            "matching the requested schema. Do not include "
                            "analysis, Markdown, or code fences."
                        ),
                    }
                ]

        if response is None or result is None:
            raise PlannerResponseError(
                f"Planner did not produce a response: {last_error}"
            )
        decision = self._decision_from_response(
            response=response,
            data_profile=data_profile,
            existing_plans=existing_plans,
            next_experiment_id=next_experiment_id,
            call_result=result,
        )
        if decision.action == "run_experiment":
            self._validate_gpu_oom_recovery(
                plan=decision.plan,
                experiment_history=experiment_history,
                existing_plans=existing_plans,
            )
        return decision

    def _build_messages(
        self,
        data_profile,
        experiment_history,
        existing_plans,
        remaining_budget_rmb,
        next_experiment_id,
        adapter_specs,
    ):
        compact_history = [
            self._compact_history_record(record)
            for record in experiment_history
        ]
        context = {
            "data_profile": data_profile,
            "completed_experiments": compact_history,
            "existing_plans": [
                plan.to_dict() for plan in existing_plans
            ],
            "remaining_budget_rmb": str(remaining_budget_rmb),
            "next_experiment_id": next_experiment_id,
            "allowed_adapters": adapter_specs,
        }

        system_message = (
            "You are the experiment planner inside a general-purpose "
            "classification SuperAgent. Decide whether to run exactly one "
            "new experiment or stop. Use only the supplied data profile, "
            "validation results, remaining budget, and allow-listed adapter "
            "schemas. Never use task names, dataset identities, test data, "
            "or test metrics. Never propose arbitrary Python code. Do not "
            "repeat a completed configuration. Return one JSON object only, "
            "without Markdown or commentary. If the latest image experiment "
            "failed with error_type gpu_out_of_memory, the next experiment "
            "must reduce batch_size, reduce image_size, or choose a smaller "
            "image model. Never repeat the resource configuration that ran "
            "out of GPU memory."
        )
        user_message = (
            "Choose the next action. For an experiment, return exactly: "
            '{"action":"run_experiment","adapter_name":"...",'
            '"hyperparameters":{...},"reason":"..."}. '
            "For stopping, return exactly: "
            '{"action":"stop","reason":"..."}. '
            "The program assigns the experiment ID; do not include one. "
            "All required hyperparameters must be present and no unknown "
            "hyperparameters are allowed.\n\nCONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    @staticmethod
    def _compact_history_record(record):
        if hasattr(record, "to_dict"):
            record = record.to_dict()
        if not isinstance(record, dict):
            raise TypeError(
                "Experiment history entries must be dictionaries or expose "
                "to_dict()."
            )

        allowed_keys = (
            "experiment_id",
            "model_name",
            "status",
            "validation_accuracy",
            "train_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "balanced_accuracy",
            "generalization_gap",
            "class_recall",
            "predicted_class_distribution",
            "majority_class_accuracy",
            "feature_count",
            "model_size_bytes",
            "prediction_time_seconds",
            "diagnosis",
            "training_time_seconds",
            "model_config",
            "error_type",
            "error_message",
            "epoch_count",
            "best_epoch",
            "peak_gpu_memory_mb",
            "early_stopped",
        )
        compact = {
            key: record.get(key)
            for key in allowed_keys
            if key in record
        }
        if "f1_macro" in record:
            compact["macro_f1"] = record.get("f1_macro")
        if "training_time_seconds" in record:
            compact["training_time"] = record.get("training_time_seconds")
        return compact

    @staticmethod
    def _validate_gpu_oom_recovery(
        plan,
        experiment_history,
        existing_plans,
    ):
        if plan is None or not experiment_history:
            return
        latest = experiment_history[-1]
        if hasattr(latest, "to_dict"):
            latest = latest.to_dict()
        if not isinstance(latest, dict):
            return
        if latest.get("error_type") != "gpu_out_of_memory":
            return

        failed_id = latest.get("experiment_id")
        failed_plan = next(
            (
                value
                for value in reversed(existing_plans)
                if value.experiment_id == failed_id
            ),
            None,
        )
        if failed_plan is None:
            raise PlannerResponseError(
                "Cannot validate GPU OOM recovery because the failed plan "
                "is missing from experiment history."
            )

        previous = failed_plan.hyperparameters
        proposed = plan.hyperparameters
        batch_reduced = (
            "batch_size" in previous
            and "batch_size" in proposed
            and proposed["batch_size"] < previous["batch_size"]
        )
        image_reduced = (
            "image_size" in previous
            and "image_size" in proposed
            and proposed["image_size"] < previous["image_size"]
        )
        model_scale = {
            "image_small_cnn": 1,
            "image_mobilenet_v3_small": 2,
            "image_resnet18": 3,
        }
        model_reduced = (
            plan.adapter_name in model_scale
            and failed_plan.adapter_name in model_scale
            and model_scale[plan.adapter_name]
            < model_scale[failed_plan.adapter_name]
        )
        if not (batch_reduced or image_reduced or model_reduced):
            raise PlannerResponseError(
                "After gpu_out_of_memory, the next image experiment must "
                "reduce batch_size, reduce image_size, or use a smaller model."
            )

    @staticmethod
    def _parse_json_object(content):
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(
                r"^```(?:json)?\s*",
                "",
                stripped,
                flags=re.IGNORECASE,
            )
            stripped = re.sub(r"\s*```$", "", stripped)

        first_brace = stripped.find("{")
        last_brace = stripped.rfind("}")
        if first_brace < 0 or last_brace < first_brace:
            raise PlannerResponseError(
                "Planner response did not contain a JSON object."
            )

        json_text = stripped[first_brace : last_brace + 1]
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as error:
            raise PlannerResponseError(
                f"Planner returned invalid JSON: {error}"
            ) from error

        if not isinstance(parsed, dict):
            raise PlannerResponseError(
                "Planner response must be a JSON object."
            )
        return parsed

    def _decision_from_response(
        self,
        response,
        data_profile,
        existing_plans,
        next_experiment_id,
        call_result,
    ):
        action = response.get("action")
        reason = response.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise PlannerResponseError(
                "Planner response requires a non-empty reason."
            )

        metadata = {
            "call_id": call_result.get("call_id"),
            "usage": call_result.get("usage"),
            "cost": call_result.get("cost"),
        }

        if action == "stop":
            allowed_keys = {"action", "reason"}
            unknown = set(response) - allowed_keys
            if unknown:
                raise PlannerResponseError(
                    f"Stop response contains unknown keys: {sorted(unknown)}"
                )
            return PlannerDecision(
                action="stop",
                reason=reason.strip(),
                plan=None,
                **metadata,
            )

        if action != "run_experiment":
            raise PlannerResponseError(
                "action must be 'run_experiment' or 'stop'."
            )

        allowed_keys = {
            "action",
            "adapter_name",
            "hyperparameters",
            "reason",
        }
        unknown = set(response) - allowed_keys
        missing = allowed_keys - set(response)
        if unknown:
            raise PlannerResponseError(
                f"Experiment response contains unknown keys: {sorted(unknown)}"
            )
        if missing:
            raise PlannerResponseError(
                f"Experiment response is missing keys: {sorted(missing)}"
            )
        if not isinstance(response["adapter_name"], str):
            raise PlannerResponseError("adapter_name must be a string.")
        if not isinstance(response["hyperparameters"], dict):
            raise PlannerResponseError("hyperparameters must be an object.")

        plan = ExperimentPlan(
            experiment_id=next_experiment_id,
            adapter_name=response["adapter_name"],
            hyperparameters=response["hyperparameters"],
            reason=reason.strip(),
        )
        validated_plan = self.validator.validate_against_history(
            plan=plan,
            data_profile=data_profile,
            existing_plans=existing_plans,
        )
        return PlannerDecision(
            action="run_experiment",
            reason=reason.strip(),
            plan=validated_plan,
            **metadata,
        )
