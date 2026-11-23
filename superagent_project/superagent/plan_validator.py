import copy
import json

from superagent.experiment_plan import ExperimentPlan
from superagent.image_runtime import (
    estimate_image_training_seconds,
    estimate_image_training_seconds_with_source,
    recommended_image_size,
)


PARAMETER_RULES = {
    "image_small_cnn": {
        "modality": "image",
        "parameters": {
            "image_size": {"type": int, "minimum": 16, "maximum": 512},
            "batch_size": {"type": int, "minimum": 1, "maximum": 512},
            "learning_rate": {
                "type": (int, float),
                "minimum": 0.000001,
                "maximum": 1.0,
            },
            "epochs": {"type": int, "minimum": 1, "maximum": 100},
            "data_augmentation": {"type": bool},
            "early_stopping_patience": {
                "type": int,
                "minimum": 1,
                "maximum": 30,
            },
            "weight_decay": {
                "type": (int, float),
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
    },
    "image_resnet18": {
        "modality": "image",
        "parameters": {
            "image_size": {"type": int, "minimum": 32, "maximum": 512},
            "batch_size": {"type": int, "minimum": 1, "maximum": 512},
            "learning_rate": {
                "type": (int, float),
                "minimum": 0.000001,
                "maximum": 1.0,
            },
            "epochs": {"type": int, "minimum": 1, "maximum": 100},
            "data_augmentation": {"type": bool},
            "early_stopping_patience": {
                "type": int,
                "minimum": 1,
                "maximum": 30,
            },
            "weight_decay": {
                "type": (int, float),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "pretrained": {"type": bool},
            "freeze_backbone_epochs": {
                "type": int,
                "minimum": 0,
                "maximum": 50,
            },
        },
    },
    "image_mobilenet_v3_small": {
        "modality": "image",
        "parameters": {
            "image_size": {"type": int, "minimum": 32, "maximum": 512},
            "batch_size": {"type": int, "minimum": 1, "maximum": 512},
            "learning_rate": {
                "type": (int, float),
                "minimum": 0.000001,
                "maximum": 1.0,
            },
            "epochs": {"type": int, "minimum": 1, "maximum": 100},
            "data_augmentation": {"type": bool},
            "early_stopping_patience": {
                "type": int,
                "minimum": 1,
                "maximum": 30,
            },
            "weight_decay": {
                "type": (int, float),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "pretrained": {"type": bool},
            "freeze_backbone_epochs": {
                "type": int,
                "minimum": 0,
                "maximum": 50,
            },
        },
    },
    "tabular_one_hot_logistic": {
        "modality": "tabular",
        "parameters": {
            "max_iter": {"type": int, "minimum": 100, "maximum": 10000},
            "C": {"type": (int, float), "minimum": 0.001, "maximum": 100.0},
            "numeric_preprocessing": {
                "choices": ["standard", "minmax", "none"],
                "required": False,
            },
            "categorical_preprocessing": {
                "choices": ["onehot", "ordinal"],
                "required": False,
            },
            "ordinal_preprocessing": {
                "choices": ["ordinal", "onehot"],
                "required": False,
            },
            "class_weight": {
                "choices": ["balanced", None],
                "required": False,
            },
        },
    },
    "tabular_extra_trees": {
        "modality": "tabular",
        "parameters": {
            "n_estimators": {"type": int, "minimum": 50, "maximum": 1000},
            "max_features": {"choices": ["sqrt", "log2", None]},
            "min_samples_leaf": {"type": int, "minimum": 1, "maximum": 50},
            "class_weight": {"choices": ["balanced", None]},
            "numeric_preprocessing": {
                "choices": ["none", "standard", "minmax"],
                "required": False,
            },
            "categorical_preprocessing": {
                "choices": ["onehot", "ordinal"],
                "required": False,
            },
            "ordinal_preprocessing": {
                "choices": ["ordinal", "onehot"],
                "required": False,
            },
        },
    },
    "tabular_random_forest": {
        "modality": "tabular",
        "parameters": {
            "n_estimators": {"type": int, "minimum": 50, "maximum": 1000},
            "max_depth": {"type": int, "minimum": 2, "maximum": 100},
            "max_features": {"choices": ["sqrt", "log2", None]},
            "min_samples_leaf": {"type": int, "minimum": 1, "maximum": 50},
            "class_weight": {"choices": ["balanced", None]},
            "numeric_preprocessing": {
                "choices": ["none", "standard", "minmax"],
                "required": False,
            },
            "categorical_preprocessing": {
                "choices": ["onehot", "ordinal"],
                "required": False,
            },
            "ordinal_preprocessing": {
                "choices": ["ordinal", "onehot"],
                "required": False,
            },
        },
    },
    "text_pair_word_tfidf_logistic": {
        "modality": "text_pair",
        "parameters": {
            "ngram_min": {"type": int, "minimum": 1, "maximum": 3},
            "ngram_max": {"type": int, "minimum": 1, "maximum": 4},
            "min_df": {"type": int, "minimum": 1, "maximum": 20},
            "max_df": {"type": (int, float), "minimum": 0.5, "maximum": 1.0},
            "max_features": {"type": int, "minimum": 1000, "maximum": 200000},
            "sublinear_tf": {"type": bool},
            "C": {"type": (int, float), "minimum": 0.001, "maximum": 100.0},
            "max_iter": {"type": int, "minimum": 100, "maximum": 10000},
        },
    },
    "text_pair_char_tfidf_logistic": {
        "modality": "text_pair",
        "parameters": {
            "ngram_min": {"type": int, "minimum": 2, "maximum": 8},
            "ngram_max": {"type": int, "minimum": 2, "maximum": 8},
            "min_df": {"type": int, "minimum": 1, "maximum": 20},
            "max_features": {"type": int, "minimum": 1000, "maximum": 200000},
            "sublinear_tf": {"type": bool},
            "C": {"type": (int, float), "minimum": 0.001, "maximum": 100.0},
            "max_iter": {"type": int, "minimum": 100, "maximum": 10000},
        },
    },
    "text_pair_char_tfidf_svm": {
        "modality": "text_pair",
        "parameters": {
            "ngram_min": {"type": int, "minimum": 2, "maximum": 8},
            "ngram_max": {"type": int, "minimum": 2, "maximum": 8},
            "min_df": {"type": int, "minimum": 1, "maximum": 20},
            "max_features": {"type": int, "minimum": 1000, "maximum": 200000},
            "sublinear_tf": {"type": bool},
            "C": {"type": (int, float), "minimum": 0.001, "maximum": 100.0},
            "max_iter": {"type": int, "minimum": 100, "maximum": 10000},
        },
    },
}

for _single_name, _pair_name in {
    "text_word_tfidf_logistic": "text_pair_word_tfidf_logistic",
    "text_char_tfidf_logistic": "text_pair_char_tfidf_logistic",
    "text_char_tfidf_svm": "text_pair_char_tfidf_svm",
}.items():
    PARAMETER_RULES[_single_name] = copy.deepcopy(PARAMETER_RULES[_pair_name])
    PARAMETER_RULES[_single_name]["modality"] = "text"


class PlanValidationError(ValueError):
    pass


class PlanValidator:
    @staticmethod
    def _configuration_signature(plan):
        return json.dumps(
            {
                "adapter": plan.adapter_name,
                "hyperparameters": plan.hyperparameters,
            },
            sort_keys=True,
        )

    @staticmethod
    def allowed_adapter_specs(modality):
        specs = {}

        for adapter_name, adapter_rules in PARAMETER_RULES.items():
            if adapter_rules["modality"] != modality:
                continue

            parameters = {}
            for name, rule in adapter_rules["parameters"].items():
                if "choices" in rule:
                    parameters[name] = {
                        "required": rule.get("required", True),
                        "choices": rule["choices"],
                    }
                    continue

                expected_type = rule["type"]
                if isinstance(expected_type, tuple):
                    type_name = "number"
                elif expected_type is bool:
                    type_name = "boolean"
                elif expected_type is int:
                    type_name = "integer"
                else:
                    type_name = expected_type.__name__

                parameter_spec = {
                    "required": rule.get("required", True),
                    "type": type_name,
                }
                if "minimum" in rule:
                    parameter_spec["minimum"] = rule["minimum"]
                if "maximum" in rule:
                    parameter_spec["maximum"] = rule["maximum"]
                parameters[name] = parameter_spec

            specs[adapter_name] = {
                "modality": modality,
                "parameters": parameters,
            }

        return specs

    def validate_many(self, plans, data_profile):
        validated = []
        seen_ids = set()
        seen_configurations = set()

        for plan in plans:
            validated_plan = self.validate(plan, data_profile)
            if validated_plan.experiment_id in seen_ids:
                raise PlanValidationError(
                    f"Duplicate experiment ID: {validated_plan.experiment_id}"
                )

            signature = self._configuration_signature(validated_plan)
            if signature in seen_configurations:
                raise PlanValidationError(
                    "Duplicate experiment configuration: "
                    f"{validated_plan.adapter_name}"
                )

            seen_ids.add(validated_plan.experiment_id)
            seen_configurations.add(signature)
            validated.append(validated_plan)

        return validated

    def validate_against_history(
        self,
        plan,
        data_profile,
        existing_plans,
    ):
        validated_plan = self.validate(plan, data_profile)
        existing_ids = {
            existing_plan.experiment_id
            for existing_plan in existing_plans
        }
        if validated_plan.experiment_id in existing_ids:
            raise PlanValidationError(
                f"Duplicate experiment ID: {validated_plan.experiment_id}"
            )

        new_signature = self._configuration_signature(validated_plan)
        existing_signatures = {
            self._configuration_signature(existing_plan)
            for existing_plan in existing_plans
        }
        if new_signature in existing_signatures:
            raise PlanValidationError(
                "The LLM proposed an experiment configuration that has "
                "already been executed."
            )

        return validated_plan

    def validate(self, plan, data_profile):
        if not isinstance(plan, ExperimentPlan):
            raise PlanValidationError("Plan must be an ExperimentPlan.")
        if not plan.experiment_id:
            raise PlanValidationError("experiment_id cannot be empty.")
        if not plan.reason.strip():
            raise PlanValidationError("Every experiment needs a reason.")

        adapter_rules = PARAMETER_RULES.get(plan.adapter_name)
        if adapter_rules is None:
            raise PlanValidationError(
                f"Adapter is not allow-listed: {plan.adapter_name}"
            )

        modality = data_profile["modality"]
        if adapter_rules["modality"] != modality:
            raise PlanValidationError(
                f"Adapter {plan.adapter_name} does not support {modality}."
            )

        expected = adapter_rules["parameters"]
        provided = plan.hyperparameters
        missing = {
            name
            for name, rule in expected.items()
            if rule.get("required", True) and name not in provided
        }
        unknown = set(provided) - set(expected)
        if missing:
            raise PlanValidationError(
                f"Missing parameters for {plan.adapter_name}: {sorted(missing)}"
            )
        if unknown:
            raise PlanValidationError(
                f"Unknown parameters for {plan.adapter_name}: {sorted(unknown)}"
            )

        for name, rule in expected.items():
            if name not in provided:
                continue
            value = provided[name]
            if "choices" in rule:
                if value not in rule["choices"]:
                    raise PlanValidationError(
                        f"{name}={value!r} is not in {rule['choices']!r}."
                    )
                continue

            expected_type = rule["type"]
            if expected_type is bool:
                type_matches = isinstance(value, bool)
            else:
                type_matches = isinstance(value, expected_type) and not isinstance(
                    value, bool
                )
            if not type_matches:
                raise PlanValidationError(
                    f"{name} has invalid type {type(value).__name__}."
                )
            if "minimum" in rule and value < rule["minimum"]:
                raise PlanValidationError(
                    f"{name}={value} is below {rule['minimum']}."
                )
            if "maximum" in rule and value > rule["maximum"]:
                raise PlanValidationError(
                    f"{name}={value} is above {rule['maximum']}."
                )

        if "ngram_min" in provided and (
            provided["ngram_min"] > provided["ngram_max"]
        ):
            raise PlanValidationError(
                "ngram_min cannot be greater than ngram_max."
            )
        if adapter_rules["modality"] == "image":
            estimated, estimate_source = (
                estimate_image_training_seconds_with_source(
                    plan.adapter_name,
                    plan.hyperparameters,
                    data_profile,
                )
            )
            timeout = (data_profile.get("execution_context") or {}).get(
                "experiment_timeout_seconds"
            )
            if estimated is not None and timeout is not None:
                safe_limit = float(timeout) * 0.8
                if estimated > safe_limit:
                    raise PlanValidationError(
                        "Estimated image training time "
                        f"{estimated:.0f}s ({plan.adapter_name}, "
                        f"{plan.hyperparameters.get('image_size')}px, "
                        f"{plan.hyperparameters.get('epochs')} epochs"
                        f"{self._estimate_source_note(estimate_source)}) "
                        "exceeds 80% of the "
                        f"{timeout}s experiment timeout "
                        f"(limit {safe_limit:.0f}s). "
                        "Reduce epochs or image_size, or choose a smaller "
                        "image adapter."
                        + self._feasible_image_reference(data_profile, safe_limit)
                    )
        return plan

    @staticmethod
    def _estimate_source_note(estimate_source):
        if estimate_source == "measured":
            return ", from a measured run on this data and machine"
        return ""

    @staticmethod
    def _feasible_image_reference(data_profile, safe_limit):
        """Describe image configurations that do fit, so the planner can
        revise instead of guessing again."""
        timeout = (data_profile.get("execution_context") or {}).get(
            "experiment_timeout_seconds"
        )
        if timeout is None:
            return ""
        size = recommended_image_size(data_profile)

        entries = []
        for adapter_name, rules in PARAMETER_RULES.items():
            if rules["modality"] != "image":
                continue
            chosen = None
            for epochs in (5, 3, 2, 1):
                estimated = estimate_image_training_seconds(
                    adapter_name,
                    {
                        "image_size": size,
                        "epochs": epochs,
                        "batch_size": 32,
                    },
                    data_profile,
                )
                if estimated is not None and estimated <= safe_limit:
                    chosen = (
                        f"{adapter_name}@{size}px/{epochs} epochs "
                        f"about {estimated:.0f}s"
                    )
                    break
            if chosen is None:
                chosen = (
                    f"{adapter_name} cannot fit even 1 epoch at {size}px "
                    f"within {safe_limit:.0f}s"
                )
            entries.append(chosen)

        if not entries:
            return ""
        return (
            " Fitting configurations measured on this machine: "
            + "; ".join(entries)
            + "."
        )
