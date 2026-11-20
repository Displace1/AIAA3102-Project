import unittest

from superagent.experiment_plan import ExperimentPlan
from superagent.plan_validator import (
    PlanValidationError,
    PlanValidator,
)


class PlanValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = PlanValidator()
        self.tabular_profile = {"modality": "tabular"}

    def valid_plan(self, experiment_id="exp_001"):
        return ExperimentPlan(
            experiment_id=experiment_id,
            adapter_name="tabular_one_hot_logistic",
            hyperparameters={"max_iter": 2000, "C": 1.0},
            reason="Establish a baseline.",
        )

    def test_accepts_valid_plan(self):
        plan = self.valid_plan()
        self.assertIs(
            self.validator.validate(plan, self.tabular_profile),
            plan,
        )

    def test_rejects_unknown_parameter(self):
        plan = ExperimentPlan(
            experiment_id="exp_001",
            adapter_name="tabular_one_hot_logistic",
            hyperparameters={
                "max_iter": 2000,
                "C": 1.0,
                "read_test_labels": True,
            },
            reason="Invalid plan.",
        )
        with self.assertRaises(PlanValidationError):
            self.validator.validate(plan, self.tabular_profile)

    def test_rejects_out_of_range_value(self):
        plan = ExperimentPlan(
            experiment_id="exp_001",
            adapter_name="tabular_one_hot_logistic",
            hyperparameters={"max_iter": 10, "C": 1.0},
            reason="Invalid plan.",
        )
        with self.assertRaises(PlanValidationError):
            self.validator.validate(plan, self.tabular_profile)

    def test_rejects_modality_mismatch(self):
        plan = self.valid_plan()
        with self.assertRaises(PlanValidationError):
            self.validator.validate(plan, {"modality": "text_pair"})

    def test_rejects_duplicate_experiment_ids(self):
        first = self.valid_plan("exp_001")
        second = ExperimentPlan(
            experiment_id="exp_001",
            adapter_name="tabular_extra_trees",
            hyperparameters={
                "n_estimators": 400,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
            reason="Try a nonlinear model.",
        )
        with self.assertRaises(PlanValidationError):
            self.validator.validate_many(
                [first, second],
                self.tabular_profile,
            )


if __name__ == "__main__":
    unittest.main()
