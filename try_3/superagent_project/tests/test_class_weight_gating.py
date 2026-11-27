"""class_weight is only offered when the declared metric can use it.

Background
----------
``balanced`` reweights the loss toward rare classes. That aligns with a metric
that averages over classes (macro_f1, balanced_accuracy, recall) and is at best
neutral, usually harmful, for plain accuracy.

Measured on Covertype (7 classes, accuracy metric) over six runs of one config:
the share of experiments that used ``balanced`` correlated with test accuracy at
r = -0.989, and the two runs that used it every time scored 3.1pp below the runs
that never did.

KDD99 (12 classes, 3 holding 98% of rows) is the counter-example that keeps this
from being a blanket ban: there the same flag raised macro-F1 by 0.25pp. So the
right response is to gate it on the METRIC, not to delete it.
"""
import unittest

from superagent.experiment_plan import ExperimentPlan
from superagent.plan_validator import (
    PARAMETER_RULES,
    PlanValidationError,
    PlanValidator,
)

BALANCE_AWARE = ("macro_f1", "balanced_accuracy", "recall")
ACCURACY_LIKE = ("accuracy", "roc_auc", "log_loss")
TABULAR_TREE_ADAPTERS = ("tabular_extra_trees", "tabular_random_forest",
                         "tabular_hist_gradient_boosting")

# minority_rate values measured on this project's tasks.
KDD99_MINORITY = 0.00036    # 12 classes, smallest holds 8/22500
COVERTYPE_MINORITY = 0.0047  # 7 classes, smallest ~0.47%
BANK_MINORITY = 0.1169
TASK1_MINORITY = 0.4316


def tabular_profile(metric, minority_rate=None):
    profile = {
        "modality": "tabular",
        "task_type": "binary_classification",
        "data_format": "npy_split",
        "train_samples": 100,
        "validation_samples": 50,
        "n_features": 3,
        "numeric_column_indices": [0, 1],
        "ordinal_column_indices": [],
        "categorical_column_indices": [2],
        "class_distribution": {"0": 60, "1": 40},
        "metric": metric,
    }
    if minority_rate is not None:
        profile["minority_rate"] = minority_rate
    return profile


class AdapterSpecGatingTest(unittest.TestCase):
    def test_balanced_hidden_for_accuracy(self):
        specs = PlanValidator.allowed_adapter_specs("tabular", metric="accuracy")
        for adapter in TABULAR_TREE_ADAPTERS:
            choices = specs[adapter]["parameters"]["class_weight"]["choices"]
            self.assertNotIn("balanced", choices,
                             f"{adapter} must not offer balanced for accuracy")
            self.assertIn(None, choices,
                          "the parameter must stay visible with the neutral option")

    def test_balanced_visible_for_balance_aware_metrics(self):
        for metric in BALANCE_AWARE:
            specs = PlanValidator.allowed_adapter_specs("tabular", metric=metric)
            choices = specs["tabular_extra_trees"]["parameters"]["class_weight"]["choices"]
            self.assertIn("balanced", choices,
                          f"{metric} should keep balanced selectable")

    def test_balanced_hidden_for_other_accuracy_like_metrics(self):
        for metric in ACCURACY_LIKE:
            specs = PlanValidator.allowed_adapter_specs("tabular", metric=metric)
            choices = specs["tabular_extra_trees"]["parameters"]["class_weight"]["choices"]
            self.assertNotIn("balanced", choices, f"{metric} should hide balanced")

    def test_no_metric_means_no_filtering(self):
        """Without a declared metric the schema stays permissive."""
        specs = PlanValidator.allowed_adapter_specs("tabular")
        choices = specs["tabular_extra_trees"]["parameters"]["class_weight"]["choices"]
        self.assertIn("balanced", choices)
        self.assertIn(None, choices)


class ExtremeImbalanceTest(unittest.TestCase):
    """An extremely rare smallest class justifies weighting even under accuracy.

    This is the KDD99 case: metric is accuracy, yet hiding balanced cost 4.91pp
    of macro-F1 and cut the rarest class's recall from 0.875 to 0.364.
    """

    def choices_for(self, metric, minority_rate):
        specs = PlanValidator.allowed_adapter_specs(
            "tabular", metric=metric, minority_rate=minority_rate)
        return specs["tabular_extra_trees"]["parameters"]["class_weight"]["choices"]

    def test_kdd99_like_imbalance_keeps_balanced(self):
        choices = self.choices_for("accuracy", KDD99_MINORITY)
        self.assertIn("balanced", choices,
                      "a 0.036% smallest class must keep balanced available")

    def test_covertype_imbalance_hides_balanced(self):
        choices = self.choices_for("accuracy", COVERTYPE_MINORITY)
        self.assertNotIn("balanced", choices,
                         "0.47% is common enough that weighting hurt accuracy")

    def test_bank_and_task1_hide_balanced(self):
        for rate in (BANK_MINORITY, TASK1_MINORITY):
            choices = self.choices_for("accuracy", rate)
            self.assertNotIn("balanced", choices, f"minority {rate} should hide")

    def test_validator_accepts_balanced_for_kdd99_like_data(self):
        plan = ExperimentPlan(
            experiment_id="exp_003",
            adapter_name="tabular_extra_trees",
            hyperparameters={
                "n_estimators": 200, "max_features": "sqrt",
                "min_samples_leaf": 2, "class_weight": "balanced",
            },
            reason="Extreme imbalance justifies weighting.",
        )
        result = PlanValidator().validate(
            plan, tabular_profile("accuracy", KDD99_MINORITY))
        self.assertEqual(result.hyperparameters["class_weight"], "balanced")

    def test_validator_rejects_balanced_for_covertype_like_data(self):
        plan = ExperimentPlan(
            experiment_id="exp_003",
            adapter_name="tabular_extra_trees",
            hyperparameters={
                "n_estimators": 200, "max_features": "sqrt",
                "min_samples_leaf": 2, "class_weight": "balanced",
            },
            reason="Covertype-like imbalance does not justify weighting.",
        )
        with self.assertRaises(PlanValidationError):
            PlanValidator().validate(
                plan, tabular_profile("accuracy", COVERTYPE_MINORITY))

    def test_missing_minority_rate_falls_back_to_metric_rule(self):
        """Unknown imbalance must not silently widen the option."""
        plan = ExperimentPlan(
            experiment_id="exp_003",
            adapter_name="tabular_extra_trees",
            hyperparameters={
                "n_estimators": 200, "max_features": "sqrt",
                "min_samples_leaf": 2, "class_weight": "balanced",
            },
            reason="No minority_rate supplied.",
        )
        with self.assertRaises(PlanValidationError):
            PlanValidator().validate(plan, tabular_profile("accuracy"))


class ValidatorEnforcementTest(unittest.TestCase):
    """The validator must reject balanced when the planner proposes it anyway."""

    def plan(self, class_weight):
        return ExperimentPlan(
            experiment_id="exp_003",
            adapter_name="tabular_extra_trees",
            hyperparameters={
                "n_estimators": 200,
                "max_features": "sqrt",
                "min_samples_leaf": 2,
                "class_weight": class_weight,
            },
            reason="Testing class_weight gating.",
        )

    def test_rejects_balanced_for_accuracy(self):
        with self.assertRaises(PlanValidationError) as ctx:
            PlanValidator().validate(self.plan("balanced"),
                                     tabular_profile("accuracy"))
        message = str(ctx.exception)
        self.assertIn("class_weight", message)
        self.assertIn("macro_f1", message, "the error should name a usable metric")

    def test_accepts_none_for_accuracy(self):
        result = PlanValidator().validate(self.plan(None),
                                          tabular_profile("accuracy"))
        self.assertEqual(result.hyperparameters["class_weight"], None)

    def test_accepts_balanced_for_macro_f1(self):
        result = PlanValidator().validate(self.plan("balanced"),
                                          tabular_profile("macro_f1"))
        self.assertEqual(result.hyperparameters["class_weight"], "balanced")

    def test_accepts_balanced_for_balanced_accuracy(self):
        result = PlanValidator().validate(self.plan("balanced"),
                                          tabular_profile("balanced_accuracy"))
        self.assertEqual(result.hyperparameters["class_weight"], "balanced")

    def test_accepts_balanced_when_metric_absent(self):
        profile = tabular_profile("accuracy")
        del profile["metric"]
        result = PlanValidator().validate(self.plan("balanced"), profile)
        self.assertEqual(result.hyperparameters["class_weight"], "balanced")


class RuleTableUnchangedTest(unittest.TestCase):
    def test_parameter_rules_still_list_balanced(self):
        """The allow-list itself keeps both options; only the view is filtered."""
        for adapter in TABULAR_TREE_ADAPTERS:
            choices = PARAMETER_RULES[adapter]["parameters"]["class_weight"]["choices"]
            self.assertIn("balanced", choices)
            self.assertIn(None, choices)


if __name__ == "__main__":
    unittest.main()
