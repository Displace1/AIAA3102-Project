"""The deterministic tabular baseline must not pre-weight classes.

Background: exp_002 in ``default_plans_for`` used to set
``class_weight="balanced"``. Because the LLM planner reads the deterministic
baseline's configuration as prior evidence and rarely revisits it, that single
default propagated into almost every later plan. Measured cost on Covertype
(selection metric = accuracy):

    ExtraTrees  balanced vs None : -0.6pp accuracy
    HGB         balanced vs None : -2.3 to -5.6pp accuracy

"balanced" only helps when the task's metric rewards per-class recall, which the
agent cannot know at plan time, so the neutral default is the safe one. These
tests keep it neutral and keep the choice explicit in the allow-list.
"""
import unittest

from superagent.experiment_registry import ExperimentRegistry


def tabular_profile(**overrides):
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
        "metric": "accuracy",
    }
    profile.update(overrides)
    return profile


class BaselineClassWeightTest(unittest.TestCase):
    def test_deterministic_tree_baseline_is_unweighted(self):
        plans = ExperimentRegistry(random_seed=42).default_plans_for(
            tabular_profile())
        tree_plan = next(
            p for p in plans if p.adapter_name == "tabular_extra_trees")
        self.assertEqual(
            tree_plan.hyperparameters["class_weight"], None,
            "the deterministic baseline must not pre-weight classes; see this "
            "module's docstring for the measured cost")

    def test_linear_baseline_stays_unweighted(self):
        plans = ExperimentRegistry(random_seed=42).default_plans_for(
            tabular_profile())
        linear = next(
            p for p in plans if p.adapter_name == "tabular_one_hot_logistic")
        self.assertEqual(linear.hyperparameters["class_weight"], None)

    def test_weighting_remains_reachable_through_the_allow_list(self):
        """Neutral by default, but the planner may still choose balanced."""
        from superagent.plan_validator import PlanValidator

        specs = PlanValidator.allowed_adapter_specs("tabular")
        for adapter in ("tabular_extra_trees", "tabular_random_forest",
                        "tabular_hist_gradient_boosting"):
            self.assertIn(adapter, specs)
            self.assertIn(
                "balanced", specs[adapter]["parameters"]["class_weight"]["choices"],
                f"{adapter} must still allow balanced so the planner can pick it")

    def test_hgb_baseline_default_is_neutral_when_present(self):
        """If HGB ever appears in the deterministic plans, it must be neutral."""
        plans = ExperimentRegistry(random_seed=42).default_plans_for(
            tabular_profile())
        for plan in plans:
            if plan.adapter_name == "tabular_hist_gradient_boosting":
                self.assertEqual(plan.hyperparameters.get("class_weight"), None)


if __name__ == "__main__":
    unittest.main()
