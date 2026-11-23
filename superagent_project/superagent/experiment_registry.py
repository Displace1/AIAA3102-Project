from adapters.image_mobilenet_v3 import MobileNetV3Adapter
from adapters.image_resnet18 import ResNet18Adapter
from adapters.image_small_cnn import SmallCNNAdapter
from adapters.tabular_baseline import TabularBaselineAdapter
from adapters.tabular_extra_trees import TabularExtraTreesAdapter
from adapters.tabular_random_forest import TabularRandomForestAdapter
from adapters.text_pair_baseline import TextPairBaselineAdapter, TextWordTfidfAdapter
from adapters.text_pair_char_tfidf import (
    TextCharTfidfAdapter,
    TextPairCharTfidfAdapter,
)
from adapters.text_pair_char_svm import TextCharSvmAdapter, TextPairCharSvmAdapter
from superagent.experiment_plan import ExperimentPlan
from superagent.image_runtime import recommended_image_size


class ExperimentRegistry:
    """Builds modality-compatible plans and allow-listed adapters."""

    def __init__(self, random_seed):
        self.random_seed = random_seed
        self._factories = {
            "tabular_one_hot_logistic": lambda: TabularBaselineAdapter(
                random_seed=self.random_seed
            ),
            "tabular_extra_trees": lambda: TabularExtraTreesAdapter(
                random_seed=self.random_seed
            ),
            "tabular_random_forest": lambda: TabularRandomForestAdapter(
                random_seed=self.random_seed
            ),
            "text_pair_word_tfidf_logistic": lambda: (
                TextPairBaselineAdapter(random_seed=self.random_seed)
            ),
            "text_pair_char_tfidf_logistic": lambda: (
                TextPairCharTfidfAdapter(random_seed=self.random_seed)
            ),
            "text_pair_char_tfidf_svm": lambda: TextPairCharSvmAdapter(
                random_seed=self.random_seed
            ),
            "text_word_tfidf_logistic": lambda: TextWordTfidfAdapter(
                random_seed=self.random_seed
            ),
            "text_char_tfidf_logistic": lambda: TextCharTfidfAdapter(
                random_seed=self.random_seed
            ),
            "text_char_tfidf_svm": lambda: TextCharSvmAdapter(
                random_seed=self.random_seed
            ),
            "image_small_cnn": lambda: SmallCNNAdapter(
                random_seed=self.random_seed
            ),
            "image_resnet18": lambda: ResNet18Adapter(
                random_seed=self.random_seed
            ),
            "image_mobilenet_v3_small": lambda: MobileNetV3Adapter(
                random_seed=self.random_seed
            ),
        }

    def adapter_for(self, adapter_name):
        factory = self._factories.get(adapter_name)
        if factory is None:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        return factory()

    def default_plans_for(self, data_profile):
        modality = data_profile["modality"]

        if modality == "tabular":
            return [
                ExperimentPlan(
                    experiment_id="exp_001",
                    adapter_name="tabular_one_hot_logistic",
                    hyperparameters={
                        "max_iter": 2000,
                        "C": 1.0,
                        "numeric_preprocessing": "standard",
                        "categorical_preprocessing": "onehot",
                        "ordinal_preprocessing": "ordinal",
                        "class_weight": None,
                    },
                    reason=(
                        "Establish a linear one-hot baseline for "
                        "categorical tabular data."
                    ),
                ),
                ExperimentPlan(
                    experiment_id="exp_002",
                    adapter_name="tabular_extra_trees",
                    hyperparameters={
                        "n_estimators": 400,
                        "max_features": "sqrt",
                        "min_samples_leaf": 2,
                        "class_weight": "balanced",
                        "numeric_preprocessing": "none",
                        "categorical_preprocessing": "onehot",
                        "ordinal_preprocessing": "ordinal",
                    },
                    reason=(
                        "Test nonlinear interactions among categorical "
                        "features with randomized trees."
                    ),
                ),
            ]
        if modality == "text_pair":
            return [
                ExperimentPlan(
                    experiment_id="exp_001",
                    adapter_name="text_pair_word_tfidf_logistic",
                    hyperparameters={
                        "ngram_min": 1,
                        "ngram_max": 2,
                        "min_df": 2,
                        "max_df": 0.98,
                        "max_features": 50000,
                        "sublinear_tf": True,
                        "C": 1.0,
                        "max_iter": 2000,
                    },
                    reason=(
                        "Establish a word-level pairwise TF-IDF baseline."
                    ),
                ),
                ExperimentPlan(
                    experiment_id="exp_002",
                    adapter_name="text_pair_char_tfidf_logistic",
                    hyperparameters={
                        "ngram_min": 3,
                        "ngram_max": 5,
                        "min_df": 2,
                        "max_features": 80000,
                        "sublinear_tf": True,
                        "C": 1.0,
                        "max_iter": 2000,
                    },
                    reason=(
                        "Test whether character patterns, punctuation, "
                        "numbers, and headline style improve prediction."
                    ),
                ),
            ]

        if modality == "text":
            return [
                ExperimentPlan(
                    experiment_id="exp_001",
                    adapter_name="text_word_tfidf_logistic",
                    hyperparameters={
                        "ngram_min": 1,
                        "ngram_max": 2,
                        "min_df": 2,
                        "max_df": 0.98,
                        "max_features": 50000,
                        "sublinear_tf": True,
                        "C": 1.0,
                        "max_iter": 2000,
                    },
                    reason="Establish a word-level TF-IDF text baseline.",
                ),
                ExperimentPlan(
                    experiment_id="exp_002",
                    adapter_name="text_char_tfidf_logistic",
                    hyperparameters={
                        "ngram_min": 3,
                        "ngram_max": 5,
                        "min_df": 2,
                        "max_features": 80000,
                        "sublinear_tf": True,
                        "C": 1.0,
                        "max_iter": 2000,
                    },
                    reason="Test character TF-IDF for robust text patterns.",
                ),
            ]

        if modality == "image":
            image_size = recommended_image_size(data_profile)
            return [
                ExperimentPlan(
                    experiment_id="exp_001",
                    adapter_name="image_small_cnn",
                    hyperparameters={
                        "image_size": image_size,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "epochs": 15,
                        "data_augmentation": True,
                        "early_stopping_patience": 4,
                        "weight_decay": 0.0001,
                    },
                    reason=(
                        "Establish a compact CNN image baseline without "
                        "external pretrained weights."
                    ),
                ),
                ExperimentPlan(
                    experiment_id="exp_002",
                    adapter_name="image_resnet18",
                    hyperparameters={
                        "image_size": image_size,
                        "batch_size": 32,
                        "learning_rate": 0.0003,
                        "epochs": 12,
                        "data_augmentation": True,
                        "early_stopping_patience": 3,
                        "weight_decay": 0.0001,
                        "pretrained": True,
                        "freeze_backbone_epochs": 2,
                    },
                    reason=(
                        "Evaluate ImageNet-pretrained ResNet18 transfer "
                        "learning with initial backbone freezing."
                    ),
                ),
            ]

        raise ValueError(f"No experiment plans support modality: {modality}")
