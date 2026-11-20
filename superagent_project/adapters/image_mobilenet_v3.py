import torch
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

from adapters.image_base import ImageClassificationAdapter


class MobileNetV3Adapter(ImageClassificationAdapter):
    adapter_name = "image_mobilenet_v3_small"
    model_name = "mobilenet_v3_small_classifier"
    architecture_name = "mobilenet_v3_small"

    def build_model(self, number_of_classes, hyperparameters):
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if hyperparameters["pretrained"]
            else None
        )
        model = mobilenet_v3_small(weights=weights)
        final_layer = model.classifier[-1]
        model.classifier[-1] = torch.nn.Linear(
            final_layer.in_features,
            number_of_classes,
        )
        head_parameter_ids = {
            id(value) for value in model.classifier.parameters()
        }
        backbone_parameters = [
            value
            for value in model.parameters()
            if id(value) not in head_parameter_ids
        ]
        return model, backbone_parameters
