import torch
from torchvision.models import ResNet18_Weights, resnet18

from adapters.image_base import ImageClassificationAdapter


class ResNet18Adapter(ImageClassificationAdapter):
    adapter_name = "image_resnet18"
    model_name = "resnet18_classifier"
    architecture_name = "resnet18"

    def build_model(self, number_of_classes, hyperparameters):
        weights = (
            ResNet18_Weights.DEFAULT
            if hyperparameters["pretrained"]
            else None
        )
        model = resnet18(weights=weights)
        input_features = model.fc.in_features
        model.fc = torch.nn.Linear(input_features, number_of_classes)
        head_parameter_ids = {id(value) for value in model.fc.parameters()}
        backbone_parameters = [
            value
            for value in model.parameters()
            if id(value) not in head_parameter_ids
        ]
        return model, backbone_parameters
