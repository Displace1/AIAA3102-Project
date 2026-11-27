import torch

from adapters.image_base import ImageClassificationAdapter


class SmallCNN(torch.nn.Module):
    """Compact classifier suitable for a dependency-free image baseline."""

    def __init__(self, number_of_classes):
        super().__init__()
        self.features = torch.nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(128, number_of_classes),
        )

    @staticmethod
    def _block(input_channels, output_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs):
        return self.classifier(self.features(inputs))


class SmallCNNAdapter(ImageClassificationAdapter):
    adapter_name = "image_small_cnn"
    model_name = "small_cnn"
    architecture_name = "small_cnn"

    def build_model(self, number_of_classes, hyperparameters):
        del hyperparameters
        return SmallCNN(number_of_classes), []
