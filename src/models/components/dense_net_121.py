import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights
import torch.nn.functional as F
from src.utils import (
    RankedLogger
)

log = RankedLogger(__name__, rank_zero_only=True)

class DenseNet121(nn.Module):
    """Pytorch DenseNet121"""

    def __init__(
        self,
        output_size: int = 2,
        freeze_backbone: bool = False
    ) -> None:
        """Initialize a pretrained `DenseNet` module.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_features = self.model.classifier.in_features

        self.model.classifier = nn.Linear(num_features, output_size)
        
        if freeze_backbone:
            log.info('frozen backbone!')
            for param in self.model.features.parameters():
                param.requires_grad = False
        else:
            log.info('not freezing bacbone')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tuple (features, output), where `features` are before classification.
        """
        # batch_size, channels, width, height = x.size()

        features = self.model.features(x)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        out = self.model.classifier(features)

        return features, out

if __name__ == "__main__":
    _ = DenseNet121()
