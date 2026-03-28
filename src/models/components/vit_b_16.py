import torch
from torch import nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from src.utils import (
    RankedLogger
)

log = RankedLogger(__name__, rank_zero_only=True)

class ViT_B16(nn.Module):
    """PyTorch Vision Transformer (ViT-B/16)"""

    def __init__(
        self,
        output_size: int = 2,
        freeze_backbone: bool = False
    ) -> None:
        """
        Initialize a pretrained ViT-B/16 module.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # replace the classification head
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_features, output_size)

        if freeze_backbone:
            log.info("Freezing backbone params ...")
            for name, param in self.model.named_parameters():
                if not name.startswith("heads"):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tuple (features, output), where `features` are before classification.
        """
        # process input patches and add class token
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # pass through transformer encoder
        x = self.model.encoder(x)

        # CLS token embedding (features before classification)
        features = x[:, 0]

        out = self.model.heads(features)

        return features, out


if __name__ == "__main__":
    _ = ViT_B16()