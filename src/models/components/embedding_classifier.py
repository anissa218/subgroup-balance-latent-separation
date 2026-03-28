import torch
from torch import nn

class EmbeddingClassifier(nn.Module):
    """Classifier head for precomputed embeddings."""

    def __init__(self, output_size: int = 2, hidden_size: int = 0, dropout: int = 0, input_size: int = 1024):
        """
        :param output_size: Number of output classes/labels.
        :param hidden_size: If provided, use an MLP (input -> hidden -> output).
        :param dropout: if simple MLP
        """
        super().__init__()

        if hidden_size == 0:
            self.classifier = nn.Linear(input_size, output_size)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Input embeddings of shape [batch_size, 1024].
        :return: (features, output) where `features` is just x (embeddings),
                 and `output` are the logits.
        """
        features = x
        out = self.classifier(features)
        return features, out

if __name__ == "__main__":
    _ = EmbeddingClassifier()
