import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        in_channels: int = 3,
        output_size: int = 2,
        freeze_backbone: bool = False
    ) -> None:
        """Initialize a `SimpleCNN` module.

        :param input_size: The number of input channels.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=in_channels,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7,output_size)

        if freeze_backbone:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.conv2.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        batch_size, channels, width, height = x.size()

        x = self.conv1(x)
        x = self.conv2(x)
        features = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(features)

        return features, output

if __name__ == "__main__":
    _ = SimpleCNN()