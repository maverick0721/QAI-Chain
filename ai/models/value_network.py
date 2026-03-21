import torch.nn as nn


class ValueNetwork(nn.Module):

    def __init__(self, input_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)