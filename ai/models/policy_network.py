import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):

    def __init__(self, input_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
        )

        self.mean = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        x = self.net(x)

        mean = self.mean(x)
        std = torch.exp(self.log_std)

        return mean, std