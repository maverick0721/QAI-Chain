import torch.nn as nn


class MetricsEncoder(nn.Module):

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)