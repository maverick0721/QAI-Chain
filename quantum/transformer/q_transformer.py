import torch.nn as nn

from quantum.transformer.q_transformer_layer import QTransformerLayer


class QTransformer(nn.Module):

    def __init__(self, input_dim=5, dim=16, num_layers=2):
        super().__init__()

        self.embedding = nn.Linear(input_dim, dim)

        self.layers = nn.ModuleList([
            QTransformerLayer(dim) for _ in range(num_layers)
        ])

        self.output = nn.Linear(dim, 1)

    def forward(self, x):

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output(x)

        return x