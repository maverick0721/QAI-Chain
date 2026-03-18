import torch
import torch.nn as nn

from quantum.attention.quantum_attention import QuantumAttention


class QTransformerLayer(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.attn = QuantumAttention()

        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):

        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x