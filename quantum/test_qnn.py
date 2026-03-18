from pathlib import Path
import sys

# Allow running this script directly from quantum/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from quantum.models.qnn import QNN


model = QNN()

x = torch.randn(2, 5).float()

out = model(x)

print("Quantum Output:", out)