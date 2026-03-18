from pathlib import Path
import sys

# Allow running this script directly from ai/rl/.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.blockchain.blockchain import Blockchain
from ai.rl.trainer import train


blockchain = Blockchain()

train(blockchain)
