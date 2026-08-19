import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from convergence_rl.signals.rnd import RNDModel


def test_rnd_update_returns_float():
    rnd = RNDModel(obs_dim=4)
    obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    error = rnd.update(obs)

    assert isinstance(error, float)
    assert error >= 0.0