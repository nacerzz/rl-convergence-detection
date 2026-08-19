import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from convergence_rl.convergence.stopping_rules import has_converged


def test_has_converged_true():
    records = [
        {
            "eval_return_mean": 390.0,
            "rnd_prediction_error": 0.0001,
            "policy_entropy": 0.60,
        },
        {
            "eval_return_mean": 400.0,
            "rnd_prediction_error": 0.0001,
            "policy_entropy": 0.59,
        },
        {
            "eval_return_mean": 395.0,
            "rnd_prediction_error": 0.0001,
            "policy_entropy": 0.58,
        },
    ]

    assert has_converged(records) is True


def test_has_converged_false_when_return_low():
    records = [
        {
            "eval_return_mean": 100.0,
            "rnd_prediction_error": 0.0001,
            "policy_entropy": 0.60,
        },
        {
            "eval_return_mean": 120.0,
            "rnd_prediction_error": 0.0001,
            "policy_entropy": 0.59,
        },
        {
            "eval_return_mean": 130.0,
            "rnd_prediction_error": 0.0001,
            "policy_entropy": 0.58,
        },
    ]

    assert has_converged(records) is False