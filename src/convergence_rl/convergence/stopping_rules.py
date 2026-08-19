import numpy as np


def has_converged(
    records: list[dict],
    patience: int = 3,
    min_mean_return: float = 380.0,
    max_mean_rnd_error: float = 0.0002,
    max_entropy_change: float = 0.04,
) -> bool:
    if len(records) < patience:
        return False

    recent = records[-patience:]

    returns = [r["eval_return_mean"] for r in recent]
    rnd_errors = [r["rnd_prediction_error"] for r in recent]
    entropies = [r["policy_entropy"] for r in recent]

    mean_return = float(np.mean(returns))
    mean_rnd_error = float(np.mean(rnd_errors))
    entropy_change = float(max(entropies) - min(entropies))

    return (
        mean_return >= min_mean_return
        and mean_rnd_error <= max_mean_rnd_error
        and entropy_change <= max_entropy_change
    )