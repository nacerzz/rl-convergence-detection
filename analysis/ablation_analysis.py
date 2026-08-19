from pathlib import Path

import numpy as np
import pandas as pd

from load_results import load_cartpole_results


OUTPUT_DIR = Path("figures/cartpole")


def find_stopping_step(
    seed_df: pd.DataFrame,
    rule_name: str,
    patience: int = 3,
    min_mean_return: float = 380.0,
    max_mean_rnd_error: float = 0.0002,
    max_entropy_change: float = 0.04,
) -> int:
    records = seed_df.sort_values("timesteps").to_dict("records")

    for i in range(patience - 1, len(records)):
        recent = records[i - patience + 1 : i + 1]

        returns = [r["eval_return_mean"] for r in recent]
        rnd_errors = [r["rnd_prediction_error"] for r in recent]
        entropies = [r["policy_entropy"] for r in recent]

        mean_return = float(np.mean(returns))
        mean_rnd_error = float(np.mean(rnd_errors))
        entropy_change = float(max(entropies) - min(entropies))

        if rule_name == "return_only":
            stop = mean_return >= min_mean_return
        elif rule_name == "rnd_only":
            stop = mean_rnd_error <= max_mean_rnd_error
        elif rule_name == "entropy_only":
            stop = entropy_change <= max_entropy_change
        elif rule_name == "combined":
            stop = (
                mean_return >= min_mean_return
                and mean_rnd_error <= max_mean_rnd_error
                and entropy_change <= max_entropy_change
            )
        else:
            raise ValueError(f"Unknown rule: {rule_name}")

        if stop:
            return int(records[i]["timesteps"])

    return int(records[-1]["timesteps"])


def main() -> None:
    df = load_cartpole_results()

    fixed_df = df[df["method"] == "Fixed budget"].copy()

    rules = [
        "return_only",
        "rnd_only",
        "entropy_only",
        "combined",
    ]

    rows = []

    for rule in rules:
        for seed, seed_df in fixed_df.groupby("seed"):
            stopping_step = find_stopping_step(seed_df, rule_name=rule)

            rows.append(
                {
                    "rule": rule,
                    "seed": int(seed),
                    "stopping_step": stopping_step,
                }
            )

    result_df = pd.DataFrame(rows)

    summary = (
        result_df.groupby("rule")["stopping_step"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(OUTPUT_DIR / "ablation_stopping_steps.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)

    print("\nAblation stopping steps:")
    print(result_df)

    print("\nAblation summary:")
    print(summary)


if __name__ == "__main__":
    main()