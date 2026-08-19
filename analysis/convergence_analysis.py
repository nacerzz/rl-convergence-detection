from pathlib import Path

import pandas as pd

from load_results import load_cartpole_results


OUTPUT_DIR = Path("figures/cartpole")


def summarize_training_steps(df: pd.DataFrame) -> pd.DataFrame:
    final_steps = (
        df.groupby(["method", "seed"])["timesteps"]
        .max()
        .reset_index()
    )

    summary = (
        final_steps.groupby("method")["timesteps"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    return summary


def main() -> None:
    df = load_cartpole_results()

    summary = summarize_training_steps(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "convergence_summary.csv", index=False)

    print("\nTraining step summary:")
    print(summary)

    auto_steps = summary.loc[
        summary["method"] == "Automatic stopping", "mean"
    ].iloc[0]

    fixed_steps = summary.loc[
        summary["method"] == "Fixed budget", "mean"
    ].iloc[0]

    saved_steps = fixed_steps - auto_steps
    saved_percent = 100 * saved_steps / fixed_steps

    print(f"\nAutomatic stopping mean steps: {auto_steps:.0f}")
    print(f"Fixed budget mean steps: {fixed_steps:.0f}")
    print(f"Saved steps: {saved_steps:.0f}")
    print(f"Saved percentage: {saved_percent:.1f}%")


if __name__ == "__main__":
    main()