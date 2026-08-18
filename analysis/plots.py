from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("results/cartpole")
FIGURE_DIR = Path("figures/cartpole")


def load_results() -> pd.DataFrame:
    csv_files = list(RESULT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in results/cartpole/. "
            "Run experiments/run_experiment.py first."
        )

    dataframes = []

    for file in csv_files:
        df = pd.read_csv(file)

        if "fixed_budget" in file.name:
            df["method"] = "Fixed budget"
        elif "auto_stop" in file.name:
            df["method"] = "Automatic stopping"
        else:
            df["method"] = "Unknown"

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))

    for method, method_df in df.groupby("method"):
        grouped = (
            method_df.groupby("timesteps")[metric]
            .agg(["mean", "std"])
            .reset_index()
        )

        plt.plot(
            grouped["timesteps"],
            grouped["mean"],
            label=method,
        )

        plt.fill_between(
            grouped["timesteps"],
            grouped["mean"] - grouped["std"].fillna(0),
            grouped["mean"] + grouped["std"].fillna(0),
            alpha=0.2,
        )

    plt.xlabel("Training steps")
    plt.ylabel(ylabel)
    plt.title(ylabel + " during PPO training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def plot_stopping_steps(df: pd.DataFrame) -> None:
    final_steps = (
        df.groupby(["method", "seed"])["timesteps"]
        .max()
        .reset_index()
    )

    summary = (
        final_steps.groupby("method")["timesteps"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(7, 5))
    plt.bar(summary["method"], summary["mean"])
    plt.ylabel("Training steps")
    plt.title("Training steps used by each method")
    plt.grid(axis="y")
    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / "training_steps_comparison.png", dpi=300)
    plt.close()

    summary.to_csv(FIGURE_DIR / "training_steps_summary.csv", index=False)


def main() -> None:
    df = load_results()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(FIGURE_DIR / "combined_cartpole_results.csv", index=False)

    plot_metric(
        df,
        metric="eval_return_mean",
        ylabel="Evaluation return",
        filename="evaluation_return.png",
    )

    plot_metric(
        df,
        metric="rnd_prediction_error",
        ylabel="RND prediction error",
        filename="rnd_prediction_error.png",
    )

    plot_metric(
        df,
        metric="policy_entropy",
        ylabel="Policy entropy",
        filename="policy_entropy.png",
    )

    plot_stopping_steps(df)

    print(f"Plots saved in {FIGURE_DIR}")


if __name__ == "__main__":
    main()