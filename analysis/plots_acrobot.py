from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("results/acrobot")
FIGURE_DIR = Path("figures/acrobot")


def load_results() -> pd.DataFrame:
    csv_files = list(RESULT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in results/acrobot/. "
            "Run experiments/run_acrobot.py first."
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
        method_df = method_df.sort_values("timesteps")

        plt.plot(
            method_df["timesteps"],
            method_df[metric],
            label=method,
        )

    plt.xlabel("Training steps")
    plt.ylabel(ylabel)
    plt.title(ylabel + " during PPO training on Acrobot-v1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def main() -> None:
    df = load_results()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(FIGURE_DIR / "combined_acrobot_results.csv", index=False)

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

    plot_metric(
        df,
        metric="visited_states",
        ylabel="Visited states",
        filename="visited_states.png",
    )

    print(f"Acrobot plots saved in {FIGURE_DIR}")


if __name__ == "__main__":
    main()