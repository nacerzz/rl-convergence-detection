from pathlib import Path

import pandas as pd


def load_cartpole_results(result_dir: str = "results/cartpole") -> pd.DataFrame:
    result_path = Path(result_dir)
    csv_files = list(result_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {result_dir}")

    dataframes = []

    for file in csv_files:
        df = pd.read_csv(file)

        if "auto_stop" in file.name:
            df["method"] = "Automatic stopping"
        elif "fixed_budget" in file.name:
            df["method"] = "Fixed budget"
        else:
            df["method"] = "Unknown"

        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)