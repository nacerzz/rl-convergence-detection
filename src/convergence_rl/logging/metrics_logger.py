import os

import pandas as pd


def save_records(records: list[dict], result_path: str) -> None:
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(result_path, index=False)