"""
Module: Data Loader
-------------------
Role: Ingest raw data from CSV. If the file is missing, generates a
      deterministic dummy dataset so the pipeline runs end-to-end.
Input: Path to CSV file.
Output: pandas DataFrame (raw).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Load the raw NHL player statistics CSV file.

    If the file does not exist, a dummy dataset compatible with the
    default SETTINGS in main.py is generated and saved so the full
    pipeline can run out-of-the-box.

    Args:
        raw_data_path: Path to the raw CSV data.

    Returns:
        Raw DataFrame with player statistics.
    """
    raw_data_path = Path(raw_data_path)

    if raw_data_path.exists():
        df = load_csv(raw_data_path)
        print(f"[load_data] Loaded {len(df)} rows from {raw_data_path}")
        return df

    # --- Dummy baseline so the script runs end-to-end ---
    print(f"[load_data] {raw_data_path} not found — generating dummy data.")
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Rank": range(1, n + 1),
        "Team": np.random.choice(["Team A", "Team B", "Team C"], n),
        "Name": [f"Player_{i}" for i in range(1, n + 1)],
        "Pos": np.random.choice(["C", "D", "L", "R"], n),
        "Games_Played": np.random.randint(50, 82, n),
        "Icetime_Minutes": np.random.uniform(800, 1800, n).round(1),
        "Goals": np.random.randint(0, 40, n),
        "Assists": np.random.randint(0, 50, n),
        "Primary_Assists": np.random.randint(0, 30, n),
        "Secondary_Assists": np.random.randint(0, 20, n),
        "Faceoff_Win_Pct": np.random.uniform(30, 65, n).round(1),
        "Takeaways": np.random.randint(5, 60, n),
        "Giveaways": np.random.randint(10, 80, n),
        "Shot_Attempts": np.random.randint(80, 400, n),
        "Shooting_Pct_On_Unblocked": np.random.uniform(3, 18, n).round(1),
        "PIM_Drawn": np.random.randint(5, 60, n),
        "Pct_Shift_Starts_Offensive_Zone": np.random.uniform(8, 25, n).round(1),
        "On_Ice_Corsi_Pct": np.random.uniform(40, 60, n).round(1),
    })

    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_data_path, index=False)
    print(f"[load_data] Saved dummy CSV to {raw_data_path}")
    return df
