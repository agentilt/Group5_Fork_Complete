import pandas as pd
from pathlib import Path
from src.utils import load_csv, save_csv

def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Attempts to load the raw data CSV. If missing, generates a 
    deterministic dummy dataset for scaffolding.
    """
    print(f"[load_data.load_raw_data] Loading raw data from {raw_data_path}")
    
    try:
        df = load_csv(raw_data_path)
        print(f"[load_data.load_raw_data] Loaded dataframe shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Raw data not found at {raw_data_path}\n"
            "Check that you opened the correct repository folder and that data/raw exists"
    )

df_raw = load_raw_data()
print("df_raw.shape:", df_raw.shape)
df_raw.head()
