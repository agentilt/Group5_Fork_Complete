from pathlib import Path
import pandas as pd
import joblib

def load_csv(filepath: Path) -> pd.DataFrame:
    print(f"[utils] Loading CSV from: {filepath}")  # replace with logging later
    if not filepath.exists():
        print(f"[utils] File not found. Creating dummy CSV at {filepath}")
        dummy_df = pd.DataFrame({
            "num_feature": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat_feature": ["A", "B", "A", "B", "C"],
            "target": [10, 20, 15, 25, 30]
        })
        filepath.parent.mkdir(parents=True, exist_ok=True)
        dummy_df.to_csv(filepath, index=False)
        print("[utils] Dummy CSV created. Replace with real NHL dataset.")
    df = pd.read_csv(filepath)
    return df

def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    print(f"[utils] Saving DataFrame to CSV at: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

def save_model(model, filepath: Path) -> None:
    print(f"[utils] Saving model to: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath: Path):
    print(f"[utils] Loading model from: {filepath}")  # TODO: replace with logging later
    model = joblib.load(filepath)
    return model
