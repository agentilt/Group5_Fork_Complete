import pandas as pd

def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: list, 
    check_missing_values: bool = False
) -> bool:
    """
    Validates the dataframe and fails fast for obvious issues.
    """
    print("[validate.validate_dataframe] Validating dataframe")
    
    if df.empty:
        raise ValueError("Validation Failed: The DataFrame is empty.")
        
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Validation Failed: The following required columns are missing: {missing_cols}")
        
    if check_missing_values and df.isnull().sum().sum() > 0:
        raise ValueError("Validation Failed: DataFrame contains missing values (NaNs).")
        
    return True
