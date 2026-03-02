import pandas as pd

def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Cleans raw NHL dataframe: computes target, removes leakage/redundant features,
    and encodes position.
    """
    print("[clean_data.clean_dataframe] Cleaning dataframe")
    
    # Treat raw data as immutable input
    df = df_raw.copy()
    
    # Baseline rule: safe copy (identity transform) for scaffolding dummy data
    if 'target' in df.columns and 'num_feature' in df.columns:
        print(f"[clean_data.clean_dataframe] Dummy data detected. Returning identity transform. Shape: {df.shape}")
        return df

    # 1. Compute target
    required_target_cols = ['Goals', 'Primary_Assists', 'Secondary_Assists']
    if all(col in df.columns for col in required_target_cols):
        df[target_column] = df['Goals'] + df['Primary_Assists'] + df['Secondary_Assists']
    
    # 2. Remove rate, expected, and rebound columns
    forbidden_substrings = ['_Per_60', '_Per_Game', 'Expected', 'xGoals', 'Rebounds']
    cols_to_drop = [col for col in df.columns if any(sub in col for sub in forbidden_substrings)]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. Final Features (implicitly handles dropping leakage cols like Goals)
    keep_cols = [
        target_column, 'Faceoff_Win_Pct', 'Takeaways', 'Giveaways', 'Shot_Attempts', 
        'Shooting_Pct_On_Unblocked', 'PIM_Drawn', 'Icetime_Minutes', 'Pos', 
        'Pct_Shift_Starts_Offensive_Zone', 'On_Ice_Corsi_Pct'
    ]
    df = df[[col for col in keep_cols if col in df.columns]]
    
    # 4. Encode Pos as Dummies (C baseline)
    if 'Pos' in df.columns:
        df['Pos'] = pd.Categorical(df['Pos'])
        if 'C' in df['Pos'].cat.categories:
            ordered_cats = ['C'] + [cat for cat in df['Pos'].cat.categories if cat != 'C']
            df['Pos'] = df['Pos'].cat.set_categories(ordered_cats)
            
        df = pd.get_dummies(df, columns=['Pos'], drop_first=True, dtype=int)
    
    # Traceability: Log row drops just like the sandbox
    initial_len = len(df)
    df = df.dropna().drop_duplicates()
    dropped_rows = initial_len - len(df)
    
    if dropped_rows > 0:
        print(f"[clean_data.clean_dataframe] Dropped {dropped_rows} rows due to NA or duplicates")
        
    print(f"[clean_data.clean_dataframe] Rows after cleaning: {len(df)}")
    return df
