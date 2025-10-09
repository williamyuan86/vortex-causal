# src/utils/data_loader.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_sachs(path="data/sachs.csv"):
    """
    Expect: CSV where columns are variables and rows are samples.
    Caller must ensure the file exists.
    """
    df = pd.read_csv(path)
    # simple cleaning: drop NA rows
    df = df.dropna().reset_index(drop=True)
    # standardize numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
