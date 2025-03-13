import pandas as pd

def load_data(file_path):
    """Loads dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Basic preprocessing: handling missing values."""
    return df.fillna(df.mean())
