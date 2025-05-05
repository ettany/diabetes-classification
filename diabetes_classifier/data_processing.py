import pandas as pd
import numpy as np
from .math_utils import mean, z_score_normalization, min_max_scaling

def load_data(file_path):
    """Loads dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the diabetes dataset using custom math utility functions.
    
    Steps:
    1. Handle missing values
    2. Feature normalization
    3. Feature engineering
    
    Parameters:
    df: pandas DataFrame with diabetes data
    
    Returns:
    Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values (0s in some columns are actually missing values)
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for column in zero_columns:
        # Replace zeros with NaN
        processed_df[column] = processed_df[column].replace(0, np.nan)
        
        # Calculate mean using our custom function
        column_values = [val for val in processed_df[column].dropna()]
        column_mean = mean(column_values)
        
        # Fill missing values with mean
        processed_df[column].fillna(column_mean, inplace=True)
    
    # Perform Z-score normalization for all features except the target
    feature_columns = [col for col in processed_df.columns if col != 'Outcome']
    
    # Apply our custom normalization functions to each column
    for column in feature_columns:
        values = processed_df[column].values.tolist()
        processed_df[column] = z_score_normalization(values)
    
    # Feature engineering: Add BMI categories
    processed_df['BMI_Category'] = processed_df['BMI'].apply(
        lambda x: 0 if x < -1 else (1 if x < 0 else (2 if x < 1 else 3))
    )
    
    # Feature engineering: Add age groups
    processed_df['Age_Group'] = processed_df['Age'].apply(
        lambda x: 0 if x < -1 else (1 if x < 0 else (2 if x < 1 else 3))
    )
    
    return processed_df

def train_test_split(df, test_size=0.2, random_seed=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    df: DataFrame to split
    test_size: Proportion of data to use for testing
    random_seed: Random seed for reproducibility
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1)
    
    # Split features and target
    X = df_shuffled.drop(columns=['Outcome'])
    y = df_shuffled['Outcome']
    
    # Determine split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    X_train = X.iloc[:split_idx].values
    X_test = X.iloc[split_idx:].values
    y_train = y.iloc[:split_idx].values
    y_test = y.iloc[split_idx:].values
    
    return X_train, X_test, y_train, y_test