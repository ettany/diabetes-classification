import os
import pandas as pd
import numpy as np
import sys
import logging

# Add parent directory to path to import package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom functions from the package
from diabetes_classifier.data_processing import load_data, preprocess_data
from diabetes_classifier.math_utils import (
    mean, standard_deviation, min_max_scaling, z_score_normalization
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
RAW_DATA_PATH = "data/raw/diabetes.csv"
PROCESSED_DATA_PATH = "data/processed/diabetes_cleaned.csv"
STATS_PATH = "data/processed/feature_statistics.csv"

def main():
    """
    Main preprocessing script for the diabetes dataset.
    
    This script:
    1. Loads the raw diabetes data
    2. Applies preprocessing using custom math functions
    3. Saves the processed data
    4. Generates and saves feature statistics
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    # Load raw data
    logger.info(f"Loading data from {RAW_DATA_PATH}")
    try:
        df = load_data(RAW_DATA_PATH)
        logger.info(f"Loaded dataset with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Generate and save raw data statistics
    logger.info("Generating raw data statistics")
    feature_stats = calculate_feature_statistics(df)
    
    # Apply preprocessing
    logger.info("Applying preprocessing steps")
    try:
        df_processed = preprocess_data(df)
        logger.info(f"Processed dataset has shape: {df_processed.shape}")
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return
    
    # Generate processed data statistics
    logger.info("Generating processed data statistics")
    processed_stats = calculate_feature_statistics(df_processed)
    
    # Combine raw and processed statistics
    combined_stats = pd.concat([
        feature_stats.add_suffix('_raw'), 
        processed_stats.add_suffix('_processed')
    ], axis=1)
    
    # Save processed data
    logger.info(f"Saving processed data to {PROCESSED_DATA_PATH}")
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    
    # Save statistics
    logger.info(f"Saving feature statistics to {STATS_PATH}")
    combined_stats.to_csv(STATS_PATH)
    
    logger.info("Preprocessing complete!")
    print(f"Preprocessing complete! Processed data saved to {PROCESSED_DATA_PATH}")
    print(f"Feature statistics saved to {STATS_PATH}")

def calculate_feature_statistics(df):
    """
    Calculate statistics for each feature using custom math functions.
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    pandas DataFrame with feature statistics
    """
    # Initialize dictionary to store statistics
    stats = {}
    
    # Calculate statistics for each feature
    for column in df.columns:
        if column == 'Outcome':
            # Skip target variable
            continue
        
        # Get non-null values as list
        values = [val for val in df[column].values if not pd.isna(val)]
        
        # Calculate statistics using custom functions
        stats[column] = {
            'mean': mean(values),
            'std': standard_deviation(values),
            'min': min(values),
            'max': max(values),
            'missing': df[column].isna().sum()
        }
    
    # Convert to DataFrame
    return pd.DataFrame(stats).T

if __name__ == "__main__":
    main()