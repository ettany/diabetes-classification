import pandas as pd
import numpy as np
from .math_utils import (
    mean, variance, standard_deviation, euclidean_distance, 
    manhattan_distance, z_score_normalization, min_max_scaling,
    entropy, gini_index
)

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
    4. Feature selection based on information metrics
    
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
    
    # Apply different normalization techniques based on feature characteristics
    # Use Z-score normalization for normally distributed features
    z_score_columns = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction']
    for column in z_score_columns:
        values = processed_df[column].values.tolist()
        processed_df[column] = z_score_normalization(values)
    
    # Use min-max scaling for skewed distributions
    minmax_columns = ['Age', 'Pregnancies', 'SkinThickness', 'Insulin']
    for column in minmax_columns:
        values = processed_df[column].values.tolist()
        processed_df[column] = min_max_scaling(values)
    
    # Feature engineering: Add BMI categories
    processed_df['BMI_Category'] = processed_df['BMI'].apply(
        lambda x: 0 if x < -1 else (1 if x < 0 else (2 if x < 1 else 3))
    )
    
    # Feature engineering: Age groups
    processed_df['Age_Group'] = processed_df['Age'].apply(
        lambda x: 0 if x < -1 else (1 if x < 0 else (2 if x < 1 else 3))
    )
    
    # Feature engineering: Glucose-to-Insulin ratio (both are normalized)
    # This can help identify insulin resistance patterns
    processed_df['Glucose_Insulin_Ratio'] = processed_df['Glucose'] / (processed_df['Insulin'] + 1e-8)
    
    # Feature engineering: Add interaction term between BMI and Age
    # Higher BMI at older age is a stronger predictor
    processed_df['BMI_Age_Interaction'] = processed_df['BMI'] * processed_df['Age']
    
    return processed_df

def train_test_split(df, test_size=0.2, random_seed=42, stratify=True):
    """
    Split data into training and testing sets, with optional stratification.
    
    Parameters:
    df: DataFrame to split
    test_size: Proportion of data to use for testing
    random_seed: Random seed for reproducibility 
    stratify: Whether to maintain the same class distribution in both sets
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Split features and target
    X = df.drop(columns=['Outcome'])
    y = df['Outcome'].values
    
    if stratify:
        # Ensure balanced class distribution
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
        
        # Shuffle the indices
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        
        # Calculate split points
        pos_split = int(len(positive_indices) * (1 - test_size))
        neg_split = int(len(negative_indices) * (1 - test_size))
        
        # Create train and test indices
        train_indices = np.concatenate([positive_indices[:pos_split], negative_indices[:neg_split]])
        test_indices = np.concatenate([positive_indices[pos_split:], negative_indices[neg_split:]])
        
        # Shuffle the indices again
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # Split data
        X_train = X.iloc[train_indices].values
        X_test = X.iloc[test_indices].values
        y_train = y[train_indices]
        y_test = y[test_indices]
    else:
        # Shuffle the DataFrame
        df_shuffled = df.sample(frac=1, random_state=random_seed)
        
        # Determine split point
        split_idx = int(len(df) * (1 - test_size))
        
        # Split data
        X_train = X.iloc[:split_idx].values
        X_test = X.iloc[split_idx:].values
        y_train = y[:split_idx]
        y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def select_features_by_distance(df, target='Outcome', n_features=5):
    """
    Select features based on their Euclidean distance separation power.
    
    This function evaluates how well each feature separates the classes
    by calculating the distance between class means.
    
    Parameters:
    df: DataFrame with features and target
    target: Target column name
    n_features: Number of top features to select
    
    Returns:
    List of selected feature names
    """
    feature_scores = {}
    for column in df.columns:
        if column == target:
            continue
        
        # Get values for each class
        class0_values = df[df[target] == 0][column].values
        class1_values = df[df[target] == 1][column].values
        
        # Calculate class means
        mean0 = mean(class0_values)
        mean1 = mean(class1_values)
        
        # Calculate distance between means
        distance = abs(mean0 - mean1)
        
        # Normalize by pooled standard deviation for effect size
        std0 = standard_deviation(class0_values)
        std1 = standard_deviation(class1_values)
        pooled_std = np.sqrt((std0**2 + std1**2) / 2)
        
        # Store effect size (standardized distance)
        feature_scores[column] = distance / pooled_std if pooled_std > 0 else 0
    
    # Sort features by score and select top n
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_features[:n_features]]

def calculate_feature_entropy(df, target='Outcome'):
    """
    Calculate entropy for each feature with respect to the target.
    Lower entropy indicates better predictive power.
    
    Parameters:
    df: DataFrame with features and target
    target: Target column name
    
    Returns:
    Dictionary of feature entropy values
    """
    feature_entropy = {}
    for column in df.columns:
        if column == target:
            continue
            
        # Discretize continuous features for entropy calculation
        if df[column].nunique() > 10:
            # Create 5 bins for continuous features
            df[f'{column}_binned'] = pd.qcut(df[column], 5, labels=False, duplicates='drop')
            column = f'{column}_binned'
            
        # Calculate entropy for each feature value
        value_entropies = []
        for value in df[column].unique():
            # Get target distribution for this feature value
            target_counts = df[df[column] == value][target].value_counts(normalize=True)
            
            # Convert to list of probabilities
            probs = [target_counts.get(i, 0) for i in range(2)]  # Binary classification
            
            # Calculate entropy for this feature value
            if len(probs) > 1:  # Ensure we have both classes
                value_entropy = entropy(probs)
                # Weight by proportion of samples with this feature value
                weight = len(df[df[column] == value]) / len(df)
                value_entropies.append(weight * value_entropy)
        
        # Overall weighted entropy for the feature
        feature_entropy[column.replace('_binned', '')] = sum(value_entropies)
        
        # Remove temporary binned column if created
        if '_binned' in column:
            df.drop(column, axis=1, inplace=True)
    
    return feature_entropy