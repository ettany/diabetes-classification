"""
Diabetes Dataset Analyzer Script

This script analyzes the diabetes dataset using custom mathematical utilities
from math_utils.py, providing insights on feature distributions, correlations,
and class separability.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom functions from the package
from diabetes_classifier import load_data
from diabetes_classifier.math_utils import (
    mean, variance, standard_deviation, euclidean_distance,
    manhattan_distance, cosine_similarity, entropy
)

# Define paths
DATA_PATH = "data/raw/diabetes.csv"
RESULTS_PATH = "results/analysis"
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")

# Ensure directories exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

def analyze_feature_statistics(df):
    """
    Calculate statistics for each feature using custom math functions.
    
    Parameters:
    df: DataFrame with features
    
    Returns:
    Dictionary with feature statistics
    """
    stats = {}
    
    for column in df.columns:
        if column == 'Outcome':
            continue
            
        # Get non-null values
        values = df[column].dropna().values.tolist()
        
        # Calculate statistics using custom functions
        feature_mean = mean(values)
        feature_variance = variance(values)
        feature_std = standard_deviation(values)
        
        # Calculate min, max, median
        feature_min = min(values)
        feature_max = max(values)
        feature_median = sorted(values)[len(values) // 2]
        
        # Calculate quartiles
        sorted_values = sorted(values)
        q1_idx = len(values) // 4
        q3_idx = 3 * len(values) // 4
        feature_q1 = sorted_values[q1_idx]
        feature_q3 = sorted_values[q3_idx]
        
        # Store statistics
        stats[column] = {
            'mean': feature_mean,
            'variance': feature_variance,
            'std_dev': feature_std,
            'min': feature_min,
            'max': feature_max,
            'median': feature_median,
            'q1': feature_q1,
            'q3': feature_q3,
            'iqr': feature_q3 - feature_q1,
            'count': len(values),
            'missing': len(df) - len(values)
        }
    
    return stats

def calculate_feature_correlations(df):
    """
    Calculate feature correlations using cosine similarity.
    
    Parameters:
    df: DataFrame with features
    
    Returns:
    Dictionary with feature correlations
    """
    features = df.columns.tolist()
    correlations = {}
    
    for feature1 in features:
        correlations[feature1] = {}
        
        for feature2 in features:
            if feature1 == feature2:
                correlations[feature1][feature2] = 1.0
                continue
                
            # Get values
            values1 = df[feature1].values
            values2 = df[feature2].values
            
            # Center the values around mean
            values1_centered = values1 - mean(values1)
            values2_centered = values2 - mean(values2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(values1_centered, values2_centered)
            correlations[feature1][feature2] = similarity
    
    return correlations

def analyze_class_separation(df, target='Outcome'):
    """
    Analyze how well each feature separates the classes.
    
    Parameters:
    df: DataFrame with features and target
    target: Target column name
    
    Returns:
    Dictionary with class separation metrics
    """
    separation = {}
    
    # Get classes
    class_values = df[target].unique().tolist()
    
    # Sort class values to ensure consistent keys (0_vs_1 instead of 1_vs_0)
    class_values.sort()
    
    for column in df.columns:
        if column == target:
            continue
            
        separation[column] = {}
        
        # Get values for each class
        class_data = {}
        for class_val in class_values:
            class_data[class_val] = df[df[target] == class_val][column].values.tolist()
        
        # Calculate mean and std for each class
        class_stats = {}
        for class_val, values in class_data.items():
            class_stats[class_val] = {
                'mean': mean(values),
                'std': standard_deviation(values)
            }
        
        # Calculate metrics for all class pairs
        for i, class1 in enumerate(class_values):
            for class2 in class_values[i+1:]:
                # Get class statistics
                mean1 = class_stats[class1]['mean']
                std1 = class_stats[class1]['std']
                mean2 = class_stats[class2]['mean']
                std2 = class_stats[class2]['std']
                
                # Calculate distance metrics
                abs_diff = abs(mean1 - mean2)
                
                # Effect size (standardized mean difference)
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                effect_size = abs_diff / pooled_std if pooled_std > 0 else 0
                
                # Store metrics
                key = f'{class1}_vs_{class2}'
                separation[column][key] = {
                    'abs_diff': abs_diff,
                    'effect_size': effect_size,
                    'mean1': mean1,
                    'mean2': mean2,
                    'std1': std1,
                    'std2': std2
                }
    
    return separation

def analyze_feature_entropy(df, target='Outcome'):
    """
    Calculate entropy-based metrics for each feature.
    
    Parameters:
    df: DataFrame with features and target
    target: Target column name
    
    Returns:
    Dictionary with entropy metrics
    """
    entropy_metrics = {}
    
    # Calculate overall target entropy
    target_counts = df[target].value_counts(normalize=True)
    target_probs = [target_counts.get(i, 0) for i in range(len(target_counts))]
    overall_entropy = entropy(target_probs)
    
    for column in df.columns:
        if column == target:
            continue
            
        # Bin continuous features for entropy calculation
        if df[column].nunique() > 10:
            # Create 5 equal-width bins
            df[f'{column}_binned'] = pd.cut(df[column], 5, labels=False)
            column = f'{column}_binned'
        
        # Calculate conditional entropy
        conditional_entropy = 0
        
        for value in df[column].unique():
            # Skip NaN values
            if pd.isna(value):
                continue
                
            # Get subset with this feature value
            subset = df[df[column] == value]
            
            # Calculate weight (proportion of samples with this value)
            weight = len(subset) / len(df)
            
            # Calculate target distribution in this subset
            target_counts = subset[target].value_counts(normalize=True)
            target_probs = [target_counts.get(i, 0) for i in range(len(target_counts))]
            
            # Calculate entropy for this value
            if len(target_probs) > 1:
                value_entropy = entropy(target_probs)
                conditional_entropy += weight * value_entropy
        
        # Calculate information gain
        info_gain = overall_entropy - conditional_entropy
        
        # Store metrics
        original_column = column.replace('_binned', '')
        entropy_metrics[original_column] = {
            'conditional_entropy': conditional_entropy,
            'information_gain': info_gain,
            'overall_entropy': overall_entropy
        }
        
        # Remove temporary binned column
        if column != original_column:
            df.drop(column, axis=1, inplace=True)
    
    return entropy_metrics

def plot_class_separation(df, target='Outcome', save_path=None):
    """
    Plot feature distributions by class to visualize separation.
    
    Parameters:
    df: DataFrame with features and target
    target: Target column name
    save_path: Path to save the plot
    """
    features = [col for col in df.columns if col != target]
    class_values = df[target].unique()
    
    # Calculate number of rows and columns for subplots
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot histograms for each class
        for class_val in class_values:
            values = df[df[target] == class_val][feature].values
            plt.hist(values, bins=20, alpha=0.5, label=f'Class {class_val}')
        
        # Calculate class means
        class_means = {}
        for class_val in class_values:
            values = df[df[target] == class_val][feature].values.tolist()
            class_means[class_val] = mean(values)
        
        # Add vertical lines for class means
        colors = ['blue', 'red', 'green', 'purple']
        for i, (class_val, class_mean) in enumerate(class_means.items()):
            plt.axvline(class_mean, color=colors[i % len(colors)], 
                        linestyle='--', label=f'Mean Class {class_val}')
        
        plt.title(feature)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_correlation_heatmap(correlations, save_path=None):
    """
    Plot correlation heatmap using calculated cosine similarities.
    
    Parameters:
    correlations: Dictionary with feature correlations
    save_path: Path to save the plot
    """
    features = list(correlations.keys())
    
    # Create correlation matrix
    corr_matrix = np.zeros((len(features), len(features)))
    
    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features):
            corr_matrix[i, j] = correlations[feature1][feature2]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    
    # Add labels
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.yticks(range(len(features)), features)
    
    # Add values
    for i in range(len(features)):
        for j in range(len(features)):
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                    ha='center', va='center', color=color)
    
    plt.title('Feature Correlation Matrix (Cosine Similarity)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_boxplots(df, target='Outcome', save_path=None):
    """
    Plot boxplots for each feature grouped by class.
    
    Parameters:
    df: DataFrame with features and target
    target: Target column name
    save_path: Path to save the plot
    """
    features = [col for col in df.columns if col != target]
    
    # Calculate number of rows and columns for subplots
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Create boxplot data
        data = []
        labels = []
        
        for class_val in sorted(df[target].unique()):
            data.append(df[df[target] == class_val][feature].values)
            labels.append(f'Class {class_val}')
        
        # Plot boxplot
        plt.boxplot(data, labels=labels)
        
        plt.title(feature)
        plt.ylabel('Value')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(entropy_metrics, save_path=None):
    """
    Plot feature importance based on information gain.
    
    Parameters:
    entropy_metrics: Dictionary with entropy-based metrics
    save_path: Path to save the plot
    """
    features = list(entropy_metrics.keys())
    info_gains = [entropy_metrics[feature]['information_gain'] for feature in features]
    
    # Sort features by information gain
    indices = np.argsort(info_gains)
    sorted_features = [features[i] for i in indices]
    sorted_gains = [info_gains[i] for i in indices]
    
    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_gains, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Information Gain')
    plt.title('Feature Importance (Information Gain)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Load the dataset
    print(f"Loading dataset from {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"Dataset loaded with shape: {df.shape}")
    
    # Basic dataset info
    print("\nDataset Summary:")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")
    print(f"Target distribution: {df['Outcome'].value_counts().to_dict()}")
    
    # Analyze feature statistics
    print("\nCalculating feature statistics using custom functions...")
    stats = analyze_feature_statistics(df)
    
    # Print feature statistics
    print("\nFeature Statistics:")
    for feature, feature_stats in stats.items():
        print(f"\n{feature}:")
        print(f"  Mean: {feature_stats['mean']:.4f}")
        print(f"  Std Dev: {feature_stats['std_dev']:.4f}")
        print(f"  Min: {feature_stats['min']:.4f}")
        print(f"  Max: {feature_stats['max']:.4f}")
        print(f"  Missing: {feature_stats['missing']}")
    
    # Calculate feature correlations
    print("\nCalculating feature correlations using cosine similarity...")
    correlations = calculate_feature_correlations(df)
    
    # Plot correlation heatmap
    print("Generating correlation heatmap...")
    corr_path = os.path.join(PLOTS_PATH, "correlation_heatmap.png")
    plot_correlation_heatmap(correlations, save_path=corr_path)
    
    # Analyze class separation
    print("\nAnalyzing class separation metrics...")
    separation = analyze_class_separation(df)
    
    # Print top features by class separation
    print("\nTop Features by Class Separation (Effect Size):")
    feature_scores = {}
    
    for feature, metrics in separation.items():
        # Get effect size for diabetic vs non-diabetic
        # Make sure key exists (using .get with default 0)
        effect_size = metrics.get('0_vs_1', {}).get('effect_size', 0)
        feature_scores[feature] = effect_size
    
    # Sort features by effect size
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    for feature, score in sorted_features:
        print(f"  {feature}: {score:.4f}")
    
    # Analyze feature entropy
    print("\nCalculating entropy-based metrics...")
    entropy_metrics = analyze_feature_entropy(df)
    
    # Print features by information gain
    print("\nFeatures by Information Gain:")
    sorted_by_gain = sorted(
        [(f, m['information_gain']) for f, m in entropy_metrics.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    for feature, gain in sorted_by_gain:
        print(f"  {feature}: {gain:.4f}")
    
    # Generate plots
    print("\nGenerating analysis plots...")
    
    # Plot class distributions
    print("Plotting feature distributions by class...")
    dist_path = os.path.join(PLOTS_PATH, "class_distributions.png")
    plot_class_separation(df, save_path=dist_path)
    
    # Plot boxplots
    print("Plotting boxplots...")
    box_path = os.path.join(PLOTS_PATH, "feature_boxplots.png")
    plot_boxplots(df, save_path=box_path)
    
    # Plot feature importance
    print("Plotting feature importance...")
    imp_path = os.path.join(PLOTS_PATH, "feature_importance.png")
    plot_feature_importance(entropy_metrics, save_path=imp_path)
    
    # Save analysis results
    print("\nSaving analysis results...")
    
    results = {
        'feature_statistics': stats,
        'class_separation': separation,
        'feature_correlations': {k: {k2: float(v2) for k2, v2 in v.items()} 
                                 for k, v in correlations.items()},
        'entropy_metrics': entropy_metrics
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    results = convert_to_serializable(results)
    
    # Save to JSON
    results_file = os.path.join(RESULTS_PATH, "analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Analysis results saved to {results_file}")
    print(f"Plots saved to {PLOTS_PATH}")
    
    print("\nTop 3 most important features:")
    for feature, gain in sorted_by_gain[:3]:
        # Get statistics for this feature
        feature_stats = stats[feature]
        
        # Get separation stats safely, with error handling
        separation_stats = separation.get(feature, {}).get('0_vs_1', {})
        
        # Default values if separation stats not available
        mean1 = separation_stats.get('mean1', feature_stats['mean'])
        mean2 = separation_stats.get('mean2', feature_stats['mean'])
        effect_size = separation_stats.get('effect_size', 0)
        
        print(f"\n{feature}:")
        print(f"  Information Gain: {gain:.4f}")
        print(f"  Effect Size: {effect_size:.4f}")
        print(f"  Mean (Non-diabetic): {mean1:.4f}")
        print(f"  Mean (Diabetic): {mean2:.4f}")
        print(f"  Overall Range: {feature_stats['min']:.4f} - {feature_stats['max']:.4f}")

if __name__ == "__main__":
    main()