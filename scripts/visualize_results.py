"""
Results Visualization Script

This script generates advanced visualizations for diabetes classification results
using the custom mathematical functions from math_utils.py.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom functions from the package
from diabetes_classifier import load_data, preprocess_data
from diabetes_classifier.data_processing import train_test_split
from diabetes_classifier.math_utils import (
    mean, standard_deviation, cosine_similarity, 
    euclidean_distance, softmax
)

# Define paths
DATA_PATH = "data/raw/diabetes.csv"
RESULTS_PATH = "results"
ANALYSIS_PATH = os.path.join(RESULTS_PATH, "analysis")
PLOTS_PATH = os.path.join(RESULTS_PATH, "visualizations")

# Ensure directories exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(ANALYSIS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

def load_analysis_results():
    """Load analysis results from JSON file."""
    results_file = os.path.join(ANALYSIS_PATH, "analysis_results.json")
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Analysis results file not found: {results_file}")
        return None

def plot_feature_importance_radar(analysis_results, save_path=None):
    """
    Create a radar plot for feature importance using information gain.
    
    Parameters:
    analysis_results: Dictionary with analysis results
    save_path: Path to save the plot
    """
    if not analysis_results or 'entropy_metrics' not in analysis_results:
        print("No entropy metrics found in analysis results")
        return
    
    entropy_metrics = analysis_results['entropy_metrics']
    
    # Get feature names and information gain values
    features = list(entropy_metrics.keys())
    gains = [entropy_metrics[f]['information_gain'] for f in features]
    
    # Sort by information gain (highest first)
    indices = np.argsort(gains)[::-1]
    features = [features[i] for i in indices]
    gains = [gains[i] for i in indices]
    
    # Limit to top 8 features for readability
    if len(features) > 8:
        features = features[:8]
        gains = gains[:8]
    
    # Number of features
    N = len(features)
    
    # Calculate angles for radar plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the feature gains (normalized to 0-1)
    max_gain = max(gains)
    normalized_gains = [g / max_gain for g in gains]
    normalized_gains += normalized_gains[:1]  # Close the loop
    
    # Create radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Plot feature importance line
    ax.plot(angles, normalized_gains, 'o-', linewidth=2, label='Information Gain')
    ax.fill(angles, normalized_gains, alpha=0.25)
    
    # Add feature names and customize plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_rlabel_position(0)
    
    plt.title('Feature Importance (Information Gain)', size=14)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_correlation_network(analysis_results, save_path=None):
    """
    Create a network visualization of feature correlations.
    
    Parameters:
    analysis_results: Dictionary with analysis results
    save_path: Path to save the plot
    """
    if not analysis_results or 'feature_correlations' not in analysis_results:
        print("No correlation data found in analysis results")
        return
    
    correlations = analysis_results['feature_correlations']
    
    # Get feature names
    features = list(correlations.keys())
    
    # Create correlation matrix
    n_features = len(features)
    corr_matrix = np.zeros((n_features, n_features))
    
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            corr_matrix[i, j] = correlations[f1][f2]
    
    # Plot size
    plt.figure(figsize=(12, 10))
    
    # Node positions in a circle
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
    pos = {}
    for i, feature in enumerate(features):
        pos[i] = [np.cos(angles[i]), np.sin(angles[i])]
    
    # Plot nodes
    for i, feature in enumerate(features):
        plt.plot(pos[i][0], pos[i][1], 'o', markersize=10, 
                 color='skyblue', alpha=0.8)
        plt.text(1.1*pos[i][0], 1.1*pos[i][1], feature, 
                 fontsize=12, ha='center', va='center')
    
    # Plot edges (correlations)
    for i in range(n_features):
        for j in range(i+1, n_features):
            correlation = corr_matrix[i, j]
            
            # Skip weak correlations
            if abs(correlation) < 0.3:
                continue
            
            # Line width based on correlation strength
            line_width = abs(correlation) * 3
            
            # Line color based on positive/negative correlation
            color = 'green' if correlation > 0 else 'red'
            
            # Line alpha based on correlation strength
            alpha = abs(correlation)
            
            # Plot line
            plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 
                     '-', linewidth=line_width, color=color, alpha=alpha)
    
    # Remove axis
    plt.axis('off')
    plt.title('Feature Correlation Network', fontsize=16)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_class_separation_heatmap(analysis_results, save_path=None):
    """
    Create a heatmap showing how well each feature separates the classes.
    
    Parameters:
    analysis_results: Dictionary with analysis results
    save_path: Path to save the plot
    """
    if not analysis_results or 'class_separation' not in analysis_results:
        print("No class separation data found in analysis results")
        return
    
    separation = analysis_results['class_separation']
    
    # Get features and their effect sizes
    features = []
    effect_sizes = []
    
    for feature, metrics in separation.items():
        if '0_vs_1' in metrics:
            features.append(feature)
            effect_sizes.append(metrics['0_vs_1']['effect_size'])
    
    # Sort by effect size (highest first)
    indices = np.argsort(effect_sizes)[::-1]
    features = [features[i] for i in indices]
    effect_sizes = [effect_sizes[i] for i in indices]
    
    # Create heatmap data
    data = np.zeros((len(features), 1))
    for i, effect_size in enumerate(effect_sizes):
        data[i, 0] = effect_size
    
    # Create custom colormap
    colors = [(0.1, 0.1, 0.7), (0.9, 0.9, 0.9), (0.7, 0.1, 0.1)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)
    
    # Create plot
    plt.figure(figsize=(8, 12))
    
    # Plot heatmap
    plt.imshow(data, cmap=cmap, aspect='auto')
    
    # Add feature names
    plt.yticks(range(len(features)), features)
    plt.xticks([0], ['Effect Size'])
    
    # Add colorbar
    cbar = plt.colorbar(orientation='horizontal', pad=0.05)
    cbar.set_label('Effect Size (Standardized Mean Difference)')
    
    # Add values to heatmap
    for i, effect_size in enumerate(effect_sizes):
        plt.text(0, i, f'{effect_size:.2f}', 
                ha='center', va='center', 
                color='white' if effect_size > 2 else 'black')
    
    plt.title('Feature Class Separation Power', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_distributions_density(df, save_path=None):
    """
    Plot kernel density estimates for feature distributions by class.
    
    Parameters:
    df: DataFrame with features and target
    save_path: Path to save the plot
    """
    # Get features (excluding Outcome)
    features = [col for col in df.columns if col != 'Outcome']
    
    # Number of rows and columns for subplots
    n_cols = 2
    n_rows = (len(features) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Get values for each class
        values_0 = df[df['Outcome'] == 0][feature].values
        values_1 = df[df['Outcome'] == 1][feature].values
        
        # Calculate custom statistics
        mean_0 = mean(values_0)
        std_0 = standard_deviation(values_0)
        mean_1 = mean(values_1)
        std_1 = standard_deviation(values_1)
        
        # Generate KDE curves manually
        x = np.linspace(min(min(values_0), min(values_1)), 
                        max(max(values_0), max(values_1)), 1000)
        
        # Calculate KDE values using Gaussian approximation
        y_0 = np.zeros_like(x)
        y_1 = np.zeros_like(x)
        
        for val in values_0:
            # Add Gaussian contribution from each data point
            y_0 += np.exp(-((x - val) ** 2) / (2 * std_0 ** 2)) / (std_0 * np.sqrt(2 * np.pi))
        
        for val in values_1:
            # Add Gaussian contribution from each data point
            y_1 += np.exp(-((x - val) ** 2) / (2 * std_1 ** 2)) / (std_1 * np.sqrt(2 * np.pi))
        
        # Normalize
        y_0 /= len(values_0)
        y_1 /= len(values_1)
        
        # Plot density curves
        plt.plot(x, y_0, 'b', linewidth=2, label='Non-Diabetic (0)')
        plt.plot(x, y_1, 'r', linewidth=2, label='Diabetic (1)')
        
        # Add mean lines
        plt.axvline(mean_0, color='blue', linestyle='--', alpha=0.7, label=f'Mean (0): {mean_0:.2f}')
        plt.axvline(mean_1, color='red', linestyle='--', alpha=0.7, label=f'Mean (1): {mean_1:.2f}')
        
        # Calculate effect size and add to title
        pooled_std = np.sqrt((std_0 ** 2 + std_1 ** 2) / 2)
        effect_size = abs(mean_0 - mean_1) / pooled_std if pooled_std > 0 else 0
        
        plt.title(f'{feature}\nEffect Size: {effect_size:.2f}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend(loc='best')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_feature_profiles(df, analysis_results, save_path=None):
    """
    Create comprehensive feature profile visualizations.
    
    Parameters:
    df: DataFrame with features and target
    analysis_results: Dictionary with analysis results
    save_path: Path to save the plot
    """
    if not analysis_results:
        print("No analysis results provided")
        return
    
    # Get top 4 features by information gain
    if 'entropy_metrics' in analysis_results:
        entropy_metrics = analysis_results['entropy_metrics']
        feature_gains = [(f, m['information_gain']) for f, m in entropy_metrics.items()]
        feature_gains.sort(key=lambda x: x[1], reverse=True)
        top_features = [f for f, _ in feature_gains[:4]]
    else:
        # Fallback to all features
        top_features = [col for col in df.columns if col != 'Outcome'][:4]
    
    # Create a 2x2 grid for top 4 features
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    for i, feature in enumerate(top_features):
        row, col = divmod(i, 2)
        
        # Create subplot in the grid
        ax = plt.subplot(gs[row, col])
        
        # Get feature statistics
        values = df[feature].values
        values_0 = df[df['Outcome'] == 0][feature].values
        values_1 = df[df['Outcome'] == 1][feature].values
        
        # Calculate custom statistics
        feature_mean = mean(values)
        feature_std = standard_deviation(values)
        mean_0 = mean(values_0)
        std_0 = standard_deviation(values_0)
        mean_1 = mean(values_1)
        std_1 = standard_deviation(values_1)
        
        # Calculate effect size
        pooled_std = np.sqrt((std_0 ** 2 + std_1 ** 2) / 2)
        effect_size = abs(mean_0 - mean_1) / pooled_std if pooled_std > 0 else 0
        
        # Get information gain
        info_gain = entropy_metrics.get(feature, {}).get('information_gain', 0)
        
        # Main plot area
        plt.hist(values_0, bins=15, alpha=0.6, color='blue', label='Non-Diabetic')
        plt.hist(values_1, bins=15, alpha=0.6, color='red', label='Diabetic')
        
        # Add vertical lines for means
        plt.axvline(mean_0, color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean (0): {mean_0:.2f}')
        plt.axvline(mean_1, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean (1): {mean_1:.2f}')
        
        # Add panel title
        plt.title(f'{feature} Distribution by Class', fontsize=14)
        
        # Add legend
        plt.legend(loc='upper right')
        
        # Add annotation with statistics
        stats_text = (
            f"Feature Statistics:\n"
            f"Overall Mean: {feature_mean:.2f}\n"
            f"Overall Std: {feature_std:.2f}\n\n"
            f"Non-Diabetic Mean: {mean_0:.2f}\n"
            f"Non-Diabetic Std: {std_0:.2f}\n\n"
            f"Diabetic Mean: {mean_1:.2f}\n"
            f"Diabetic Std: {std_1:.2f}\n\n"
            f"Effect Size: {effect_size:.2f}\n"
            f"Information Gain: {info_gain:.4f}"
        )
        
        # Position text in the upper left
        plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Feature Profiles for Diabetes Classification', fontsize=20)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_pair_analysis(df, save_path=None):
    """
    Create pairwise feature analysis for the top features.
    
    Parameters:
    df: DataFrame with features and target
    save_path: Path to save the plot
    """
    # Get top 4 features for visualization
    features = ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']
    
    # Create a 2x2 grid of scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    # Feature pairs to plot
    pairs = [
        ('Glucose', 'BMI'),
        ('Glucose', 'Age'),
        ('BMI', 'Age'),
        ('DiabetesPedigreeFunction', 'Glucose')
    ]
    
    for i, (feature1, feature2) in enumerate(pairs):
        # Get data for this pair
        x = df[feature1].values
        y = df[feature2].values
        colors = ['blue' if outcome == 0 else 'red' for outcome in df['Outcome']]
        
        # Create scatter plot
        axes[i].scatter(x, y, c=colors, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # Calculate class centroids
        x0 = df[df['Outcome'] == 0][feature1].values
        y0 = df[df['Outcome'] == 0][feature2].values
        x1 = df[df['Outcome'] == 1][feature1].values
        y1 = df[df['Outcome'] == 1][feature2].values
        
        x0_mean = mean(x0)
        y0_mean = mean(y0)
        x1_mean = mean(x1)
        y1_mean = mean(y1)
        
        # Plot centroids
        axes[i].scatter(x0_mean, y0_mean, c='blue', marker='X', s=200, 
                       edgecolors='black', linewidth=2, label='Non-Diabetic Centroid')
        axes[i].scatter(x1_mean, y1_mean, c='red', marker='X', s=200, 
                       edgecolors='black', linewidth=2, label='Diabetic Centroid')
        
        # Calculate distance between centroids using euclidean_distance
        centroid_distance = euclidean_distance([x0_mean, y0_mean], [x1_mean, y1_mean])
        
        # Add labels and title
        axes[i].set_xlabel(feature1, fontsize=12)
        axes[i].set_ylabel(feature2, fontsize=12)
        axes[i].set_title(f'{feature1} vs {feature2}\nCentroid Distance: {centroid_distance:.2f}', 
                         fontsize=14)
        
        # Add legend (only for the first plot)
        if i == 0:
            axes[i].legend(loc='upper right')
    
    plt.tight_layout()
    plt.suptitle('Pairwise Feature Analysis for Diabetes Classification', fontsize=16, y=0.995)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Load diabetes dataset
    print(f"Loading dataset from {DATA_PATH}")
    df = load_data(DATA_PATH)
    
    # Preprocess data
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Load analysis results
    print("Loading analysis results...")
    analysis_results = load_analysis_results()
    
    if not analysis_results:
        print("Analysis results not found. Run dataset_analyzer.py first.")
        return
    
    # Generate visualizations
    print("\nGenerating advanced visualizations...")
    
    # Feature importance radar plot
    print("Creating feature importance radar plot...")
    radar_path = os.path.join(PLOTS_PATH, "feature_importance_radar.png")
    plot_feature_importance_radar(analysis_results, save_path=radar_path)
    
    # Feature correlation network
    print("Creating feature correlation network...")
    network_path = os.path.join(PLOTS_PATH, "feature_correlation_network.png")
    plot_feature_correlation_network(analysis_results, save_path=network_path)
    
    # Class separation heatmap
    print("Creating class separation heatmap...")
    heatmap_path = os.path.join(PLOTS_PATH, "class_separation_heatmap.png")
    plot_class_separation_heatmap(analysis_results, save_path=heatmap_path)
    
    # Feature distributions density plot
    print("Creating feature density plots...")
    density_path = os.path.join(PLOTS_PATH, "feature_densities.png")
    plot_feature_distributions_density(df_processed, save_path=density_path)
    
    # Feature profiles
    print("Creating feature profiles...")
    profiles_path = os.path.join(PLOTS_PATH, "feature_profiles.png")
    create_feature_profiles(df_processed, analysis_results, save_path=profiles_path)
    
    # Feature pair analysis
    print("Creating pairwise feature analysis...")
    pairs_path = os.path.join(PLOTS_PATH, "feature_pairs.png")
    plot_feature_pair_analysis(df_processed, save_path=pairs_path)
    
    print(f"\nAll visualizations saved to {PLOTS_PATH}")
    print("The following visualizations were created:")
    print("1. Feature Importance Radar Chart")
    print("2. Feature Correlation Network")
    print("3. Class Separation Heatmap")
    print("4. Feature Density Plots")
    print("5. Feature Profiles")
    print("6. Pairwise Feature Analysis")

if __name__ == "__main__":
    main()