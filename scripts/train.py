import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the package
from diabetes_classifier import DiabetesClassifier, load_data, preprocess_data
from diabetes_classifier.data_processing import (
    train_test_split, select_features_by_distance, calculate_feature_entropy
)
from diabetes_classifier.evaluation import (
    evaluate_model, calculate_confusion_matrix, 
    distance_based_evaluation, calibration_analysis
)
from diabetes_classifier.math_utils import (
    mean, variance, standard_deviation, softmax,
    cosine_similarity
)

# Define paths
DATA_PATH = "data/raw/diabetes.csv"
RESULTS_PATH = "results"
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")

# Ensure directories exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance based on model weights."""
    importance = np.abs(model.weights)
    # Normalize to sum to 1
    importance = importance / np.sum(importance)
    
    # Sort features by importance
    indices = np.argsort(importance)
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_names)), sorted_importance, align='center')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_loss(losses, save_path=None):
    """Plot training loss over iterations."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_calibration_curve(calib_data, save_path=None):
    """Plot model calibration curve."""
    bin_metrics = calib_data['bin_metrics']
    
    # Extract data from bin metrics
    mean_predicted_probs = [m['mean_predicted_prob'] for m in bin_metrics]
    observed_freqs = [m['observed_frequency'] for m in bin_metrics]
    sample_counts = [m['sample_count'] for m in bin_metrics]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Actual calibration
    plt.scatter(mean_predicted_probs, observed_freqs, 
                s=[count/10 for count in sample_counts], alpha=0.7)
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Load and preprocess data
    print("Loading data from", DATA_PATH)
    df = load_data(DATA_PATH)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Calculate feature entropy for feature selection
    print("Calculating feature importance metrics...")
    feature_entropy = calculate_feature_entropy(df_processed)
    
    # Sort features by entropy (lower is better)
    sorted_entropy = sorted(feature_entropy.items(), key=lambda x: x[1])
    print("\nFeature Entropy (lower is better):")
    for feature, score in sorted_entropy:
        print(f"  {feature}: {score:.4f}")
    
    # Select features by distance-based metrics
    selected_features = select_features_by_distance(df_processed, n_features=8)
    print("\nSelected features:", selected_features)
    
    # Keep only selected features plus target
    df_selected = df_processed[selected_features + ['Outcome']]
    
    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_selected, test_size=0.2, stratify=True
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Calculate class distribution
    train_pos = sum(y_train == 1)
    train_neg = sum(y_train == 0)
    test_pos = sum(y_test == 1)
    test_neg = sum(y_test == 0)
    
    print(f"Training set class distribution: {train_pos} positive, {train_neg} negative")
    print(f"Testing set class distribution: {test_pos} positive, {test_neg} negative")
    
    # Calculate feature statistics
    print("\nFeature Statistics:")
    for i, feature in enumerate(selected_features):
        train_values = X_train[:, i]
        # Using our custom functions for statistics
        feature_mean = mean(train_values)
        feature_std = standard_deviation(train_values)
        feature_var = variance(train_values)
        
        print(f"  {feature}:")
        print(f"    Mean: {feature_mean:.4f}")
        print(f"    Std Dev: {feature_std:.4f}")
        print(f"    Variance: {feature_var:.4f}")
    
    # Train the model with regularization
    print("\nTraining custom diabetes classifier...")
    model = DiabetesClassifier(learning_rate=0.01, iterations=5000, regularization=0.01)
    model.train(X_train, y_train)
    
    # Plot training loss
    loss_plot_path = os.path.join(PLOTS_PATH, "training_loss.png")
    plot_training_loss(model.training_losses, save_path=loss_plot_path)
    print(f"Training loss curve saved to {loss_plot_path}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate the model using multiple metrics
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Calculate confusion matrix
    conf_matrix = calculate_confusion_matrix(y_test, y_pred)
    
    # Calculate distance-based metrics
    dist_metrics = distance_based_evaluation(model, X_test, y_test)
    
    # Analyze calibration
    calib_data = calibration_analysis(y_test, y_pred_proba)
    
    # Plot calibration curve
    calib_plot_path = os.path.join(PLOTS_PATH, "calibration_curve.png")
    plot_calibration_curve(calib_data, save_path=calib_plot_path)
    print(f"Calibration curve saved to {calib_plot_path}")
    
    # Plot feature importance
    importance_path = os.path.join(PLOTS_PATH, "feature_importance.png")
    plot_feature_importance(model, selected_features, save_path=importance_path)
    print(f"Feature importance plot saved to {importance_path}")
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Cross-Entropy Loss: {metrics['cross_entropy_loss']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(f"True Positives: {conf_matrix['true_positives']}")
    print(f"False Positives: {conf_matrix['false_positives']}")
    print(f"False Negatives: {conf_matrix['false_negatives']}")
    print(f"True Negatives: {conf_matrix['true_negatives']}")
    
    # Print distance-based metrics
    print("\nDistance-Based Metrics:")
    for key, value in dist_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Print calibration metrics
    print("\nCalibration Metrics:")
    print(f"Average Calibration Error: {calib_data['avg_calibration_error']:.4f}")
    print(f"Maximum Calibration Error: {calib_data['max_calibration_error']:.4f}")
    
    # Find similar samples using cosine similarity
    print("\nFinding similar samples using cosine similarity...")
    # Select a random positive sample
    pos_indices = [i for i, label in enumerate(y_test) if label == 1]
    if pos_indices:
        sample_idx = np.random.choice(pos_indices)
        sample = X_test[sample_idx]
        
        # Calculate similarities
        similarities = [model.similarity_score(sample, x) for x in X_test]
        
        # Get top 5 most similar samples (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:6]
        
        print(f"Sample {sample_idx} (Diabetic):")
        print("Top 5 most similar samples:")
        for i, idx in enumerate(similar_indices):
            print(f"  {i+1}. Sample {idx} (True label: {y_test[idx]}, " 
                  f"Predicted: {y_pred[idx]}, Similarity: {similarities[idx]:.4f})")
    
    # Combine all metrics
    all_metrics = {
        **metrics,
        'confusion_matrix': conf_matrix,
        'distance_metrics': dist_metrics,
        'calibration': {
            'avg_error': calib_data['avg_calibration_error'],
            'max_error': calib_data['max_calibration_error']
        },
        'feature_selection': {
            'selected_features': selected_features,
            'feature_entropy': {k: float(v) for k, v in feature_entropy.items()}
        }
    }
    
    # Save results
    results_file = os.path.join(RESULTS_PATH, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f"\nResults saved to {results_file}")
    print(f"Plots saved to {PLOTS_PATH}")

if __name__ == "__main__":
    main()