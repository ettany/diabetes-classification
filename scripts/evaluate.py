import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root directory to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"Added path: {project_root}")

# Import the package
from diabetes_classifier import (
    DiabetesClassifier, load_data, preprocess_data, evaluate_model
)
from diabetes_classifier.data_processing import (
    train_test_split, select_features_by_distance, calculate_feature_entropy
)
from diabetes_classifier.evaluation import (
    calculate_confusion_matrix, distance_based_evaluation, calibration_analysis
)
from diabetes_classifier.math_utils import (
    mean, variance, standard_deviation, euclidean_distance,
    manhattan_distance, cosine_similarity, entropy, 
    sigmoid, softmax, dot_product, cross_entropy_loss
)

# Define paths
DATA_PATH = "data/raw/diabetes.csv"
MODEL_PATH = "models/diabetes_model.pkl"
RESULTS_PATH = "results"
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")

# Ensure directories exist
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve using numpy operations instead of sklearn."""
    # Manual ROC calculation
    thresholds = np.linspace(0, 1, 100)
    tpr_values = []
    fpr_values = []
    
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_scores]
        
        # True positives, false positives, true negatives, false negatives
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
        tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1-Specificity
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    # Calculate AUC using trapezoidal rule
    auc_value = 0
    for i in range(1, len(fpr_values)):
        auc_value += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_values, tpr_values, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return auc_value

def plot_confusion_matrix(conf_matrix, save_path=None):
    """Plot confusion matrix."""
    # Extract values from confusion matrix
    tn = conf_matrix['true_negatives']
    fp = conf_matrix['false_positives']
    fn = conf_matrix['false_negatives']
    tp = conf_matrix['true_positives']
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-Diabetic (0)', 'Diabetic (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_distributions(df, target='Outcome', save_path=None):
    """Plot feature distributions by class."""
    features = [col for col in df.columns if col != target]
    n_features = len(features)
    n_rows = (n_features + 1) // 2  # Ceiling division
    
    plt.figure(figsize=(15, n_rows * 4))
    
    for i, feature in enumerate(features):
        plt.subplot(n_rows, 2, i + 1)
        
        # Get feature values by class
        values_0 = df[df[target] == 0][feature].values
        values_1 = df[df[target] == 1][feature].values
        
        # Plot histograms
        plt.hist(values_0, alpha=0.5, bins=20, label='Non-Diabetic')
        plt.hist(values_1, alpha=0.5, bins=20, label='Diabetic')
        
        # Add statistics using custom functions
        mean_0 = mean(values_0.tolist())
        std_0 = standard_deviation(values_0.tolist())
        mean_1 = mean(values_1.tolist())
        std_1 = standard_deviation(values_1.tolist())
        
        plt.axvline(mean_0, color='b', linestyle='--', alpha=0.5)
        plt.axvline(mean_1, color='r', linestyle='--', alpha=0.5)
        
        # Calculate class separation using euclidean distance
        feature_separation = abs(mean_0 - mean_1) / np.sqrt((std_0**2 + std_1**2) / 2)
        
        plt.title(f'{feature}\nClass Separation: {feature_separation:.2f}')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_correlation(df, save_path=None):
    """
    Plot feature correlation matrix using custom cosine similarity.
    """
    features = df.columns.tolist()
    n_features = len(features)
    
    # Initialize correlation matrix
    corr_matrix = np.zeros((n_features, n_features))
    
    # Calculate correlations using custom cosine similarity
    for i in range(n_features):
        for j in range(n_features):
            values_i = df[features[i]].values
            values_j = df[features[j]].values
            
            # Normalize values to have zero mean
            values_i = values_i - mean(values_i.tolist())
            values_j = values_j - mean(values_j.tolist())
            
            # Calculate cosine similarity
            corr_matrix[i, j] = cosine_similarity(values_i, values_j)
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(n_features)
    plt.xticks(tick_marks, features, rotation=45, ha='right')
    plt.yticks(tick_marks, features)
    
    # Add text annotations
    for i in range(n_features):
        for j in range(n_features):
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                     horizontalalignment="center",
                     color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    
    plt.title('Feature Correlation Matrix (Cosine Similarity)')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_decision_boundary(model, X, y, feature_idx1=0, feature_idx2=1, save_path=None):
    """
    Plot decision boundary for a 2D slice of the feature space.
    
    Parameters:
    model: Trained classifier
    X: Feature matrix
    y: Target values
    feature_idx1, feature_idx2: Indices of features to use for the 2D plot
    save_path: Path to save the plot
    """
    # Extract the two selected features
    X_2d = X[:, [feature_idx1, feature_idx2]]
    
    # Create a mesh grid
    h = 0.02  # Step size
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create a simplified dataset for the mesh grid
    # Fill other features with their mean values
    feature_means = [mean(X[:, i]) for i in range(X.shape[1])]
    
    # Create mesh points with all features
    mesh_points = np.zeros((xx.ravel().shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        if i == feature_idx1:
            mesh_points[:, i] = xx.ravel()
        elif i == feature_idx2:
            mesh_points[:, i] = yy.ravel()
        else:
            mesh_points[:, i] = feature_means[i]
    
    # Get predictions for the mesh
    Z = np.array(model.predict(mesh_points))
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Plot data points
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors='k', 
               cmap=plt.cm.coolwarm, alpha=0.8)
    plt.legend(*scatter.legend_elements(), title="Classes")
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(f'Feature {feature_idx1}')
    plt.ylabel(f'Feature {feature_idx2}')
    plt.title('Decision Boundary')
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
    print("Analyzing feature importance...")
    feature_entropy = calculate_feature_entropy(df_processed)
    
    # Select most informative features
    selected_features = select_features_by_distance(df_processed, n_features=6)
    print(f"Selected features: {selected_features}")
    
    # Keep only selected features plus target
    df_selected = df_processed[selected_features + ['Outcome']]
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_selected, test_size=0.2, stratify=True
    )
    
    # Plot feature distributions
    print("Generating feature distribution plots...")
    dist_path = os.path.join(PLOTS_PATH, "feature_distributions.png")
    plot_feature_distributions(df_selected, save_path=dist_path)
    
    # Plot feature correlations
    print("Generating feature correlation matrix...")
    corr_path = os.path.join(PLOTS_PATH, "feature_correlations.png")
    plot_feature_correlation(df_selected, save_path=corr_path)
    
    # Train the model
    print("Training custom diabetes classifier...")
    model = DiabetesClassifier(learning_rate=0.01, iterations=5000, regularization=0.01)
    model.train(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Calculate confusion matrix
    conf_matrix = calculate_confusion_matrix(y_test, y_pred)
    
    # Calculate distance-based metrics
    dist_metrics = distance_based_evaluation(model, X_test, y_test)
    
    # Analyze calibration
    calib_data = calibration_analysis(y_test, y_pred_proba)
    
    # Generate plots
    print("Generating plots...")
    
    # ROC curve
    roc_path = os.path.join(PLOTS_PATH, "roc_curve.png")
    auc_value = plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
    print(f"ROC curve saved to {roc_path}")
    
    # Confusion matrix
    cm_path = os.path.join(PLOTS_PATH, "confusion_matrix.png")
    plot_confusion_matrix(conf_matrix, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    
    # Decision boundary (using the first two selected features)
    if len(selected_features) >= 2:
        db_path = os.path.join(PLOTS_PATH, "decision_boundary.png")
        plot_decision_boundary(model, X_test, y_test, 0, 1, save_path=db_path)
        print(f"Decision boundary plot saved to {db_path}")
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Cross-Entropy Loss: {metrics['cross_entropy_loss']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"AUC: {auc_value:.4f}")
    
    # Add AUC to metrics
    metrics['auc'] = auc_value
    
    # Print distance-based metrics
    print("\nDistance-Based Metrics:")
    for key, value in dist_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = model.feature_importance(selected_features)
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Identify most challenging samples
    print("\nIdentifying most challenging samples...")
    
    # Get prediction probabilities
    probs = y_pred_proba
    
    # Find misclassified samples with high confidence
    misclassified = [(i, p) for i, (y, p, t) in enumerate(zip(y_pred, probs, y_test)) if y != t]
    
    if misclassified:
        # Sort by confidence (higher probability = higher confidence)
        misclassified.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        
        print("Top 5 most confidently misclassified samples:")
        for i, (idx, prob) in enumerate(misclassified[:5]):
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            confidence = max(prob, 1 - prob)
            
            print(f"  {i+1}. Sample {idx}:")
            print(f"     True label: {true_label}, Predicted: {pred_label}")
            print(f"     Confidence: {confidence:.4f}")
            
            # Extract feature values
            sample_features = {feature: X_test[idx, j] for j, feature in enumerate(selected_features)}
            print("     Feature values:")
            for feature, value in sample_features.items():
                print(f"       {feature}: {value:.4f}")
    
    # Save all results
    all_results = {
        'metrics': metrics,
        'confusion_matrix': conf_matrix,
        'distance_metrics': dist_metrics,
        'feature_importance': {k: float(v) for k, v in feature_importance.items()},
        'calibration': {
            'avg_error': float(calib_data['avg_calibration_error']),
            'max_error': float(calib_data['max_calibration_error'])
        }
    }
    
    # Save results
    results_file = os.path.join(RESULTS_PATH, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()