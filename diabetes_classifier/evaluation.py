import numpy as np
from .math_utils import (
    mean_squared_error, cross_entropy_loss, 
    entropy, cosine_similarity, euclidean_distance
)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance using multiple metrics from math_utils.
    
    Parameters:
    model: Trained classifier
    X_test: Test features
    y_test: True labels
    
    Returns:
    Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    correct = sum(1 for pred, true in zip(y_pred, y_test) if pred == true)
    accuracy = correct / len(y_test)
    
    # Calculate MSE using custom function
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate cross-entropy loss using custom function
    # Convert y_test to one-hot encoding for cross-entropy
    y_test_one_hot = np.zeros((len(y_test), 2))
    for i, val in enumerate(y_test):
        y_test_one_hot[i, val] = 1
    
    # Reshape predictions for cross-entropy
    y_pred_proba_reshaped = np.array([[1-p, p] for p in y_pred_proba])
    
    # Calculate cross-entropy loss
    ce_loss = cross_entropy_loss(y_test_one_hot, y_pred_proba_reshaped)
    
    # Calculate precision, recall, f1-score
    true_positives = sum(1 for pred, true in zip(y_pred, y_test) if pred == 1 and true == 1)
    false_positives = sum(1 for pred, true in zip(y_pred, y_test) if pred == 1 and true == 0)
    false_negatives = sum(1 for pred, true in zip(y_pred, y_test) if pred == 0 and true == 1)
    true_negatives = sum(1 for pred, true in zip(y_pred, y_test) if pred == 0 and true == 0)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate specificity
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # Calculate class probabilities for entropy calculation
    overall_probs = [y_test.tolist().count(0)/len(y_test), y_test.tolist().count(1)/len(y_test)]
    pred_probs = [y_pred.count(0)/len(y_pred), y_pred.count(1)/len(y_pred)]
    
    # Calculate prediction entropy
    pred_entropy = entropy(pred_probs)
    
    # Calculate prediction KL divergence (relative entropy)
    kl_divergence = sum(p * np.log2(p/q) if p > 0 and q > 0 else 0 
                        for p, q in zip(overall_probs, pred_probs))
    
    # Calculate calibration error (mean absolute difference between predicted probs and actual)
    calibration_error = np.mean([abs(prob - actual) 
                                for prob, actual in zip(y_pred_proba, y_test)])
    
    return {
        'accuracy': accuracy,
        'mse': mse,
        'cross_entropy_loss': ce_loss,
        'precision': precision,
        'recall': recall, 
        'f1_score': f1,
        'specificity': specificity,
        'prediction_entropy': pred_entropy,
        'kl_divergence': kl_divergence,
        'calibration_error': calibration_error
    }

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for binary classification.
    
    Parameters:
    y_true: True labels
    y_pred: Predicted labels
    
    Returns:
    Dictionary with confusion matrix values
    """
    true_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    false_positives = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    false_negatives = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    true_negatives = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }

def distance_based_evaluation(model, X_test, y_test):
    """
    Evaluate model using distance-based metrics.
    
    This function measures how well the model separates classes in the feature space
    using distance metrics from math_utils.
    
    Parameters:
    model: Trained classifier
    X_test: Test features
    y_test: True labels
    
    Returns:
    Dictionary with distance-based metrics
    """
    # Split data by actual class
    X_positive = X_test[y_test == 1]
    X_negative = X_test[y_test == 0]
    
    # Get predictions
    y_pred = np.array(model.predict(X_test))
    
    # Split data by predicted class
    X_pred_positive = X_test[y_pred == 1]
    X_pred_negative = X_test[y_pred == 0]
    
    # Calculate metrics
    metrics = {}
    
    # Calculate class separation (if we have samples in both classes)
    if len(X_positive) > 0 and len(X_negative) > 0:
        # Calculate centroid for each class
        positive_centroid = np.mean(X_positive, axis=0)
        negative_centroid = np.mean(X_negative, axis=0)
        
        # Calculate distance between centroids
        euclidean_class_separation = euclidean_distance(positive_centroid, negative_centroid)
        metrics['euclidean_class_separation'] = euclidean_class_separation
        
        # Calculate average within-class distance (cohesion)
        if len(X_positive) > 1:
            positive_distances = [euclidean_distance(x, positive_centroid) for x in X_positive]
            metrics['positive_class_cohesion'] = sum(positive_distances) / len(positive_distances)
        
        if len(X_negative) > 1:
            negative_distances = [euclidean_distance(x, negative_centroid) for x in X_negative]
            metrics['negative_class_cohesion'] = sum(negative_distances) / len(negative_distances)
    
    # Calculate prediction consistency using cosine similarity
    # Higher similarity indicates more consistent predictions
    if len(X_test) > 1:
        # Calculate cosine similarity between consecutive samples with same prediction
        similarities = []
        for i in range(len(X_test) - 1):
            if y_pred[i] == y_pred[i+1]:
                similarities.append(cosine_similarity(X_test[i], X_test[i+1]))
        
        if similarities:
            metrics['prediction_consistency'] = sum(similarities) / len(similarities)
    
    return metrics

def calibration_analysis(y_true, y_pred_proba, n_bins=10):
    """
    Analyze model calibration using custom math utilities.
    
    A well-calibrated model has predicted probabilities that match observed frequencies.
    
    Parameters:
    y_true: True labels
    y_pred_proba: Predicted probabilities
    n_bins: Number of bins for analysis
    
    Returns:
    Calibration metrics
    """
    # Create bins of predicted probabilities
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure valid indices
    
    # Calculate observed frequencies and predicted probabilities for each bin
    bin_metrics = []
    for bin_idx in range(n_bins):
        # Get samples in this bin
        bin_samples = bin_indices == bin_idx
        
        if np.sum(bin_samples) > 0:
            # Calculate mean predicted probability in this bin
            mean_pred_prob = np.mean(y_pred_proba[bin_samples])
            
            # Calculate observed frequency in this bin
            observed_freq = np.mean(y_true[bin_samples])
            
            # Calculate absolute calibration error
            calibration_error = abs(mean_pred_prob - observed_freq)
            
            bin_metrics.append({
                'bin_idx': bin_idx,
                'bin_start': bin_edges[bin_idx],
                'bin_end': bin_edges[bin_idx + 1],
                'sample_count': np.sum(bin_samples),
                'mean_predicted_prob': mean_pred_prob,
                'observed_frequency': observed_freq,
                'calibration_error': calibration_error
            })
    
    # Calculate overall calibration metrics
    if bin_metrics:
        avg_calibration_error = np.mean([m['calibration_error'] for m in bin_metrics])
        max_calibration_error = max([m['calibration_error'] for m in bin_metrics])
    else:
        avg_calibration_error = 0
        max_calibration_error = 0
    
    return {
        'bin_metrics': bin_metrics,
        'avg_calibration_error': avg_calibration_error,
        'max_calibration_error': max_calibration_error
    }