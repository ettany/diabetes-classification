import numpy as np
from .math_utils import mean_squared_error, cross_entropy_loss

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance using multiple metrics.
    
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
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate cross-entropy loss
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
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'mse': mse,
        'cross_entropy_loss': ce_loss,
        'precision': precision,
        'recall': recall, 
        'f1_score': f1
    }