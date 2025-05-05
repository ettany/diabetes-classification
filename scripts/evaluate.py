import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to Python's path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"Added path: {project_root}")

# Now import the package
from diabetes_classifier import DiabetesClassifier, load_data, preprocess_data, evaluate_model
from diabetes_classifier.data_processing import train_test_split

# Define paths
DATA_PATH = "data/raw/diabetes.csv"
MODEL_PATH = "models/diabetes_model.pkl"
RESULTS_PATH = "results"

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)

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

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    # Calculate confusion matrix
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    
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

def main():
    # Load and preprocess data
    print("Loading data from", DATA_PATH)
    df = load_data(DATA_PATH)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(df_processed, test_size=0.2)
    
    # Train the model
    print("Training custom diabetes classifier...")
    model = DiabetesClassifier(learning_rate=0.01, iterations=5000)
    model.train(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Cross-Entropy Loss: {metrics['cross_entropy_loss']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Generate and save plots
    print("Generating plots...")
    roc_path = os.path.join(RESULTS_PATH, "roc_curve.png")
    cm_path = os.path.join(RESULTS_PATH, "confusion_matrix.png")
    
    auc_value = plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
    plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
    
    # Add AUC to metrics
    metrics['auc'] = auc_value
    
    # Save results
    results_file = os.path.join(RESULTS_PATH, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to {results_file}")
    print(f"ROC curve saved to {roc_path}")
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    main()