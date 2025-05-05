import os
import sys
import json

# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the package
from diabetes_classifier import DiabetesClassifier, load_data, preprocess_data
from diabetes_classifier.data_processing import train_test_split
from diabetes_classifier.evaluation import evaluate_model

# Define paths
DATA_PATH = "data/raw/diabetes.csv"
RESULTS_PATH = "results"

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)

def main():
    # Load and preprocess data
    print("Loading data from", DATA_PATH)
    df = load_data(DATA_PATH)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(df_processed, test_size=0.2)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Train the model
    print("Training custom diabetes classifier...")
    model = DiabetesClassifier(learning_rate=0.01, iterations=5000)
    model.train(X_train, y_train)
    
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
    
    # Save results
    results_file = os.path.join(RESULTS_PATH, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()