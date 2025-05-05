import numpy as np
from .math_utils import sigmoid, dot_product, mean_squared_error

class DiabetesClassifier:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def train(self, X_train, y_train):
        """
        Train logistic regression model using gradient descent.
        
        Parameters:
        X_train: Features matrix
        y_train: Target values
        """
        # Initialize parameters
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
            # Linear model: z = X · w + b
            linear_model = np.dot(X_train, self.weights) + self.bias
            
            # Apply sigmoid activation
            y_predicted = np.array([sigmoid(x) for x in linear_model])
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X_train.T, (y_predicted - y_train))
            db = (1 / n_samples) * np.sum(y_predicted - y_train)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """Return probability estimates for samples in X"""
        linear_model = np.dot(X, self.weights) + self.bias
        return np.array([sigmoid(x) for x in linear_model])
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        probas = self.predict_proba(X)
        return [1 if p > 0.5 else 0 for p in probas]