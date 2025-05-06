import numpy as np
from .math_utils import (
    sigmoid, dot_product, mean_squared_error, 
    cross_entropy_loss, cosine_similarity
)

class DiabetesClassifier:
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.training_losses = []

    def train(self, X_train, y_train):
        """
        Train logistic regression model using gradient descent with L2 regularization.
        
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
            # Compute linear model using custom dot product
            linear_model = np.array([dot_product(x, self.weights) for x in X_train]) + self.bias
            
            # Apply sigmoid activation
            y_predicted = np.array([sigmoid(x) for x in linear_model])
            
            # Compute loss with regularization
            y_pred_reshaped = np.array([[1-p, p] for p in y_predicted])
            y_train_one_hot = np.zeros((len(y_train), 2))
            for i, val in enumerate(y_train):
                y_train_one_hot[i, val] = 1
            
            loss = cross_entropy_loss(y_train_one_hot, y_pred_reshaped)
            loss += (self.regularization / (2 * n_samples)) * dot_product(self.weights, self.weights)
            self.training_losses.append(loss)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X_train.T, (y_predicted - y_train))
            # Add L2 regularization term
            dw += (self.regularization / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y_train)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """Return probability estimates for samples in X"""
        # Using dot product from math utils for each sample
        linear_model = np.array([dot_product(x, self.weights) for x in X]) + self.bias
        return np.array([sigmoid(x) for x in linear_model])
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        probas = self.predict_proba(X)
        return [1 if p > 0.5 else 0 for p in probas]
    
    def feature_importance(self, feature_names):
        """Calculate feature importance based on weight magnitude"""
        if self.weights is None:
            raise ValueError("Model has not been trained yet")
        
        # Get absolute weight values
        importance = np.abs(self.weights)
        # Normalize to sum to 1
        importance = importance / np.sum(importance)
        
        return dict(zip(feature_names, importance))
    
    def similarity_score(self, X1, X2):
        """
        Calculate similarity between two samples using cosine similarity.
        Useful for finding similar patients in the dataset.
        """
        return cosine_similarity(X1, X2)