# __init__.py

"""
Diabetes Classifier Package
Contains modules for data processing, classification, evaluation, and utilities.
This implementation uses custom math functions from math_utils.py.
"""

# Import specific functions for easier access
from .data_processing import load_data, preprocess_data
from .classifier import DiabetesClassifier
from .evaluation import evaluate_model
from .math_utils import (
    sigmoid, mean, variance, standard_deviation,
    euclidean_distance, manhattan_distance,
    min_max_scaling, z_score_normalization,
    relu, leaky_relu, softmax, dot_product,
    cosine_similarity, entropy, mean_squared_error,
    cross_entropy_loss
)