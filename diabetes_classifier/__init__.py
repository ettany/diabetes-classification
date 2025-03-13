# __init__.py

"""
Diabetes Classifier Package
Contains modules for data processing, classification, evaluation, and utilities.
"""

# Optionally, you can import specific functions for easier access
from .data_processing import load_data, preprocess_data
from .classifier import DiabetesClassifier
from .evaluation import evaluate_model
from .math_utils import sigmoid
