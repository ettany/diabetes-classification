# Diabetes Classifier

A machine learning project for diabetes prediction using custom mathematical implementations.

## Overview

This project implements a diabetes classification system using logistic regression with custom mathematical functions. Instead of relying on standard libraries like scikit-learn for all operations, many core mathematical functions are implemented from scratch in the `math_utils.py` module, providing educational value and greater control over the implementation.

## Key Features

- Custom implementation of common mathematical functions (mean, variance, distance metrics, etc.)
- Custom implementation of machine learning operations (sigmoid, softmax, loss functions, etc.)
- Logistic regression classifier with L2 regularization
- Multiple feature normalization techniques (z-score, min-max)
- Feature selection based on class separation metrics
- Comprehensive model evaluation metrics
- Visualization of model performance and feature importance
- Advanced analysis including decision boundaries and calibration curves

## Project Structure

```
├── diabetes_classifier/        # Main package
│   ├── __init__.py             # Package initialization
│   ├── math_utils.py           # Custom mathematical functions
│   ├── data_processing.py      # Data loading and preprocessing
│   ├── classifier.py           # Custom classification logic
│   ├── evaluation.py           # Model evaluation
│
├── data/                       # Dataset storage
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│
├── tests/                      # Unit tests
│   ├── test_math_utils.py      # Tests for math functions
│   ├── test_classifier.py      # Tests for classification logic
│
├── scripts/                    # Executable scripts
│   ├── preprocess.py           # Data preprocessing
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
```

## Mathematical Functions

The project includes custom implementations of:

### Basic Statistics
- Mean, variance, standard deviation
- Min-max scaling, z-score normalization

### Distance Metrics
- Euclidean distance
- Manhattan distance
- Cosine similarity

### ML Functions
- Sigmoid activation
- ReLU and Leaky ReLU activations
- Softmax function
- Dot product operations

### Loss Functions
- Mean squared error
- Cross-entropy loss
- Entropy calculation
- Gini index
- Information gain

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/diabetes-classifier.git
cd diabetes-classifier

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing

```bash
python scripts/preprocess.py
```

### Model Training

```bash
python scripts/train.py
```

### Model Evaluation

```bash
python scripts/evaluate.py
```

## Dataset

The project uses the Pima Indians Diabetes Dataset, which contains various health-related features such as glucose level, blood pressure, BMI, etc., for predicting diabetes diagnosis.

Features include:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target variable)

## Future Enhancements

- Implementation of additional classification algorithms
- Support for multiclass classification problems
- Feature engineering improvements
- Hyperparameter optimization using custom metrics
- Interactive visualization dashboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
