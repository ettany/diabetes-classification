# Diabetes Classification Project

A machine learning project for diabetes prediction using custom mathematical functions without relying on external ML libraries.

## Project Overview

This project implements a diabetes classifier from scratch using custom mathematical functions. Instead of using scikit-learn or other ML libraries, we've built a custom implementation of logistic regression and evaluation metrics using the functions in our `math_utils.py` library.

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
│   │   └── diabetes.csv        # Original diabetes dataset
│   └── processed/              # Processed data
│
├── tests/                      # Unit tests
│   ├── test_math_utils.py      # Tests for math functions
│   └── test_classifier.py      # Tests for classification logic
│
├── scripts/                    # Executable scripts
│   ├── preprocess.py           # Data preprocessing
│   ├── train.py                # Model training
│   └── evaluate.py             # Model evaluation
│
├── results/                    # Directory for evaluation results
├── README.md                   # Project overview
└── requirements.txt            # Dependencies
```

## Features

- **Custom Mathematical Implementation**: Uses our own mathematical functions for all ML operations
- **Logistic Regression Classifier**: Built from scratch for diabetes prediction
- **Preprocessing Pipeline**: Custom data cleaning and normalization
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Unit Tests**: Ensuring mathematical functions work correctly

## Key Components

1. `math_utils.py`: Contains custom mathematical functions for:
   - Descriptive statistics (mean, variance, standard deviation)
   - Distance metrics (Euclidean, Manhattan)
   - Normalization (min-max scaling, z-score)
   - Activation functions (sigmoid, ReLU, Leaky ReLU)
   - Evaluation metrics (MSE, cross-entropy loss)

2. **Preprocess the data:**
   ```bash
   python scripts/preprocess.py
   ```

3. **Train the model:**
   ```bash
   python scripts/train.py
   ```

4. **Evaluate the model:**
   ```bash
   python scripts/evaluate.py
   ```

## Implementation Details

### Custom Logistic Regression

Our custom implementation of logistic regression uses gradient descent to optimize the model parameters. The implementation follows these steps:

1. Initialize weights and bias to zeros
2. For each iteration:
   - Calculate the linear model: z = X·w + b
   - Apply the sigmoid activation function
   - Compute gradients for weights and bias
   - Update parameters using the learning rate

```python
# Simplified example of our training logic
def train(X_train, y_train):
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for _ in range(iterations):
        linear_model = np.dot(X_train, weights) + bias
        y_predicted = np.array([sigmoid(x) for x in linear_model])
        
        dw = (1 / n_samples) * np.dot(X_train.T, (y_predicted - y_train))
        db = (1 / n_samples) * np.sum(y_predicted - y_train)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
```

### Data Preprocessing

Our preprocessing pipeline includes:

1. Handling missing values (zeros in certain features)
2. Feature normalization using z-score (from our math_utils library)
3. Feature engineering, including BMI and age categorization

### Evaluation Metrics

We calculate metrics using our custom functions from math_utils.py:

- Accuracy: Proportion of correct predictions
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
- MSE: Custom implementation from math_utils
- Cross-Entropy Loss: Custom implementation from math_utils
- ROC Curve and AUC: Manual implementation without sklearn

## Dataset

The Pima Indians Diabetes Dataset consists of medical predictor variables including:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Class variable (0 or 1) - 1 means tested positive for diabetes

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request. `classifier.py`: Custom logistic regression implementation using:
   - Sigmoid activation function
   - Gradient descent optimization
   - Custom prediction logic

3. `evaluation.py`: Model evaluation logic with custom metrics:
   - Accuracy, precision, recall, F1 score
   - Mean squared error and cross-entropy loss
   - ROC curve and AUC calculation

## Installation & Usage

1. **Setup the environment:**
   ```bash
   pip install -r requirements.txt
   ```

2.