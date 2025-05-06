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
│   ├── dataset_analyzer.py     # Statistical dataset analysis
│   ├── preprocess.py           # Data preprocessing
│   ├── train.py                # Model training
│   ├── evaluate.py             # Model evaluation
│   ├── visualize_results.py    # Advanced results visualization
│
├── results/                    # Results storage
│   ├── analysis/               # Dataset analysis results
│   ├── plots/                  # Evaluation plots
│   ├── visualizations/         # Advanced visualizations
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

## Scripts Description

### dataset_analyzer.py
- Analyzes the diabetes dataset using custom math functions
- Calculates feature statistics like mean, variance, std dev
- Measures feature correlation using cosine similarity
- Analyzes class separation for each feature
- Calculates entropy-based metrics for feature selection
- Generates visualizations of feature distributions

### visualize_results.py
- Creates advanced visualizations of model results
- Generates radar charts for feature importance
- Creates feature correlation network graphs
- Produces comprehensive feature profiles
- Visualizes decision boundaries
- Shows pairwise feature relationships

### preprocess.py
- Handles missing values using custom statistics
- Applies different normalization techniques based on feature characteristics
- Creates feature engineering steps
- Splits data with stratification

### train.py
- Trains the classification model with regularization
- Tracks training loss
- Evaluates model performance
- Calculates feature importance
- Generates performance plots

### evaluate.py
- Comprehensive model evaluation
- Generates ROC curve and confusion matrix
- Calculates distance-based metrics
- Analyzes model calibration
- Creates detailed evaluation reports

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/diabetes-classifier.git
cd diabetes-classifier

# Install dependencies
pip install -r requirements.txt
```

### Data Analysis

```bash
# Analyze the diabetes dataset
python scripts/dataset_analyzer.py
```

### Data Preprocessing

```bash
# Preprocess the data
python scripts/preprocess.py
```

### Model Training

```bash
# Train the classification model
python scripts/train.py
```

### Model Evaluation

```bash
# Evaluate the model
python scripts/evaluate.py
```

### Results Visualization

```bash
# Create advanced visualizations
python scripts/visualize_results.py
```

## Dataset

The project uses the Pima Indians Diabetes Dataset, which contains various health-related features for predicting diabetes diagnosis.

Features include:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Class variable (0 or 1, where 1 indicates diabetes)

## Custom Implementation Benefits

By implementing mathematical operations from scratch, this project offers:

1. **Educational Value**: Better understanding of the mathematical foundations
2. **Customizability**: Ability to modify algorithms for specific needs
3. **Transparency**: Clear visibility into how each calculation works
4. **Performance Control**: Fine-grained control over numeric operations
5. **Reduced Dependencies**: Fewer external library requirements

## Results and Visualization

The project generates various visualizations to help understand both the dataset and model performance:

- Feature distribution plots showing class separation
- Correlation heatmaps using cosine similarity
- ROC curves and confusion matrices
- Feature importance visualizations
- Decision boundary plots
- Calibration curves
- Feature profiles combining multiple metrics
- Feature correlation networks

## Future Enhancements

- Implementation of additional classification algorithms
- Support for multiclass classification problems
- Feature engineering improvements
- Hyperparameter optimization using custom metrics
- Interactive visualization dashboard

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
