import numpy as np

def mean(values):
    """Calculate the mean (average) of a list of numbers.
    
    The mean is computed as the sum of all values divided by the number of values.
    """
    return sum(values) / len(values)

def variance(values):
    """Calculate the variance of a list of numbers.
    
    Variance measures how spread out the values are from the mean.
    It is calculated as the average of the squared differences from the mean.
    """
    avg = mean(values)
    return sum((x - avg) ** 2 for x in values) / len(values)

def standard_deviation(values):
    """Calculate the standard deviation of a list of numbers.
    
    Standard deviation is the square root of variance and represents the dispersion of values.
    """
    return np.sqrt(variance(values))

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points.
    
    Euclidean distance is the straight-line distance between two points in n-dimensional space.
    Formula: sqrt(sum((x_i - y_i)^2))
    """
    return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def manhattan_distance(point1, point2):
    """Calculate the Manhattan distance between two points.
    
    Manhattan distance (L1 norm) is the sum of absolute differences of coordinates.
    Formula: sum(|x_i - y_i|)
    """
    return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))

def min_max_scaling(values):
    """Perform Min-Max scaling on a list of values.
    
    This normalization technique rescales values between 0 and 1.
    Formula: (x - min) / (max - min)
    """
    min_val, max_val = min(values), max(values)
    return [(x - min_val) / (max_val - min_val) for x in values]

def z_score_normalization(values):
    """Perform Z-score normalization on a list of values.
    
    This normalization scales data based on mean and standard deviation.
    Formula: (x - mean) / std_dev
    """
    avg = mean(values)
    std_dev = standard_deviation(values)
    return [(x - avg) / std_dev for x in values] if std_dev != 0 else [0 for _ in values]

def sigmoid(x):
    """Compute the sigmoid activation function.
    
    The sigmoid function maps values to the range (0, 1), often used in logistic regression.
    Formula: 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """Compute the ReLU (Rectified Linear Unit) activation function.
    
    ReLU returns 0 for negative inputs and x for positive inputs.
    Formula: max(0, x)
    """
    return max(0, x)

def leaky_relu(x, alpha=0.01):
    """Compute the Leaky ReLU activation function.
    
    Unlike ReLU, Leaky ReLU allows a small gradient for negative inputs.
    Formula: x if x > 0 else alpha * x
    """
    return x if x > 0 else alpha * x

def softmax(values):
    """Compute the softmax function for a list of values.
    
    Softmax converts raw scores (logits) into probabilities summing to 1.
    A stability improvement is applied by subtracting the maximum value.
    Formula: exp(x) / sum(exp(x))
    """
    exp_values = np.exp(values - np.max(values))  # Stability improvement
    return exp_values / exp_values.sum()

def dot_product(vector1, vector2):
    """Compute the dot product of two vectors.
    
    The dot product is a measure of vector similarity.
    Formula: sum(a_i * b_i)
    """
    return sum(x * y for x, y in zip(vector1, vector2))

def cosine_similarity(vector1, vector2):
    """Compute the cosine similarity between two vectors.
    
    Cosine similarity measures the angle between two vectors.
    Formula: dot(A, B) / (||A|| * ||B||)
    """
    dot_prod = dot_product(vector1, vector2)
    norm1 = np.sqrt(dot_product(vector1, vector1))
    norm2 = np.sqrt(dot_product(vector2, vector2))
    return dot_prod / (norm1 * norm2) if norm1 and norm2 else 0

def entropy(probabilities):
    """Compute the entropy of a probability distribution.
    
    Entropy quantifies uncertainty in a probability distribution.
    Formula: -sum(p * log2(p))
    """
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def gini_index(groups, classes):
    """Compute the Gini index for a split dataset.
    
    The Gini index measures data impurity in classification tasks.
    Formula: Gini = 1 - sum(p^2)
    """
    total_samples = sum(len(group) for group in groups)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion ** 2
        gini += (1.0 - score) * (size / total_samples)
    return gini

def information_gain(parent_entropy, left_child, right_child):
    """Compute information gain from splitting a dataset.
    
    Information gain measures the reduction in entropy after a split.
    Formula: IG = Parent_Entropy - Weighted_Avg_Child_Entropy
    """
    total_samples = len(left_child) + len(right_child)
    left_weight = len(left_child) / total_samples
    right_weight = len(right_child) / total_samples
    child_entropy = left_weight * entropy(left_child) + right_weight * entropy(right_child)
    return parent_entropy - child_entropy

def mean_squared_error(y_true, y_pred):
    """Compute Mean Squared Error (MSE) between actual and predicted values.
    
    MSE measures the average squared difference between predictions and actual values.
    Formula: (1/n) * sum((y_true - y_pred)^2)
    """
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def cross_entropy_loss(y_true, y_pred):
    """Compute the cross-entropy loss.
    
    Cross-entropy loss is used in classification models to compare predicted probabilities.
    To prevent log(0), predictions are clipped to a small value.
    Formula: -sum(y_true * log(y_pred)) / n
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)
