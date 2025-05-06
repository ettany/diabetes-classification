import unittest
import numpy as np
import math
from diabetes_classifier.math_utils import (
    mean, variance, standard_deviation, euclidean_distance,
    manhattan_distance, min_max_scaling, z_score_normalization,
    sigmoid, relu, leaky_relu, softmax, dot_product,
    cosine_similarity, entropy, gini_index, information_gain,
    mean_squared_error, cross_entropy_loss
)

class TestMathUtils(unittest.TestCase):
    """Test cases for math_utils.py module."""
    
    def setUp(self):
        """Set up test data."""
        # Basic numeric lists
        self.empty_list = []
        self.single_value = [5]
        self.zeros = [0, 0, 0, 0, 0]
        self.ones = [1, 1, 1, 1, 1]
        self.values = [1, 2, 3, 4, 5]
        self.negative_values = [-5, -4, -3, -2, -1]
        self.mixed_values = [-2, -1, 0, 1, 2]
        
        # Points and vectors
        self.point1 = [1, 2, 3]
        self.point2 = [4, 5, 6]
        self.point3 = [1, 2, 3]  # Same as point1 for identity tests
        self.origin = [0, 0, 0]
        self.unit_x = [1, 0, 0]
        self.unit_y = [0, 1, 0]
        self.unit_z = [0, 0, 1]
        
        # Probability distributions
        self.uniform_probs = [0.25, 0.25, 0.25, 0.25]
        self.certain_probs = [1.0, 0.0, 0.0, 0.0]
        self.skewed_probs = [0.1, 0.2, 0.3, 0.4]
        self.binary_probs = [0.3, 0.7]
        
        # Classification data
        self.y_true_binary = [0, 1, 0, 1, 0]
        self.y_pred_binary = [0, 1, 1, 1, 0]
        self.y_pred_probs = [0.2, 0.8, 0.6, 0.9, 0.3]
        
        # One-hot encoded data
        self.y_true_onehot = np.array([
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0]
        ])
        self.y_pred_onehot = np.array([
            [0.8, 0.2],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.1, 0.9],
            [0.7, 0.3]
        ])
        
        # Dataset for gini index and information gain tests
        self.left_child = [
            [1.2, 0.5, 0],
            [2.3, 1.1, 0],
            [2.5, 0.7, 0]
        ]
        self.right_child = [
            [3.1, 1.5, 1],
            [3.5, 0.9, 1],
            [1.9, 1.2, 1]
        ]
        self.classes = [0, 1]

    # ===== Tests for mean =====
    def test_mean(self):
        """Test mean calculation."""
        # Basic test cases
        self.assertEqual(mean(self.values), 3.0)
        self.assertEqual(mean(self.zeros), 0.0)
        self.assertEqual(mean(self.ones), 1.0)
        self.assertEqual(mean(self.negative_values), -3.0)
        self.assertEqual(mean(self.mixed_values), 0.0)
        
        # Edge cases
        self.assertEqual(mean(self.single_value), 5.0)
        with self.assertRaises(ZeroDivisionError):
            mean(self.empty_list)

    # ===== Tests for variance =====
    def test_variance(self):
        """Test variance calculation."""
        # Basic test cases
        self.assertAlmostEqual(variance(self.values), 2.0)
        self.assertEqual(variance(self.zeros), 0.0)
        self.assertEqual(variance(self.ones), 0.0)
        self.assertAlmostEqual(variance(self.negative_values), 2.0)
        self.assertAlmostEqual(variance(self.mixed_values), 2.0)
        
        # Edge cases
        self.assertEqual(variance(self.single_value), 0.0)
        with self.assertRaises(ZeroDivisionError):
            variance(self.empty_list)

    # ===== Tests for standard_deviation =====
    def test_standard_deviation(self):
        """Test standard deviation calculation."""
        # Basic test cases
        self.assertAlmostEqual(standard_deviation(self.values), math.sqrt(2.0))
        self.assertEqual(standard_deviation(self.zeros), 0.0)
        self.assertEqual(standard_deviation(self.ones), 0.0)
        self.assertAlmostEqual(standard_deviation(self.negative_values), math.sqrt(2.0))
        self.assertAlmostEqual(standard_deviation(self.mixed_values), math.sqrt(2.0))
        
        # Edge cases
        self.assertEqual(standard_deviation(self.single_value), 0.0)
        with self.assertRaises(ZeroDivisionError):
            standard_deviation(self.empty_list)

    # ===== Tests for euclidean_distance =====
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        # Basic test cases
        self.assertAlmostEqual(euclidean_distance(self.point1, self.point2), math.sqrt(27))
        self.assertEqual(euclidean_distance(self.point1, self.point3), 0.0)  # Same points
        self.assertAlmostEqual(euclidean_distance(self.origin, self.point1), math.sqrt(14))
        
        # Orthogonal vectors
        self.assertEqual(euclidean_distance(self.unit_x, self.unit_y), math.sqrt(2))
        
        # Edge cases
        with self.assertRaises(Exception):
            euclidean_distance(self.point1, [])  # Different dimensions

    # ===== Tests for manhattan_distance =====
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        # Basic test cases
        self.assertEqual(manhattan_distance(self.point1, self.point2), 9)
        self.assertEqual(manhattan_distance(self.point1, self.point3), 0)  # Same points
        self.assertEqual(manhattan_distance(self.origin, self.point1), 6)
        
        # Orthogonal vectors
        self.assertEqual(manhattan_distance(self.unit_x, self.unit_y), 2)
        
        # Edge cases
        with self.assertRaises(Exception):
            manhattan_distance(self.point1, [])  # Different dimensions

    # ===== Tests for min_max_scaling =====
    def test_min_max_scaling(self):
        """Test min-max scaling."""
        # Basic test cases
        scaled_values = min_max_scaling(self.values)
        self.assertEqual(min(scaled_values), 0.0)
        self.assertEqual(max(scaled_values), 1.0)
        self.assertEqual(scaled_values, [0.0, 0.25, 0.5, 0.75, 1.0])
        
        scaled_negative = min_max_scaling(self.negative_values)
        self.assertEqual(min(scaled_negative), 0.0)
        self.assertEqual(max(scaled_negative), 1.0)
        self.assertEqual(scaled_negative, [0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Edge cases
        scaled_constant = min_max_scaling(self.ones)
        self.assertEqual(scaled_constant, [0.0, 0.0, 0.0, 0.0, 0.0])  # All same value
        
        scaled_single = min_max_scaling(self.single_value)
        self.assertEqual(scaled_single, [0.0])  # Single value
        
        with self.assertRaises(Exception):
            min_max_scaling(self.empty_list)  # Empty list

    # ===== Tests for z_score_normalization =====
    def test_z_score_normalization(self):
        """Test z-score normalization."""
        # Basic test cases
        normalized_values = z_score_normalization(self.values)
        self.assertAlmostEqual(mean(normalized_values), 0.0, places=10)
        self.assertAlmostEqual(standard_deviation(normalized_values), 1.0)
        
        normalized_negative = z_score_normalization(self.negative_values)
        self.assertAlmostEqual(mean(normalized_negative), 0.0, places=10)
        self.assertAlmostEqual(standard_deviation(normalized_negative), 1.0)
        
        # Edge cases
        normalized_constant = z_score_normalization(self.ones)
        self.assertEqual(normalized_constant, [0.0, 0.0, 0.0, 0.0, 0.0])  # All same value
        
        normalized_single = z_score_normalization(self.single_value)
        self.assertEqual(normalized_single, [0.0])  # Single value
        
        with self.assertRaises(Exception):
            z_score_normalization(self.empty_list)  # Empty list

    # ===== Tests for sigmoid =====
    def test_sigmoid(self):
        """Test sigmoid function."""
        # Basic test cases
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertTrue(0 < sigmoid(-5) < 0.5)
        self.assertTrue(0.5 < sigmoid(5) < 1)
        
        # Symmetry property: sigmoid(-x) = 1 - sigmoid(x)
        for x in self.values:
            self.assertAlmostEqual(sigmoid(-x), 1 - sigmoid(x))
        
        # Edge cases
        self.assertAlmostEqual(sigmoid(100), 1.0, places=10)  # Very large positive
        self.assertAlmostEqual(sigmoid(-100), 0.0, places=10)  # Very large negative

    # ===== Tests for relu =====
    def test_relu(self):
        """Test ReLU activation function."""
        # Basic test cases
        self.assertEqual(relu(0), 0)
        self.assertEqual(relu(5), 5)
        self.assertEqual(relu(-5), 0)
        
        # Test with lists
        relu_values = [relu(x) for x in self.mixed_values]
        self.assertEqual(relu_values, [0, 0, 0, 1, 2])

    # ===== Tests for leaky_relu =====
    def test_leaky_relu(self):
        """Test Leaky ReLU activation function."""
        # With default alpha
        self.assertEqual(leaky_relu(5), 5)
        self.assertEqual(leaky_relu(0), 0)
        self.assertEqual(leaky_relu(-5), -0.05)  # Default alpha=0.01
        
        # With custom alpha
        self.assertEqual(leaky_relu(-5, alpha=0.1), -0.5)
        self.assertEqual(leaky_relu(-2, alpha=0.2), -0.4)
        
        # Test with lists and different alphas
        leaky_values_default = [leaky_relu(x) for x in self.mixed_values]
        expected_default = [-0.02, -0.01, 0, 1, 2]
        for actual, expected in zip(leaky_values_default, expected_default):
            self.assertAlmostEqual(actual, expected)
        
        leaky_values_custom = [leaky_relu(x, alpha=0.1) for x in self.mixed_values]
        expected_custom = [-0.2, -0.1, 0, 1, 2]
        for actual, expected in zip(leaky_values_custom, expected_custom):
            self.assertAlmostEqual(actual, expected)

    # ===== Tests for softmax =====
    def test_softmax(self):
        """Test softmax function."""
        # Basic test cases
        soft_values = softmax(self.values)
        
        # Check properties of softmax
        self.assertAlmostEqual(sum(soft_values), 1.0)  # Sum to 1
        self.assertTrue(all(0 < x < 1 for x in soft_values))  # All values between 0 and 1
        
        # Should maintain order
        for i in range(1, len(soft_values)):
            self.assertTrue(soft_values[i] > soft_values[i-1])
        
        # Check numerical stability with large values
        large_values = [100, 200, 300]
        large_soft = softmax(large_values)
        self.assertAlmostEqual(sum(large_soft), 1.0)
        self.assertAlmostEqual(large_soft[2], 1.0, places=10)  # Largest dominates
        
        # Edge cases
        uniform_case = softmax([5, 5, 5])
        for val in uniform_case:
            self.assertAlmostEqual(val, 1/3)  # Equal distribution
        
        with self.assertRaises(Exception):
            softmax([])  # Empty list

    # ===== Tests for dot_product =====
    def test_dot_product(self):
        """Test dot product calculation."""
        # Basic test cases
        self.assertEqual(dot_product(self.point1, self.point2), 32)
        self.assertEqual(dot_product(self.point1, self.origin), 0)
        
        # Orthogonal vectors
        self.assertEqual(dot_product(self.unit_x, self.unit_y), 0)
        self.assertEqual(dot_product(self.unit_x, self.unit_z), 0)
        self.assertEqual(dot_product(self.unit_y, self.unit_z), 0)
        
        # Parallel vectors
        self.assertEqual(dot_product(self.unit_x, [2, 0, 0]), 2)
        
        # Edge cases
        self.assertEqual(dot_product([], []), 0)  # Empty vectors
        with self.assertRaises(Exception):
            dot_product(self.point1, [1, 2])  # Different dimensions

    # ===== Tests for cosine_similarity =====
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        # Basic test cases
        self.assertAlmostEqual(
            cosine_similarity(self.point1, self.point2),
            32 / (math.sqrt(14) * math.sqrt(77))
        )
        
        # Orthogonal vectors
        self.assertAlmostEqual(cosine_similarity(self.unit_x, self.unit_y), 0)
        
        # Parallel vectors
        self.assertAlmostEqual(cosine_similarity(self.unit_x, [2, 0, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity(self.point1, [-1, -2, -3]), -1.0)  # Opposite
        
        # Same vectors
        self.assertAlmostEqual(cosine_similarity(self.point1, self.point1), 1.0)
        
        # Edge cases
        self.assertEqual(cosine_similarity(self.origin, self.point1), 0)  # Zero vector
        self.assertEqual(cosine_similarity([], []), 0)  # Empty vectors
        with self.assertRaises(Exception):
            cosine_similarity(self.point1, [1, 2])  # Different dimensions

    # ===== Tests for entropy =====
    def test_entropy(self):
        """Test entropy calculation."""
        # Basic test cases
        uniform_entropy = entropy(self.uniform_probs)
        self.assertAlmostEqual(uniform_entropy, 2.0)  # log2(4) = 2
        
        certain_entropy = entropy(self.certain_probs)
        self.assertAlmostEqual(certain_entropy, 0.0)  # No uncertainty
        
        skewed_entropy = entropy(self.skewed_probs)
        expected_skewed = -(0.1*math.log2(0.1) + 0.2*math.log2(0.2) + 
                          0.3*math.log2(0.3) + 0.4*math.log2(0.4))
        self.assertAlmostEqual(skewed_entropy, expected_skewed)
        
        binary_entropy = entropy(self.binary_probs)
        expected_binary = -(0.3*math.log2(0.3) + 0.7*math.log2(0.7))
        self.assertAlmostEqual(binary_entropy, expected_binary)
        
        # Edge cases
        zero_probs = [0, 1.0]
        zero_entropy = entropy(zero_probs)
        self.assertAlmostEqual(zero_entropy, 0.0)  # 0*log(0) should be treated as 0

    # ===== Tests for gini_index =====
    def test_gini_index(self):
        """Test Gini index calculation."""
        groups = [self.left_child, self.right_child]
        
        # Calculate Gini index
        gini = gini_index(groups, self.classes)
        
        # Manual calculation for comparison
        left_size = len(self.left_child)
        right_size = len(self.right_child)
        total_size = left_size + right_size
        
        # Count classes in left group
        left_class_0 = sum(1 for sample in self.left_child if sample[-1] == 0)
        left_class_1 = left_size - left_class_0
        
        # Count classes in right group
        right_class_0 = sum(1 for sample in self.right_child if sample[-1] == 0)
        right_class_1 = right_size - right_class_0
        
        # Calculate Gini for each group
        left_gini = 1 - ((left_class_0/left_size)**2 + (left_class_1/left_size)**2)
        right_gini = 1 - ((right_class_0/right_size)**2 + (right_class_1/right_size)**2)
        
        # Calculate weighted average
        expected_gini = (left_size/total_size) * left_gini + (right_size/total_size) * right_gini
        
        # Check result
        self.assertAlmostEqual(gini, expected_gini)
        
        # Test perfect split
        perfect_left = [[1, 2, 0], [3, 4, 0], [5, 6, 0]]
        perfect_right = [[7, 8, 1], [9, 10, 1], [11, 12, 1]]
        perfect_groups = [perfect_left, perfect_right]
        
        perfect_gini = gini_index(perfect_groups, self.classes)
        self.assertEqual(perfect_gini, 0.0)  # Perfect split should have Gini of 0
        
        # Edge case: empty group
        empty_groups = [[], self.right_child]
        empty_gini = gini_index(empty_groups, self.classes)
        self.assertEqual(empty_gini, 0.0)  # Edge case, should handle gracefully

    # ===== Tests for information_gain =====
    def test_information_gain(self):
        """Test information gain calculation."""
        # Manual entropy calculation for parent
        left_probs = [3/3, 0/3]  # All class 0 in left child
        right_probs = [0/3, 3/3]  # All class 1 in right child
        
        # Before split entropy
        parent_entropy = entropy([3/6, 3/6])  # 3 of each class, equal probability
        
        # Calculate information gain
        gain = information_gain(parent_entropy, left_probs, right_probs)
        
        # For a perfect split, gain should equal parent entropy
        self.assertAlmostEqual(gain, 1.0)  # Binary entropy with p=0.5 is 1.0
        
        # Imperfect split
        imperfect_left = [2/3, 1/3]  # 2 class 0, 1 class 1
        imperfect_right = [1/3, 2/3]  # 1 class 0, 2 class 1
        
        imperfect_gain = information_gain(parent_entropy, imperfect_left, imperfect_right)
        
        # Calculate expected gain
        child_entropy = (3/6) * entropy(imperfect_left) + (3/6) * entropy(imperfect_right)
        expected_gain = parent_entropy - child_entropy
        
        self.assertAlmostEqual(imperfect_gain, expected_gain)
        self.assertTrue(0 < imperfect_gain < 1.0)  # Imperfect split has less gain

    # ===== Tests for mean_squared_error =====
    def test_mean_squared_error(self):
        """Test MSE calculation."""
        # Basic test cases
        mse = mean_squared_error(self.y_true_binary, self.y_pred_binary)
        
        # Calculate expected MSE
        expected = sum((t - p)**2 for t, p in zip(self.y_true_binary, self.y_pred_binary)) / len(self.y_true_binary)
        self.assertAlmostEqual(mse, expected)
        
        # Test with floating point predictions
        mse_prob = mean_squared_error(self.y_true_binary, self.y_pred_probs)
        expected_prob = sum((t - p)**2 for t, p in zip(self.y_true_binary, self.y_pred_probs)) / len(self.y_true_binary)
        self.assertAlmostEqual(mse_prob, expected_prob)
        
        # Edge cases
        perfect_mse = mean_squared_error([1, 0, 1], [1, 0, 1])  # Perfect predictions
        self.assertEqual(perfect_mse, 0.0)
        
        worst_mse = mean_squared_error([0, 0, 0], [1, 1, 1])  # Worst predictions
        self.assertEqual(worst_mse, 1.0)
        
        with self.assertRaises(Exception):
            mean_squared_error([], [])  # Empty lists
        
        with self.assertRaises(Exception):
            mean_squared_error([1, 2], [1])  # Different lengths

    # ===== Tests for cross_entropy_loss =====
    def test_cross_entropy_loss(self):
        """Test cross-entropy loss calculation."""
        # Basic test case with one-hot encoded data
        ce_loss = cross_entropy_loss(self.y_true_onehot, self.y_pred_onehot)
        
        # Calculate expected loss manually
        eps = 1e-15  # Same as in the function
        clipped_preds = np.clip(self.y_pred_onehot, eps, 1 - eps)
        expected_loss = -np.sum(self.y_true_onehot * np.log(clipped_preds)) / len(self.y_true_onehot)
        
        self.assertAlmostEqual(ce_loss, expected_loss)
        
        # Test with perfect predictions
        perfect_preds = np.array([
            [0.999, 0.001],
            [0.001, 0.999],
            [0.999, 0.001],
            [0.001, 0.999],
            [0.999, 0.001]
        ])
        perfect_loss = cross_entropy_loss(self.y_true_onehot, perfect_preds)
        self.assertTrue(perfect_loss < 0.1)  # Should be close to 0
        
        # Test with worst predictions
        worst_preds = np.array([
            [0.001, 0.999],
            [0.999, 0.001],
            [0.001, 0.999],
            [0.999, 0.001],
            [0.001, 0.999]
        ])
        worst_loss = cross_entropy_loss(self.y_true_onehot, worst_preds)
        self.assertTrue(worst_loss > 5)  # Should be large
        
        # Edge cases
        with self.assertRaises(Exception):
            cross_entropy_loss(np.array([]), np.array([]))  # Empty arrays

if __name__ == '__main__':
    unittest.main()