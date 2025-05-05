import unittest
import numpy as np
from diabetes_classifier.math_utils import (
    mean, variance, standard_deviation, euclidean_distance,
    manhattan_distance, min_max_scaling, z_score_normalization,
    sigmoid, relu, leaky_relu, softmax, dot_product,
    cosine_similarity, entropy, mean_squared_error, cross_entropy_loss
)

class TestMathUtils(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.values = [1, 2, 3, 4, 5]
        self.point1 = [1, 2, 3]
        self.point2 = [4, 5, 6]
        self.vector1 = [1, 2, 3]
        self.vector2 = [2, 3, 4]
        self.probabilities = [0.2, 0.3, 0.5]
        self.y_true = [0, 1, 0, 1]
        self.y_pred = [0.1, 0.9, 0.2, 0.8]

    def test_mean(self):
        """Test mean calculation."""
        self.assertEqual(mean(self.values), 3.0)

    def test_variance(self):
        """Test variance calculation."""
        self.assertAlmostEqual(variance(self.values), 2.0)

    def test_standard_deviation(self):
        """Test standard deviation calculation."""
        self.assertAlmostEqual(standard_deviation(self.values), np.sqrt(2.0))

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        self.assertAlmostEqual(euclidean_distance(self.point1, self.point2), np.sqrt(27))

    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        self.assertEqual(manhattan_distance(self.point1, self.point2), 9)

    def test_min_max_scaling(self):
        """Test min-max scaling."""
        scaled = min_max_scaling(self.values)
        self.assertEqual(min(scaled), 0.0)
        self.assertEqual(max(scaled), 1.0)

    def test_z_score_normalization(self):
        """Test z-score normalization."""
        normalized = z_score_normalization(self.values)
        self.assertAlmostEqual(mean(normalized), 0.0, places=10)
        self.assertAlmostEqual(variance(normalized), 1.0)

    def test_sigmoid(self):
        """Test sigmoid function."""
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertTrue(0 < sigmoid(-10) < 0.5)
        self.assertTrue(0.5 < sigmoid(10) < 1)

    def test_relu(self):
        """Test ReLU function."""
        self.assertEqual(relu(3), 3)
        self.assertEqual(relu(-3), 0)

    def test_leaky_relu(self):
        """Test Leaky ReLU function."""
        self.assertEqual(leaky_relu(3), 3)
        self.assertEqual(leaky_relu(-3, alpha=0.1), -0.3)

    def test_softmax(self):
        """Test softmax function."""
        result = softmax(self.values)
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertTrue(all(0 < x < 1 for x in result))

    def test_dot_product(self):
        """Test dot product calculation."""
        self.assertEqual(dot_product(self.vector1, self.vector2), 20)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        self.assertAlmostEqual(
            cosine_similarity(self.vector1, self.vector2),
            20 / (np.sqrt(14) * np.sqrt(29))
        )

    def test_entropy(self):
        """Test entropy calculation."""
        self.assertAlmostEqual(
            entropy(self.probabilities),
            -(0.2 * np.log2(0.2) + 0.3 * np.log2(0.3) + 0.5 * np.log2(0.5))
        )

    def test_mean_squared_error(self):
        """Test MSE calculation."""
        y_true_numeric = [0, 1, 0, 1]
        y_pred_numeric = [0.1, 0.9, 0.2, 0.8]
        expected_mse = np.mean([(a - b) ** 2 for a, b in zip(y_true_numeric, y_pred_numeric)])
        self.assertAlmostEqual(mean_squared_error(y_true_numeric, y_pred_numeric), expected_mse)

    def test_cross_entropy_loss(self):
        """Test cross-entropy loss calculation."""
        # Convert to one-hot encoding
        y_true_one_hot = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_pred_one_hot = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        
        loss = cross_entropy_loss(y_true_one_hot, y_pred_one_hot)
        self.assertTrue(loss > 0)

if __name__ == '__main__':
    unittest.main()