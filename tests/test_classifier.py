# tests/test_classifier.py
import unittest
from diabetes_classifier import DiabetesClassifier
from diabetes_classifier.data_processing import load_data

class TestDiabetesClassifier(unittest.TestCase):
    def setUp(self):
        """Set up any necessary data or states for testing."""
        self.model = DiabetesClassifier()
        self.df = load_data("data/processed/diabetes_cleaned.csv")
        self.X = self.df.drop(columns=["Outcome"])
        self.y = self.df["Outcome"]

    def test_model_training(self):
        """Test if the model trains without errors."""
        self.model.train(self.X, self.y)
        self.assertIsNotNone(self.model.model.coef_)

if __name__ == '__main__':
    unittest.main()
