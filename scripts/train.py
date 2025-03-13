from diabetes_classifier import DiabetesClassifier, load_data, preprocess_data

DATA_PATH = "data/processed/diabetes_cleaned.csv"

# Load and preprocess data
df = load_data(DATA_PATH)
df_cleaned = preprocess_data(df)

X = df_cleaned.drop(columns=["Outcome"])
y = df_cleaned["Outcome"]

# Train the model
model = DiabetesClassifier()
model.train(X, y)
print("Model training complete.")
