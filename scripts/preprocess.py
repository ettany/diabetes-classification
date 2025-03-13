from diabetes_classifier import load_data, preprocess_data

RAW_DATA_PATH = "data/raw/diabetes.csv"
PROCESSED_DATA_PATH = "data/processed/diabetes_cleaned.csv"

# Load and preprocess data
df = load_data(RAW_DATA_PATH)
df_cleaned = preprocess_data(df)

# Save the processed data
df_cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
print("Preprocessing complete. Saved to", PROCESSED_DATA_PATH)
