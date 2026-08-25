import pandas as pd
import joblib

print("Loading model...")

# Load trained model
model = joblib.load(
    "models/sentiment_model.pkl"
)

print("Loading TMDB reviews...")

df = pd.read_csv(
    "external_data/tmdb_reviews.csv"
)

print(f"Total Reviews: {len(df)}")

print("Generating predictions...")

# Predict labels
predictions = model.predict(
    df["review"]
)

# Predict probabilities
probabilities = model.predict_proba(
    df["review"]
)

# Confidence = highest probability
confidence = probabilities.max(axis=1)

# Save results
df["prediction"] = predictions
df["confidence"] = confidence

output_file = (
    "external_data/tmdb_predictions.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nPrediction completed!")
print(f"Saved to: {output_file}")

print("\nPrediction Distribution:")
print(df["prediction"].value_counts())

print("\nAverage Confidence:")
print(round(df["confidence"].mean(), 4))