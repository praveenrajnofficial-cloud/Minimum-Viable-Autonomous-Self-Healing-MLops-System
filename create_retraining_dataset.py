import pandas as pd

print("Loading datasets...")

# Original labeled dataset
imdb_df = pd.read_csv(
    "data/IMDB Dataset.csv"
)

# TMDB reviews with predictions and confidence
tmdb_df = pd.read_csv(
    "external_data/tmdb_predictions.csv"
)

# Keep only high-confidence predictions
tmdb_df = tmdb_df[
    tmdb_df["confidence"] >= 0.90
]

print(
    f"High Confidence TMDB Records: {len(tmdb_df)}"
)

# Keep only required columns
tmdb_df = tmdb_df[
    ["review", "prediction"]
]

# Rename prediction -> sentiment
tmdb_df = tmdb_df.rename(
    columns={
        "prediction": "sentiment"
    }
)

print(
    f"IMDB Records: {len(imdb_df)}"
)

# Merge datasets
combined_df = pd.concat(
    [imdb_df, tmdb_df],
    ignore_index=True
)

print(
    f"Combined Records: {len(combined_df)}"
)

# Save retraining dataset
combined_df.to_csv(
    "external_data/combined_dataset.csv",
    index=False
)

print("\nRetraining dataset created!")
print("Saved to external_data/combined_dataset.csv")