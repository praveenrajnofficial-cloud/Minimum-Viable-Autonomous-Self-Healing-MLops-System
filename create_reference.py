import pandas as pd

print("Loading training dataset...")

df = pd.read_csv("data/IMDB Dataset.csv")

# Save first 1000 reviews as reference data
reference_df = df[["review"]].head(1000)

reference_df.to_csv(
    "reference_data.csv",
    index=False
)

print("Reference data created successfully!")
print(f"Total rows saved: {len(reference_df)}")