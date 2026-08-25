import pandas as pd
import json

print("Loading datasets...")

reference_df = pd.read_csv("reference_data.csv")
tmdb_df = pd.read_csv("external_data/tmdb_predictions.csv")

print("\n===== Drift Report V2 =====")

# ----------------------------
# Length Drift
# ----------------------------

reference_avg = reference_df["review"].str.len().mean()
tmdb_avg = tmdb_df["review"].str.len().mean()

length_diff = abs(reference_avg - tmdb_avg)

print(f"\nLength Difference: {length_diff:.2f}")

# ----------------------------
# Sentiment Drift
# ----------------------------

positive_ratio = (
    (tmdb_df["prediction"] == "positive").mean()
) * 100

negative_ratio = (
    (tmdb_df["prediction"] == "negative").mean()
) * 100

print(f"\nPositive %: {positive_ratio:.2f}")
print(f"Negative %: {negative_ratio:.2f}")

# ----------------------------
# Drift Decision
# ----------------------------

drift_score = 0

if length_diff > 50:
    drift_score += 1

if positive_ratio > 80 or negative_ratio > 80:
    drift_score += 1

print(f"\nDrift Score: {drift_score}/2")

if drift_score >= 2:
    print("\nFINAL STATUS: STRONG DRIFT DETECTED")

elif drift_score == 1:
    print("\nFINAL STATUS: MODERATE DRIFT")

else:
    print("\nFINAL STATUS: NO DRIFT")
    

status = "NO_DRIFT"

if drift_score >= 2:
    status = "STRONG_DRIFT"

elif drift_score == 1:
    status = "MODERATE_DRIFT"

drift_result = {
    "drift_score": drift_score,
    "status": status
}

with open(
    "models/drift_status.json",
    "w"
) as file:
    json.dump(
        drift_result,
        file,
        indent=4
    )