import pandas as pd

print("Loading datasets...")

reference_df = pd.read_csv("reference_data.csv")
current_df = pd.read_csv("external_data/tmdb_reviews.csv")

reference_count = len(reference_df)
current_count = len(current_df)

print("\n===== Drift Report =====")

print(f"Reference Reviews : {reference_count}")
print(f"Current Reviews   : {current_count}")

if current_count < 10:
    print("\nStatus: NOT ENOUGH DATA")

else:

    reference_avg_length = reference_df["review"].str.len().mean()
    current_avg_length = current_df["review"].str.len().mean()

    print(f"\nReference Avg Length : {reference_avg_length:.2f}")
    print(f"Current Avg Length   : {current_avg_length:.2f}")

    difference = abs(reference_avg_length - current_avg_length)

    print(f"Difference           : {difference:.2f}")

    if difference > 50:
        print("\nStatus: DRIFT DETECTED")
    else:
        print("\nStatus: NO DRIFT")