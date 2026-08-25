import subprocess
import json

print(
    "\n===== SELF-HEALING PIPELINE =====\n"
)

# Step 1 - Collect Reviews
subprocess.run(
    "python src/collect_reviews.py",
    shell=True,
    check=True
)

# Step 2 - Predict Reviews
subprocess.run(
    "python src/tmdb_predict.py",
    shell=True,
    check=True
)

# Step 3 - Detect Drift
subprocess.run(
    "python src/drift_detector_v2.py",
    shell=True,
    check=True
)

# Read Drift Status
with open(
    "models/drift_status.json",
    "r"
) as file:

    drift = json.load(file)

status = drift["status"]

print(
    f"\nDrift Status: {status}"
)

# Stop if no drift
if status == "NO_DRIFT":

    print(
        "\nNo drift detected."
    )

    print(
        "Retraining skipped."
    )

    exit()

# Step 4 - Check New Data
result = subprocess.run(
    "python src/check_new_data.py",
    shell=True,
    capture_output=True,
    text=True
)

print(result.stdout)

if "NO_NEW_DATA" in result.stdout:

    print(
        "\nNo new external data detected."
    )

    print(
        "Retraining skipped."
    )

    exit()

# Continue only if drift + new data
print(
    "\nRetraining pipeline started..."
)

# Step 5 - Create Retraining Dataset
subprocess.run(
    "python src/create_retraining_dataset.py",
    shell=True,
    check=True
)

# Step 6 - Train Candidate Model
subprocess.run(
    "python src/retrain_candidate.py",
    shell=True,
    check=True
)

# Step 7 - Compare Models
subprocess.run(
    "python src/model_comparison.py",
    shell=True,
    check=True
)

# Step 8 - Promote Model
subprocess.run(
    "python src/promote_model.py",
    shell=True,
    check=True
)

print(
    "\n===== PIPELINE COMPLETED ====="
)