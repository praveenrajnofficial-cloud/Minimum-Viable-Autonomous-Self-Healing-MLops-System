import json
import shutil

# Load metrics
with open(
    "models/current_metrics.json",
    "r"
) as file:
    current = json.load(file)

with open(
    "models/candidate_metrics.json",
    "r"
) as file:
    candidate = json.load(file)

current_accuracy = current["accuracy"]
candidate_accuracy = candidate["accuracy"]

print("===== Model Promotion =====")

print(
    f"Current Model Accuracy   : "
    f"{current_accuracy:.4f}"
)

print(
    f"Candidate Model Accuracy : "
    f"{candidate_accuracy:.4f}"
)

if candidate_accuracy > current_accuracy:

    print("\nPromoting candidate model...")

    shutil.copy(
        "models/candidate_model.pkl",
        "models/sentiment_model.pkl"
    )

    shutil.copy(
        "models/candidate_metrics.json",
        "models/current_metrics.json"
    )

    print("Model promoted successfully!")

else:

    print(
        "\nCandidate model is not better."
    )

    print(
        "Keeping current model."
    )