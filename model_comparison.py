import json

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

print("===== Model Comparison =====")

print(
    f"Current Model Accuracy   : "
    f"{current_accuracy:.4f}"
)

print(
    f"Candidate Model Accuracy : "
    f"{candidate_accuracy:.4f}"
)

if candidate_accuracy > current_accuracy:

    print("\nDecision: PROMOTE MODEL")

else:

    print("\nDecision: KEEP CURRENT MODEL")