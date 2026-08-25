import json
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("Loading retraining dataset...")

df = pd.read_csv(
    "external_data/combined_dataset.csv"
)

print(f"Dataset Shape: {df.shape}")

X = df["review"]
y = df["sentiment"]

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Building candidate model...")

candidate_model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(max_features=5000)
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

mlflow.set_experiment(
    "Candidate Retraining"
)

print("\nTraining candidate model...")

with mlflow.start_run():

    candidate_model.fit(
        X_train,
        y_train
    )

    predictions = candidate_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    mlflow.log_param(
        "dataset",
        "combined_dataset"
    )

    mlflow.log_param(
        "high_confidence_threshold",
        0.90
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.sklearn.log_model(
        sk_model=candidate_model,
        artifact_path="candidate_model"
    )

    print(
        f"\nCandidate Accuracy: {accuracy:.4f}"
    )


metrics = {
    "model_name": "candidate_model",
    "accuracy": float(accuracy)
}

with open(
    "models/candidate_metrics.json",
    "w"
) as file:
    json.dump(
        metrics,
        file,
        indent=4
    )

print(
    "Candidate model metrics saved!"
)
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

joblib.dump(
    candidate_model,
    "models/candidate_model.pkl"
)

print(
    "\nCandidate model saved!"
)

print(
    "Location: models/candidate_model.pkl"
)