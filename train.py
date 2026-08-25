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

# Create MLflow experiment
mlflow.set_experiment("Sentiment Analysis")

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/IMDB Dataset.csv")

print(f"Dataset Shape: {df.shape}")

# Features (reviews)
X = df["review"]

# Target (positive/negative)
y = df["sentiment"]

print("\nSplitting dataset...")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Building model pipeline...")

# TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training model...")

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sentiment_model"
    )

    print("\n==========================")
    print(f"Accuracy: {accuracy:.4f}")
    print("==========================")


metrics = {
    "model_name": "current_model",
    "accuracy": float(accuracy)
}

with open(
    "models/current_metrics.json",
    "w"
) as file:
    json.dump(
        metrics,
        file,
        indent=4
    )

print(
    "Current model metrics saved!"
)
# Print detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model locally
joblib.dump(model, "models/sentiment_model.pkl")

print("\nModel saved successfully!")
print("Location: models/sentiment_model.pkl")