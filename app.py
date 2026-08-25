import os
from datetime import datetime

from fastapi import FastAPI
import joblib

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Part of Autonomous Self-Healing MLOps System",
    version="1.0"
)

# Load trained model
model = joblib.load(
    "models/sentiment_model.pkl"
)

LOG_FILE = "logs/predictions.csv"


def log_prediction(review, prediction):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    file_exists = os.path.exists(
        LOG_FILE
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        if not file_exists:

            file.write(
                "timestamp,review,prediction\n"
            )

        file.write(
            f'"{datetime.now()}","{review}","{prediction}"\n'
        )


@app.get("/")
def home():

    return {
        "message": "Sentiment Analysis API Running"
    }


@app.get("/predict")
def predict(review: str):

    prediction = model.predict(
        [review]
    )[0]

    log_prediction(
        review,
        prediction
    )

    return {
        "review": review,
        "prediction": prediction
    }