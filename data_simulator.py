import pandas as pd
import requests
import random
import time

# Load IMDB dataset
df = pd.read_csv("data/IMDB Dataset.csv")

print("Starting Data Simulator...")

# Number of simulated users
NUM_REQUESTS = 50

for i in range(NUM_REQUESTS):

    # Pick random review
    review = random.choice(df["review"].tolist())

    try:
        response = requests.get(
            "http://127.0.0.1:8000/predict",
            params={"review": review}
        )

        result = response.json()

        print(
            f"{i+1}/{NUM_REQUESTS} | "
            f"Prediction: {result['prediction']}"
        )

    except Exception as e:
        print(f"Error: {e}")

    # Simulate user arrival
    time.sleep(1)

print("\nSimulation Completed!")