import joblib

# Load trained model
model = joblib.load("models/sentiment_model.pkl")

while True:
    review = input("\nEnter Review (or type 'quit'): ")

    if review.lower() == "quit":
        break

    prediction = model.predict([review])[0]



    print(f"Prediction: {prediction}")