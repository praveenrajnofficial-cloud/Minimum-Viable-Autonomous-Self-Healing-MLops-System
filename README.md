# 🚀 Minimum Viable Autonomous Self-Healing MLOps System

An end-to-end **MLOps pipeline for sentiment analysis** that can monitor incoming data, detect data drift, collect new external data, retrain a candidate model, compare it with the current production model, and automatically promote the candidate only when it performs better.

The project is designed to demonstrate how a machine learning model can be **monitored, retrained, evaluated, and deployed with minimal manual intervention**.

---

## 📌 Project Overview

Traditional machine learning models can become less effective after deployment because real-world data changes over time.

This project addresses that problem by building an automated self-healing pipeline:

```text
IMDB Dataset
     ↓
Train Initial Model
     ↓
Production Model
     ↓
FastAPI Deployment
     ↓
Collect New TMDB Reviews
     ↓
Predict New Data
     ↓
Detect Data Drift
     ↓
Check for New Data
     ↓
Create Retraining Dataset
     ↓
Train Candidate Model
     ↓
Compare Candidate vs Current Model
     ↓
Promote Only If Candidate Is Better
```

---

## 🎯 Objectives

* Build a sentiment analysis model
* Deploy the model through a REST API
* Collect new external movie reviews
* Monitor incoming data
* Detect data drift
* Use high-confidence pseudo-labels
* Automatically create a retraining dataset
* Train a candidate model
* Compare candidate and production models
* Automatically promote a better model
* Track experiments using MLflow
* Containerize the application using Docker

---

## 🧠 Machine Learning Model

The project uses the **IMDB Movie Review Dataset** for initial training.

### Model Pipeline

```text
Movie Review
     ↓
Text Processing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Positive / Negative
```

The trained model is saved as:

```text
models/sentiment_model.pkl
```

---

## 🌐 FastAPI

FastAPI is used to expose the trained ML model as an API.

The API acts as a communication layer between users and the trained model.

```text
User
 ↓
FastAPI
 ↓
Trained ML Model
 ↓
Prediction
 ↓
FastAPI Response
 ↓
User
```

### Start the API locally

```bash
uvicorn src.app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Example

Input:

```text
This movie was amazing. I loved it.
```

Output:

```json
{
  "review": "This movie was amazing. I loved it.",
  "prediction": "positive"
}
```

Predictions are also logged for monitoring.

---

## 📊 Prediction Logging

Predictions are stored in:

```text
logs/predictions.csv
```

The log contains information such as:

* Timestamp
* Review
* Prediction

This provides a record of model usage and incoming production data.

---

## 🎬 TMDB External Data Collection

The project uses TMDB as a source of new movie reviews.

The file:

```text
src/collect_reviews.py
```

collects new reviews and saves them to:

```text
external_data/tmdb_reviews.csv
```

The collected data includes:

```text
movie_id
review
collection_time
```

The collection timestamp helps identify when external data was collected.

---

## 🤖 Pseudo Labeling

The TMDB reviews do not contain the sentiment labels required for retraining.

Therefore, the existing production model is used to generate predictions for the new reviews.

Example:

```text
New TMDB Review
      ↓
Current Model
      ↓
Positive / Negative
      ↓
Confidence Score
```

Only sufficiently high-confidence predictions are selected for retraining.

This reduces the risk of adding unreliable pseudo-labels to the training dataset.

---

## 🔍 Data Drift Detection

The file:

```text
src/drift_detector_v2.py
```

checks whether incoming data has changed compared with the reference data.

The project monitors characteristics such as:

* Review length
* Positive/negative distribution

The system produces statuses such as:

```text
NO_DRIFT
MODERATE_DRIFT
STRONG_DRIFT
```

The result is stored in:

```text
models/drift_status.json
```

---

## 🆕 New Data Detection

Drift alone should not cause the system to repeatedly retrain on exactly the same data.

The project therefore checks whether genuinely new external data is available.

The file:

```text
src/check_new_data.py
```

prevents unnecessary retraining when there is no new data.

Conceptually:

```text
Drift Detected
      ↓
Is New Data Available?
      ↓
 ┌────┴────┐
No         Yes
↓           ↓
Skip       Retrain
```

---

## 🔄 Retraining Dataset

The file:

```text
src/create_retraining_dataset.py
```

combines:

```text
Original IMDB Dataset
        +
High-confidence TMDB Data
        ↓
Combined Dataset
```

The resulting dataset is saved as:

```text
external_data/combined_dataset.csv
```

---

## 🧪 Candidate Model Training

The file:

```text
src/retrain_candidate.py
```

trains a new model using the updated retraining dataset.

The new model is called the **candidate model** because it is not immediately deployed.

Outputs include:

```text
models/candidate_model.pkl
models/candidate_metrics.json
```

---

## ⚖️ Model Comparison

The candidate model must prove that it is better than the current production model.

The file:

```text
src/model_comparison.py
```

compares:

```text
Current Production Model
          VS
Candidate Model
```

Example:

```text
Current Model Accuracy   : 0.8950
Candidate Model Accuracy : 0.8903
```

Decision:

```text
KEEP CURRENT MODEL
```

If the candidate performs better:

```text
PROMOTE CANDIDATE MODEL
```

---

## 🚀 Automatic Model Promotion

The file:

```text
src/promote_model.py
```

handles model promotion.

The process is:

```text
Candidate Better?
      ↓
     YES
      ↓
Promote Candidate
      ↓
New Production Model
```

If the candidate is worse:

```text
Candidate Worse
      ↓
Keep Current Production Model
```

This prevents an automatically retrained model from replacing a better production model.

---

## 🩹 Self-Healing Pipeline

The main automation file is:

```text
src/self_healing_pipeline.py
```

Run the complete pipeline with:

```bash
python src/self_healing_pipeline.py
```

The pipeline automatically coordinates the major stages:

```text
1. Collect TMDB Reviews
             ↓
2. Predict New Reviews
             ↓
3. Detect Drift
             ↓
4. Check New Data
             ↓
5. Create Retraining Dataset
             ↓
6. Train Candidate Model
             ↓
7. Compare Models
             ↓
8. Promote Better Model
```

If there is no meaningful new data, retraining can be skipped.

---

## 📈 MLflow

MLflow is used for experiment tracking.

It helps record information such as:

* Experiments
* Model runs
* Accuracy
* Parameters
* Model artifacts

Start MLflow:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 🐳 Docker

Docker is used to package the application and its runtime environment.

### Build the image

```bash
docker build -t self-healing-mlops .
```

### Run the container

```bash
docker run -p 8000:8000 self-healing-mlops
```

Then open:

```text
http://localhost:8000/docs
```

Docker allows the FastAPI application and its dependencies to run in a consistent environment.

---

## 📁 Project Structure

```text
Minimum-Viable-Autonomous-Self-Healing-MLOps-System/
│
├── data/
│   └── IMDB Dataset.csv
│
├── external_data/
│   ├── tmdb_reviews.csv
│   ├── tmdb_predictions.csv
│   └── combined_dataset.csv
│
├── logs/
│   └── predictions.csv
│
├── models/
│   ├── sentiment_model.pkl
│   ├── candidate_model.pkl
│   ├── current_metrics.json
│   ├── candidate_metrics.json
│   ├── drift_status.json
│   └── data_signature.json
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── collect_reviews.py
│   ├── tmdb_predict.py
│   ├── create_reference.py
│   ├── drift_detector.py
│   ├── drift_detector_v2.py
│   ├── check_new_data.py
│   ├── create_retraining_dataset.py
│   ├── retrain_candidate.py
│   ├── model_comparison.py
│   ├── promote_model.py
│   ├── self_healing_pipeline.py
│   └── data_simulator.py
│
├── mlruns/
│
├── reference_data.csv
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## 🧪 Testing the Project

### Test the API

```bash
uvicorn src.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Test TMDB collection

```bash
python src/collect_reviews.py
```

### Test prediction generation

```bash
python src/tmdb_predict.py
```

### Test drift detection

```bash
python src/drift_detector_v2.py
```

### Test retraining dataset

```bash
python src/create_retraining_dataset.py
```

### Test candidate training

```bash
python src/retrain_candidate.py
```

### Test model comparison

```bash
python src/model_comparison.py
```

### Test complete self-healing pipeline

```bash
python src/self_healing_pipeline.py
```

---

## 🔁 Complete System Flow

```text
                 IMDB Dataset
                      │
                      ▼
                Initial Training
                      │
                      ▼
              Production Model
                      │
                      ▼
                 FastAPI API
                      │
                      ▼
                User Reviews
                      │
                      ▼
              Prediction Logging
                      │
                      │
                      ▼
               TMDB New Reviews
                      │
                      ▼
                Current Model
                      │
                      ▼
                Pseudo Labels
                      │
                      ▼
                Drift Detection
                      │
                      ▼
               New Data Check
                      │
             ┌────────┴────────┐
             │                 │
          No New Data       New Data
             │                 │
             ▼                 ▼
       Skip Retraining    Retrain Candidate
                               │
                               ▼
                         Model Comparison
                               │
                     ┌─────────┴─────────┐
                     │                   │
              Candidate Better     Candidate Worse
                     │                   │
                     ▼                   ▼
                Promote Model      Keep Current
```

---

## 💡 Why Is It Called "Self-Healing"?

The system is called self-healing because it can respond to changes in its data environment without requiring the developer to manually perform every step.

Instead of:

```text
Detect Problem
     ↓
Developer Retrains
     ↓
Developer Tests
     ↓
Developer Deploys
```

the system automates:

```text
Detect Drift
     ↓
Check New Data
     ↓
Retrain
     ↓
Evaluate
     ↓
Promote if Better
```

The model is also protected from automatic degradation because a candidate model is promoted only after comparison with the current model.

---

## 🛠️ Technologies Used

| Technology          | Purpose                    |
| ------------------- | -------------------------- |
| Python              | Main programming language  |
| Scikit-Learn        | Machine learning           |
| Pandas              | Data processing            |
| TF-IDF              | Text feature extraction    |
| Logistic Regression | Sentiment classification   |
| FastAPI             | ML model API               |
| TMDB API            | External movie review data |
| MLflow              | Experiment tracking        |
| Docker              | Containerization           |
| Git                 | Version control            |

---

## 🎯 Key MLOps Concepts Demonstrated

* Model serving
* API deployment
* Data collection
* Data drift detection
* Data monitoring
* Pseudo labeling
* Automated retraining
* Candidate model evaluation
* Model promotion
* Experiment tracking
* Containerization
* Automation

---

## 🚀 Future Improvements

Possible future improvements include:

* Use a more robust statistical drift detection method
* Add model performance monitoring with ground-truth feedback
* Add CI/CD automation
* Add automated scheduled pipeline execution
* Deploy to a cloud platform
* Add model versioning and rollback
* Add authentication to the API
* Add automated monitoring dashboards
* Use Kubernetes for scalable deployment

---

## 👨‍💻 Author

**Praveen Raj N**

B.Tech — Artificial Intelligence and Data Science

**Project:** Minimum Viable Autonomous Self-Healing MLOps System
