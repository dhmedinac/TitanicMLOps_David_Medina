# MLOps Titanic Survival Prediction 2026 and month Feb
# test owner modification 11 12 13


A clean, end-to-end **MLOps project** that trains a **Random Forest** model on the **public Titanic dataset** and serves predictions through a **Dockerized FastAPI service**, with **Poetry** for dependency management.

This project is designed to be **simple, reproducible, and interview-ready**, while following real-world MLOps best practices.

---

## Project Overview

**Goal:** Predict passenger survival on the Titanic using a supervised machine learning model.

**What this project demonstrates:**
- Use of a **public dataset** (Titanic)
- Clear separation between **training** and **inference**
- **Reproducible dependencies** using Poetry
- **Reproducible runtime** using Docker
- Model served via a **REST API**

---

## Tech Stack

- Python 3.10+
- scikit-learn (Random Forest)
- Pandas / NumPy
- FastAPI + Uvicorn
- Poetry (dependency management)
- Docker & Docker Compose

---

## Project Structure

```
mlops-titanic-rf/
│
├── pyproject.toml          # Poetry dependencies
├── poetry.lock             # Locked dependency versions
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Local container orchestration
├── README.md               # Project documentation
│
├── data/
│   └── titanic.csv         # Public Titanic dataset
│
├── models/
│   └── model.pkl           # Trained Random Forest model
│
├── src/
│   ├── preprocess.py       # Feature engineering
│   ├── train.py            # Model training script
│   └── predict.py          # Inference logic
│
├── app/
│   └── main.py             # FastAPI application
```

---

## Dataset

- Source: **Kaggle Titanic Dataset** (`train.csv`)
- Target variable: `Survived`

**Features used:**
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare

---

## Setup Instructions

### 1. Prerequisites

Install locally:
- Python 3.10+
- Docker Desktop
- Git
- Poetry

```bash
pip install poetry
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/dhmedinac/CapgeminiMLOps_David_Medina.git
cd CapgeminiMLOps_David_Medina
```

---

### 3. Install Dependencies (Poetry)

```bash
poetry install
```

Dependencies are installed into a **Poetry-managed virtual environment**.

---

## Model Training

Train the model locally using Poetry:

```bash
poetry run python src/train.py
```

This will:
- Load the Titanic dataset
- Train a Random Forest classifier
- Evaluate accuracy
- Save the model to:

```
models/model.pkl
```

---

## Running the API (Docker)

Build and start the inference service:

```bash
docker compose build
docker compose up
```

The API will be available at:

```
http://localhost:8000/docs
```

---

## API Usage

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "ok"
}
```

---

### Prediction Endpoint

```
POST /predict
```

Example request body:

```json
{
  "pclass": 3,
  "sex": "male",
  "age": 22,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25
}
```

Example response:

```json
{
  "survived": 0,
  "probability": 0.18
}
```

---

## MLOps Design Decisions

- **Training outside the container**: The model is trained locally and saved as a versioned artifact.
- **Inference-only container**: Docker image only loads the trained model and serves predictions.
- **Poetry + Docker**: Poetry handles dependency locking, Docker ensures runtime reproducibility.

This mirrors common production MLOps patterns.

---

## Reproducibility

- Exact dependency versions are locked in `poetry.lock`
- Runtime environment is defined in the Dockerfile
- The same model artifact is used across environments

---

## Possible Extensions

- Add MLflow for experiment tracking
- Add automated tests (pytest)
- Add CI/CD pipeline
- Deploy to cloud (AWS/GCP/Azure)

---

## One-Sentence Summary

> This project demonstrates a reproducible local MLOps pipeline where a Random Forest model trained on the public Titanic dataset is served via a Dockerized FastAPI inference service using Poetry for dependency management.

