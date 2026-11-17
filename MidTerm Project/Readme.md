
# Network Intrusion Detection System

**ML Zoomcamp 2026 – Midterm Project**

## Project Overview

This project implements a Machine Learning based Network Intrusion Detection System.  
The goal is to classify network traffic as either **Normal** or **Attack** using the NSL-KDD dataset.

The complete pipeline includes:

- Data downloading and preprocessing
- Exploratory Data Analysis
- Training and comparison of multiple models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
- Model selection based on performance metrics
- Deployment of the best model as a FastAPI web service
- Containerization using Docker

## Dataset

- **NSL-KDD** (improved version of KDD Cup 99)
- Binary classification: `normal` vs `attack`
- Source: [NSL-KDD Dataset](https://github.com/defcom17/NSL_KDD)

## Project Structure
network-intrusion-detection(NID)/
├── data/                       # Dataset files
├── model/                      # Saved trained model
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── scripts/
│   ├── download_data.py
│   └── train.py
├── app/
│   └── main.py                 # FastAPI application
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

## How to Run the Project

### 1. Clone the repository and create virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

2. Download the datasetpowershell

python scripts/download_data.py

3. Train the modelpowershell

python scripts/train.py

This script will:Load and preprocess the data
Train multiple models
Select the best performing model
Save it as model/model.bin

4. Run the API locallypowershell

uvicorn app.main:app --reload --port 9696

Open the interactive documentation:
http://localhost:9696/docs5. Run with Dockerpowershell

docker compose up --build

The service will be available at:
http://localhost:9696/docsExample PredictionYou can test the API using the Swagger UI or with the following curl command:powershell




