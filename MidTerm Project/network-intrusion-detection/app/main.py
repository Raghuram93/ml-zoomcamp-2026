from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import Dict, Any

# Resolve model path relative to project root
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "model.bin"

app = FastAPI(title="Network Intrusion Detection API")

# Load model at startup
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
dv = model_data["dv"]

class TrafficData(BaseModel):
    duration: float
    protocol_type: str
    service: str
    flag: str
    src_bytes: float
    dst_bytes: float
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

@app.get("/")
def home():
    return {"message": "Network Intrusion Detection API is running"}

@app.post("/predict")
def predict(data: TrafficData) -> Dict[str, Any]:
    record = data.model_dump()
    X = dv.transform([record])
    proba = float(model.predict_proba(X)[0, 1])
    prediction = "attack" if proba >= 0.5 else "normal"

    return {
        "prediction": prediction,
        "probability": round(proba, 4)
    }
