from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explainability import load_champion, prepare_customer_input

app = FastAPI(title="Churn Predictor API", version="1.0")

model, feature_names = load_champion()

class CustomerInput(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 65.0
    TotalCharges: float = 780.0

@app.get("/")
def root():
    return {"status": "ok", "model": "LightGBM Champion", "version": "1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(customer: CustomerInput):
    import pandas as pd
    input_dict = customer.dict()
    customer_df = prepare_customer_input(input_dict, feature_names)
    prob = model.predict_proba(customer_df)[0][1]
    risk_tier = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    return {
        "churn_probability": round(float(prob), 4),
        "risk_tier": risk_tier,
        "model": "LightGBM_Champion"
    }