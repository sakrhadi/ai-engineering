from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
app = FastAPI()

model = joblib.load("churn_model.pkl")

class Customer(BaseModel):
    Age: int = Field(gt=0, le=120)
    Monthly_Charges: float = Field(gt=0)
    Tenure_Months: int = Field(ge=0)
    Support_Calls: int = Field(ge=0)

@app.get("/")
def home():
    return {"message": "AI Churn Prediction API is running!"}

@app.get("/predict")
def predict():
    customer = pd.DataFrame([{
    "Age": 25,
    "Monthly_Charges": 40,
    "Tenure_Months": 4,
    "Support_Calls": 6
}])

    prediction = model.predict(customer)

    return {
        "churn_prediction": int(prediction[0])
    }

@app.post("/predict")
def predict_customer(customer: Customer):

    new_customer = pd.DataFrame([{
        "Age": customer.Age,
        "Monthly_Charges": customer.Monthly_Charges,
        "Tenure_Months": customer.Tenure_Months,
        "Support_Calls": customer.Support_Calls
    }])

    prediction = model.predict(new_customer)
    probability = model.predict_proba(new_customer)

    churn_probability = probability[0][1]

    if churn_probability >= 0.5:
        risk_level = "High"
    else:
        risk_level = "Low"

    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(churn_probability),
        "risk_level": risk_level
    }