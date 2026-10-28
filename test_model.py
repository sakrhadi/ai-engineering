import pandas as pd
import joblib

# Load the trained model
model = joblib.load("customer_churn_model.pkl")

# Create a new customer
customer = pd.DataFrame([{
    "Age": 30,
    "Monthly_Charges": 85,
    "Tenure_Months": 5,
    "Support_Calls": 6
}])

# Predict churn
prediction = model.predict(customer)[0]
probability = model.predict_proba(customer)[0][1]

print("Churn prediction:", prediction)
print("Churn probability:", round(probability, 4))
print("Risk level:", "High" if probability >= 0.5 else "Low")