import requests
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "Age": 25,
        "Monthly_Charges": 40,
        "Tenure_Months": 4,
        "Support_Calls": 6
    }
)

result = response.json()

if response.status_code == 200:
    print("Churn Prediction:", result["churn_prediction"])
    print("Churn Probability:", result["churn_probability"])
    print("Risk Level:", result["risk_level"])
else:
    print("API Error:", result)