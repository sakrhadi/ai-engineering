import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

print("Model loaded successfully!")

age = int(input("Enter customer age: "))
monthly_charges = float(input("Enter monthly charges: "))
tenure_months = int(input("Enter tenure in months: "))
support_calls = int(input("Enter number of support calls: "))

new_customer = pd.DataFrame([{
    "Age": age,
    "Monthly_Charges": monthly_charges,
    "Tenure_Months": tenure_months,
    "Support_Calls": support_calls
}])

prediction = model.predict(new_customer)

probability = model.predict_proba(new_customer)

churn_probability = probability[0][1]

print("\nChurn Prediction:", prediction[0])
print("Churn Probability:", churn_probability)

if churn_probability >= 0.5:
    print("Risk Level: High")
else:
    print("Risk Level: Low")