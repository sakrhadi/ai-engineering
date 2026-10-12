import pandas as pd

data = {
    "Age": [22, 35, 45, 28, 52, 41, 30, 60, 25, 48],
    "Monthly_Charges": [30, 70, 90, 45, 110, 85, 55, 120, 40, 95],
    "Tenure_Months": [2, 24, 36, 8, 48, 30, 12, 60, 4, 40],
    "Support_Calls": [5, 1, 0, 4, 0, 2, 3, 0, 6, 1],
    "Churn": [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("Customer Churn Dataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)

print("\nChurn Distribution:")
print(df["Churn"].value_counts())
x = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures (x):")
print(x)

print ("\nTarget (y):")
print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(x_train))
print("Testing samples:", len(x_test))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nActual:", y_test.values)
print("Predicted:", predictions)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)

cm = confusion_matrix(y_test, predictions)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:")
print(cm)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    x,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Scores:")
print(scores)

print("\nAverage Cross-Validation Accuracy:")
print(scores.mean())

new_customer = pd.DataFrame([{
    "Age": 45,
    "Monthly_Charges": 95,
    "Tenure_Months": 8,
    "Support_Calls": 5
}])

prediction = model.predict(new_customer)

probability = model.predict_proba(new_customer)

churn_probability = probability[0][1]

if churn_probability >= 0.5:
    print("Risk Level: High")
else:
    print("Risk Level: Low")

print("\nChurn Probability:", probability[0][1])

print("\nNew Customer Churn Prediction:", prediction[0])

import joblib

joblib.dump(model, "churn_model.pkl")

print("\nModel saved successfully!")