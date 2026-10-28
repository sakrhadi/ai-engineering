import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("customers.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the machine learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Accuracy:", accuracy)

# Save the trained model
joblib.dump(model, "customer_churn_model.pkl")

print("Model saved successfully!")