import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("customers.csv")

# Separate features from the target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())