import pandas as pd

data = {
    "Age": [25, 30, 35, 40, 28, 45, 32, 38],
    "Experience": [2, 5, 10, 15, 4, 20, 7, 12],
    "City": [
        "Beirut",
        "Beirut",
        "Sidon",
        "Tripoli",
        "Sidon",
        "Beirut",
        "Tripoli",
        "Sidon"
    ],
    "Education": [
        "Bachelor",
        "Master",
        "Master",
        "Bachelor",
        "Bachelor",
        "PhD",
        "Master",
        "PhD"
    ],
    "Salary": [35000, 50000, 70000, 85000, 45000, 120000, 60000, 95000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

print("\nData Types:")
print(df.dtypes)
df_encoded = pd.get_dummies(
    df,
    columns=["City", "Education"]
)

X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nActual salaries:")
print(y_test.values)

print("\nPredicted salaries:")
print(predictions)

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)

train_predictions = model.predict(X_train)

train_mae = mean_absolute_error(y_train, train_predictions)

print("\nTraining Mean Absolute Error:", train_mae)
print("Testing Mean Absolute Error:", mae)