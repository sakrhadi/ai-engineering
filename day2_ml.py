import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = {
    "Age": [25, 28, 30, 32, 35, 38, 41, 45, 48, 52],
    "Experience": [1, 3, 5, 7, 10, 12, 15, 18, 21, 25],
    "Education": [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    "Salary": [35000, 42000, 50000, 58000, 70000,
               82000, 95000, 110000, 125000, 140000]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

X = df[["Age", "Experience", "Education"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
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

new_employee = [[37, 15, 3]]

prediction = model.predict(new_employee)

print("\nPredicted salary:")
print(prediction[0])

print("\nModel coefficients:")
print(model.coef_)

print("\nModel intercept:")
print(model.intercept_)