import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = {
    "Experience": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ],
    "Salary": [
        35000, 38000, 42000, 45000, 48000,
        52000, 55000, 59000, 62000, 67000,
        70000, 74000, 78000, 82000, 86000,
        90000, 95000, 100000, 108000, 115000
    ]
}

df = pd.DataFrame(data)

X = df[["Experience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = RandomForestRegressor(
    	n_estimators=100,
	max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
train_predictions = model.predict(X_train)

train_mae = mean_absolute_error(y_train, train_predictions)

print("\nTraining MAE:", train_mae)
print("Testing MAE:", mae)

print("Actual salaries:")
print(y_test.values)

print("\nPredicted salaries:")
print(predictions)

print("\nMean Absolute Error:", mae)