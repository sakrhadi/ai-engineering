import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = {
    "Age": [25, 30, 35, 40, 28, 45, 32, 38, 50, 55],
    "Experience": [2, 5, 10, 15, 4, 20, 7, 12, 25, 30],
    "Education": [
        "Bachelor", "Master", "Master", "Bachelor", "Bachelor",
        "PhD", "Master", "PhD", "Master", "PhD"
    ],
    "Salary": [
        35000, 50000, 70000, 85000, 45000,
        120000, 60000, 95000, 130000, 145000
    ]
}
df = pd.DataFrame(data)

df["Years_Not_Working"] = (
    df["Age"] - df["Experience"] - 18
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X = df[
    ["Age", "Experience", "Years_Not_Working", "Education"]
]

y = df["Salary"]

numeric_features = [
    "Age",
    "Experience",
    "Years_Not_Working"
]

categorical_features = [
    "Education"
]

preprocessor = ColumnTransformer([
    (
        "numeric",
        StandardScaler(),
        numeric_features
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
])

from sklearn.ensemble import RandomForestRegressor

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

print("Actual salaries:")
print(y_test.values)

print("\nPredicted salaries:")
print(predictions)

print("\nMean Absolute Error:", mae)

new_employee = pd.DataFrame([{
    "Age": 37,
    "Experience": 10,
    "Years_Not_Working": 9,
    "Education": "Master"
}])

prediction = pipeline.predict(new_employee)

print("\nPredicted salary:", prediction[0])