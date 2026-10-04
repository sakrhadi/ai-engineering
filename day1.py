import numpy as np
import pandas as pd

# Create some employee data
data = {
    "Name": ["Ali", "John", "Sarah", "Mike", "Rami"],
    "Age": [25, 32, 28, 41, 35],
    "Experience": [2, 8, 5, 15, 10],
    "Salary": [50000, 75000, 62000, 110000, 90000]
}

df = pd.DataFrame(data)

print("Employee Data:")
print(df)

print("\nAverage Age:", df["Age"].mean())
print("Average Experience:", df["Experience"].mean())
print("Average Salary:", df["Salary"].mean())