import pandas as pd

# Load the customer data
df = pd.read_csv("customers.csv")

# Display the first five rows
print(df.head())

# Display the dataset shape
print("\nDataset shape:")
print(df.shape)

# Display column information
print("\nDataset information:")
df.info()

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nChurn distribution:")
print(df["Churn"].value_counts())