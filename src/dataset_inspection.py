import pandas as pd

file_path = "data/milling/milling_features.csv"

data = pd.read_csv(file_path, sep=";", header=1)

print("========== DATASET OVERVIEW ==========")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\n========== COLUMN NAMES ==========")
for i, column in enumerate(data.columns, start=1):
    print(i, "-", column)

print("\n========== DATA TYPES ==========")
print(data.dtypes)

print("\n========== MISSING VALUES ==========")
missing = data.isnull().sum()
print(missing[missing > 0])

print("\n========== TARGET CANDIDATES ==========")
for column in data.columns:
    if "wear" in column.lower() or "failure" in column.lower() or "life" in column.lower():
        print("-", column)