import pandas as pd

file_path = "data/milling/milling_features.csv"

data = pd.read_csv(file_path, sep=";", header=1)

print("Dataset loaded successfully!")
print("Shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())