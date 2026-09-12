import pandas as pd

file_path = "data/milling/milling_features.csv"

data = pd.read_csv(file_path, sep=";", header=1)

print("========== CYCLE TO FAILURE ==========")

target = data["CycleToFailure"]

print("\nData type:")
print(target.dtype)

print("\nMinimum:")
print(target.min())

print("\nMaximum:")
print(target.max())

print("\nMean:")
print(target.mean())

print("\nMedian:")
print(target.median())

print("\nUnique values:")
print(target.nunique())

print("\nFirst 20 values:")
print(target.head(20).to_list())

print("\n========== NORMALIZED TARGET ==========")

normalized = data["CycleToFailureNormalized"]

print("\nData type:")
print(normalized.dtype)

print("\nFirst 20 values:")
print(normalized.head(20).to_list())