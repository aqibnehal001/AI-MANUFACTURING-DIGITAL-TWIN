import pandas as pd

file_path = "data/milling/milling_features.csv"

data = pd.read_csv(file_path, sep=";", header=1)

# Remove target columns from the input features
features = data.drop(
    columns=["CycleToFailure", "CycleToFailureNormalized"]
)

print("========== FEATURE ANALYSIS ==========")

print("\nTotal input features:", features.shape[1])

print("\n========== NUMERIC FEATURES ==========")

numeric_features = features.select_dtypes(include="number")

for column in numeric_features.columns:
    print("-", column)

print("\nTotal numeric features:", numeric_features.shape[1])

print("\n========== NON-NUMERIC FEATURES ==========")

non_numeric_features = features.select_dtypes(exclude="number")

for column in non_numeric_features.columns:
    print("-", column)

print("\nTotal non-numeric features:", non_numeric_features.shape[1])
print("\n========== NON-NUMERIC VALUE CHECK ==========")

for column in non_numeric_features.columns:
    print("\n", column)
    print(data[column].head(20).to_list())
    print("Unique values:", data[column].nunique())