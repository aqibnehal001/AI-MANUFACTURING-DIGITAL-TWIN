import pandas as pd

# Load cleaned dataset
file_path = "data/milling/milling_cleaned.csv"

data = pd.read_csv(file_path)

target = "CycleToFailure"

# Remove columns that should not be ML inputs
X = data.drop(
    columns=[
        target,
        "CycleToFailureNormalized",
        "ToolID"
    ],
    errors="ignore"
)

y = data[target]

print("========== FEATURE SELECTION ==========")

print("\nInitial number of features:", X.shape[1])

# Keep only numeric columns
X = X.select_dtypes(include="number")

print("Numeric features:", X.shape[1])

# --------------------------------------------------
# Remove constant features
# --------------------------------------------------

constant_features = [
    column for column in X.columns
    if X[column].nunique() <= 1
]

X = X.drop(columns=constant_features)

print("\nConstant features removed:", len(constant_features))

# --------------------------------------------------
# Correlation with target
# --------------------------------------------------

correlation = X.corrwith(y)

correlation = correlation.abs().sort_values(ascending=False)

print("\n========== TOP 20 FEATURES ==========")

print(correlation.head(20))

print("\nFinal number of features:", X.shape[1])