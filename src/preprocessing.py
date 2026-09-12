import pandas as pd

# Load dataset
file_path = "data/milling/milling_features.csv"

data = pd.read_csv(
    file_path,
    sep=";",
    header=1
)

print("Original shape:", data.shape)

# --------------------------------------------------
# 1. Preserve tool identity for train/test splitting
# --------------------------------------------------

data["ToolID"] = data["TollIndex"]

# FileName and TollIndex are identifiers/grouping variables,
# not ML prediction features
data = data.drop(columns=["FileName", "TollIndex"])
# --------------------------------------------------
# 2. Convert RDOC to numeric
# --------------------------------------------------

data["RDOC"] = pd.to_numeric(
    data["RDOC"],
    errors="coerce"
)

# --------------------------------------------------
# 3. Convert HardnessMean to numeric
#    Dataset uses comma as decimal separator
# --------------------------------------------------

data["HardnessMean"] = (
    data["HardnessMean"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

data["HardnessMean"] = pd.to_numeric(
    data["HardnessMean"],
    errors="coerce"
)

# --------------------------------------------------
# 4. Convert normalized target to numeric
# --------------------------------------------------

data["CycleToFailureNormalized"] = (
    data["CycleToFailureNormalized"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

data["CycleToFailureNormalized"] = pd.to_numeric(
    data["CycleToFailureNormalized"],
    errors="coerce"
)

# --------------------------------------------------
# 5. Check missing values
# --------------------------------------------------

print("\nMissing values after conversion:")

missing = data.isnull().sum()

print(missing[missing > 0])

# --------------------------------------------------
# 6. Display final information
# --------------------------------------------------

print("\nFinal shape:", data.shape)

print("\nData types:")

print(data.dtypes)

# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

output_path = "data/milling/milling_cleaned.csv"

data.to_csv(
    output_path,
    index=False
)

print("\nCleaned dataset saved successfully!")
print(output_path)