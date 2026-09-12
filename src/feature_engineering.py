import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

input_file = "data/milling/milling_cleaned.csv"

data = pd.read_csv(input_file)

print("Original shape:", data.shape)


# ============================================================
# 2. IDENTIFY SENSOR COLUMNS
# ============================================================

accelerometer_cols = [
    c for c in data.columns
    if "accelerometer" in c.lower()
]

current_cols = [
    c for c in data.columns
    if "current" in c.lower()
]


print("\n========== SENSOR GROUPS ==========")

print(
    "Accelerometer columns:",
    len(accelerometer_cols)
)

print(
    "Current columns:",
    len(current_cols)
)


# ============================================================
# 3. CONVERT SENSOR DATA TO NUMERIC
# ============================================================

for col in accelerometer_cols + current_cols:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


# ============================================================
# 4. ACCELEROMETER AGGREGATE FEATURES
# ============================================================

if len(accelerometer_cols) > 0:

    accel_data = data[accelerometer_cols]

    data["Accel_Row_Mean"] = accel_data.mean(axis=1)

    data["Accel_Row_STD"] = accel_data.std(axis=1)

    data["Accel_Row_RMS"] = np.sqrt(
        (accel_data ** 2).mean(axis=1)
    )

    data["Accel_Row_Max"] = accel_data.max(axis=1)

    data["Accel_Row_Min"] = accel_data.min(axis=1)

    data["Accel_Row_Range"] = (
        data["Accel_Row_Max"]
        - data["Accel_Row_Min"]
    )


# ============================================================
# 5. CURRENT AGGREGATE FEATURES
# ============================================================

if len(current_cols) > 0:

    current_data = data[current_cols]

    data["Current_Row_Mean"] = (
        current_data.mean(axis=1)
    )

    data["Current_Row_STD"] = (
        current_data.std(axis=1)
    )

    data["Current_Row_RMS"] = np.sqrt(
        (current_data ** 2).mean(axis=1)
    )

    data["Current_Row_Max"] = (
        current_data.max(axis=1)
    )

    data["Current_Row_Min"] = (
        current_data.min(axis=1)
    )

    data["Current_Row_Range"] = (
        data["Current_Row_Max"]
        - data["Current_Row_Min"]
    )


# ============================================================
# 6. VIBRATION / CURRENT RATIO FEATURES
# ============================================================

if (
    "Accel_Row_RMS" in data.columns
    and "Current_Row_RMS" in data.columns
):

    data["Vibration_Current_Ratio"] = (
        data["Accel_Row_RMS"]
        / (data["Current_Row_RMS"].abs() + 1e-6)
    )


# ============================================================
# 7. PROCESS CONDITION FEATURES
# ============================================================

# RDOC is available for only 212 of 968 cycles,
# so we do NOT create a derived feature from it.

if (
    "ToolHolderLength" in data.columns
    and "HardnessMean" in data.columns
):

    data["ToolHolder_Hardness_Ratio"] = (
        pd.to_numeric(
            data["ToolHolderLength"],
            errors="coerce"
        )
        /
        (
            pd.to_numeric(
                data["HardnessMean"],
                errors="coerce"
            )
            + 1e-6
        )
    )

# ============================================================
# 8. HANDLE INFINITE VALUES
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 9. REPORT MISSING VALUES
# ============================================================

missing = data.isna().sum()

missing = missing[
    missing > 0
].sort_values(
    ascending=False
)

print("\n========== MISSING VALUES ==========")

if len(missing) == 0:

    print("No missing values.")

else:

    print(missing)


# ============================================================
# 10. SAVE ENGINEERED DATA
# ============================================================

output_file = (
    "data/milling/milling_engineered.csv"
)

data.to_csv(
    output_file,
    index=False
)


# ============================================================
# 11. FINAL REPORT
# ============================================================

print("\n========== FEATURE ENGINEERING COMPLETE ==========")

print(
    "Original features:",
    len(pd.read_csv(input_file).columns)
)

print(
    "Engineered features:",
    len(data.columns)
)

print(
    "New features added:",
    len(data.columns)
    - len(pd.read_csv(input_file).columns)
)

print("\nNew engineering features:")

new_features = [
    c for c in data.columns
    if c not in pd.read_csv(input_file).columns
]

for feature in new_features:
    print("-", feature)

print("\nSaved to:")
print(output_file)