import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

input_file = "data/milling/milling_cleaned.csv"

data = pd.read_csv(input_file)

print("Original shape:", data.shape)


# ============================================================
# 2. SORT DATA BY TOOL AND CYCLE
# ============================================================

data = data.sort_values(
    ["ToolID", "NumberOfCycle"]
).reset_index(drop=True)


# ============================================================
# 3. IDENTIFY IMPORTANT SENSOR FEATURES
# ============================================================

sensor_columns = [
    c for c in data.columns
    if (
        "accelerometer" in c.lower()
        or "current" in c.lower()
    )
]

print("\nSensor columns found:", len(sensor_columns))


# ============================================================
# 4. CONVERT SENSOR DATA TO NUMERIC
# ============================================================

for col in sensor_columns:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


# ============================================================
# 5. CREATE TOOL-WISE ROLLING FEATURES
# ============================================================

new_features = []

for col in sensor_columns:

    safe_name = (
        col.replace(" ", "_")
           .replace("-", "_")
           .replace("+", "plus")
    )

    # Rolling mean
    rolling_mean = (
        data.groupby("ToolID")[col]
        .transform(
            lambda x: x.rolling(
                window=5,
                min_periods=1
            ).mean()
        )
    )

    data[f"{safe_name}_RollingMean"] = rolling_mean

    new_features.append(
        f"{safe_name}_RollingMean"
    )


    # Rolling standard deviation
    rolling_std = (
        data.groupby("ToolID")[col]
        .transform(
            lambda x: x.rolling(
                window=5,
                min_periods=2
            ).std()
        )
    )

    data[f"{safe_name}_RollingSTD"] = rolling_std

    new_features.append(
        f"{safe_name}_RollingSTD"
    )


    # Change from previous cycle
    data[f"{safe_name}_Delta"] = (
        data.groupby("ToolID")[col]
        .diff()
    )

    new_features.append(
        f"{safe_name}_Delta"
    )


# ============================================================
# 6. BASELINE DEVIATION
# ============================================================

for col in sensor_columns:

    safe_name = (
        col.replace(" ", "_")
           .replace("-", "_")
           .replace("+", "plus")
    )

    # First value of each tool = early-life baseline
    baseline = (
        data.groupby("ToolID")[col]
        .transform("first")
    )

    data[f"{safe_name}_BaselineDeviation"] = (
        data[col] - baseline
    )

    new_features.append(
        f"{safe_name}_BaselineDeviation"
    )


# ============================================================
# 7. HANDLE INFINITE VALUES
# ============================================================

data = data.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 8. REPORT NEW FEATURES
# ============================================================

print("\n========== NEW FEATURES ==========")

for feature in new_features:
    print("-", feature)

print(
    "\nNumber of new features:",
    len(new_features)
)


# ============================================================
# 9. SAVE DATASET
# ============================================================

output_file = (
    "data/milling/milling_degradation.csv"
)

data.to_csv(
    output_file,
    index=False
)


# ============================================================
# 10. FINAL REPORT
# ============================================================

print("\n========== COMPLETE ==========")

print(
    "New shape:",
    data.shape
)

print(
    "Saved to:",
    output_file
)