import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD ORIGINAL DATA
# ============================================================

data = pd.read_csv(
    "data/milling/milling_cleaned.csv"
)


# ============================================================
# 2. LOAD TEST PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    "reports/tuned_rf_predictions.csv"
)


print("Prediction rows:", len(predictions))


# ============================================================
# 3. CALCULATE ERROR
# ============================================================

predictions["Error"] = (
    predictions["Predicted_RUL"]
    - predictions["Actual_RUL"]
)

predictions["Absolute_Error"] = (
    predictions["Error"].abs()
)


predictions["Percentage_Error"] = (
    predictions["Absolute_Error"]
    /
    predictions["Actual_RUL"].replace(
        0,
        np.nan
    )
) * 100


# ============================================================
# 4. OVERALL ERROR ANALYSIS
# ============================================================

print("\n========== ERROR ANALYSIS ==========")

print(
    "Mean Absolute Error:",
    predictions["Absolute_Error"].mean()
)

print(
    "Maximum Absolute Error:",
    predictions["Absolute_Error"].max()
)

print(
    "Minimum Absolute Error:",
    predictions["Absolute_Error"].min()
)


# ============================================================
# 5. WORST PREDICTIONS
# ============================================================

print("\n========== 10 WORST PREDICTIONS ==========")

worst = predictions.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

print(
    worst[
        [
            "Actual_RUL",
            "Predicted_RUL",
            "Error",
            "Absolute_Error"
        ]
    ].to_string(index=False)
)


# ============================================================
# 6. BEST PREDICTIONS
# ============================================================

print("\n========== 10 BEST PREDICTIONS ==========")

best = predictions.sort_values(
    "Absolute_Error"
).head(10)

print(
    best[
        [
            "Actual_RUL",
            "Predicted_RUL",
            "Error",
            "Absolute_Error"
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. SAVE DETAILED ANALYSIS
# ============================================================

predictions.to_csv(
    "reports/prediction_error_analysis.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "reports/prediction_error_analysis.csv"
)