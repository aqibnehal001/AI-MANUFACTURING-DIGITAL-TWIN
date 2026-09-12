import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(
    "data/milling/milling_cleaned.csv"
)

target = "CycleToFailure"

y = data[target]
groups = data["ToolID"]


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

X = data.drop(
    columns=[
        target,
        "CycleToFailureNormalized",
        "ToolID",
        "NumberOfCycle"
    ],
    errors="ignore"
)

X = X.select_dtypes(
    include=np.number
)


# ============================================================
# 3. SAME TOOL-WISE TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("========== TOOL SPLIT ==========")

print(
    "Training tools:",
    sorted(groups.iloc[train_idx].unique().tolist())
)

print(
    "Testing tools:",
    sorted(groups.iloc[test_idx].unique().tolist())
)


# ============================================================
# 4. IMPUTE MISSING VALUES
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# 5. SELECT TOP 20 FEATURES
#    TRAINING DATA ONLY
# ============================================================

correlation_data = X_train.copy()

correlation_data[target] = y_train.values

correlations = (
    correlation_data
    .corr(numeric_only=True)[target]
    .drop(target)
    .abs()
    .sort_values(
        ascending=False
    )
)

selected_features = (
    correlations
    .head(20)
    .index
    .tolist()
)


print("\n========== SELECTED FEATURES ==========")

for feature in selected_features:
    print("-", feature)


X_train = X_train[selected_features]
X_test = X_test[selected_features]


# ============================================================
# 6. LOG-TRANSFORM TARGET
# ============================================================

y_train_log = np.log1p(
    y_train
)


# ============================================================
# 7. TRAIN TUNED RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=2
)


print("\n========== TRAINING LOG RANDOM FOREST ==========")

model.fit(
    X_train,
    y_train_log
)


# ============================================================
# 8. PREDICT AND CONVERT BACK TO RUL
# ============================================================

predictions_log = model.predict(
    X_test
)

predictions = np.expm1(
    predictions_log
)

# RUL cannot be negative
predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# 9. EVALUATE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n========== LOG MODEL PERFORMANCE ==========")

print(
    f"MAE  : {mae:.3f} cycles"
)

print(
    f"RMSE : {rmse:.3f} cycles"
)

print(
    f"R²   : {r2:.3f}"
)


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "ToolID": groups.iloc[test_idx].values,
    "NumberOfCycle": data.iloc[test_idx]["NumberOfCycle"].values,
    "Actual_RUL": y_test.values,
    "Predicted_RUL": predictions
})

results["Error"] = (
    results["Predicted_RUL"]
    - results["Actual_RUL"]
)

results["Absolute_Error"] = (
    results["Error"].abs()
)


results.to_csv(
    "reports/log_rf_predictions.csv",
    index=False
)


summary = pd.DataFrame([{
    "Model": "Log Random Forest",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

summary.to_csv(
    "reports/log_rf_results.csv",
    index=False
)


print("\nSaved:")

print(
    "reports/log_rf_predictions.csv"
)

print(
    "reports/log_rf_results.csv"
)