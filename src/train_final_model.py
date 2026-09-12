import pandas as pd
import numpy as np
import joblib
import os

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


print("========== FINAL MODEL ==========")

print(
    "Training tools:",
    sorted(
        groups.iloc[train_idx].unique().tolist()
    )
)

print(
    "Testing tools:",
    sorted(
        groups.iloc[test_idx].unique().tolist()
    )
)


# ============================================================
# 4. IMPUTE MISSING VALUES
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


# ============================================================
# 5. SELECT TOP 20 FEATURES
# ============================================================

correlation_data = X_train_imputed.copy()

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


print("\n========== FINAL 20 FEATURES ==========")

for i, feature in enumerate(
    selected_features,
    start=1
):
    print(
        f"{i}. {feature}"
    )


X_train_final = X_train_imputed[
    selected_features
]

X_test_final = X_test_imputed[
    selected_features
]


# ============================================================
# 6. LOG TRANSFORM TARGET
# ============================================================

y_train_log = np.log1p(
    y_train
)


# ============================================================
# 7. TRAIN FINAL LOG RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=2
)


print(
    "\nTraining final Log Random Forest..."
)

model.fit(
    X_train_final,
    y_train_log
)


# ============================================================
# 8. PREDICT
# ============================================================

predictions_log = model.predict(
    X_test_final
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
# 9. EVALUATE MODEL
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


print(
    "\n========== FINAL PERFORMANCE =========="
)

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
# 10. CREATE DIRECTORIES
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "reports",
    exist_ok=True
)


# ============================================================
# 11. SAVE COMPLETE MODEL PACKAGE
# ============================================================

model_package = {
    "model": model,
    "imputer": imputer,
    "selected_features": selected_features,
    "target": target,
    "log_transform": True
}


joblib.dump(
    model_package,
    "models/final_rul_model.pkl"
)


# ============================================================
# 12. SAVE FEATURE LIST
# ============================================================

pd.DataFrame({
    "Feature": selected_features
}).to_csv(
    "reports/final_selected_features.csv",
    index=False
)


# ============================================================
# 13. SAVE TEST PREDICTIONS
# ============================================================

final_predictions = pd.DataFrame({
    "ToolID": groups.iloc[test_idx].values,
    "Actual_RUL": y_test.values,
    "Predicted_RUL": predictions
})

final_predictions["Error"] = (
    final_predictions["Predicted_RUL"]
    - final_predictions["Actual_RUL"]
)

final_predictions["Absolute_Error"] = (
    final_predictions["Error"].abs()
)


final_predictions.to_csv(
    "reports/final_model_predictions.csv",
    index=False
)


# ============================================================
# 14. SAVE FINAL METRICS
# ============================================================

pd.DataFrame([{
    "Model": "Final Log Random Forest",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}]).to_csv(
    "reports/final_model_metrics.csv",
    index=False
)


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print(
    "\n========== SAVED =========="
)

print(
    "models/final_rul_model.pkl"
)

print(
    "reports/final_selected_features.csv"
)

print(
    "reports/final_model_predictions.csv"
)

print(
    "reports/final_model_metrics.csv"
)