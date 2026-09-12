import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD SELECTED DATASET
# ============================================================

data = pd.read_csv(
    "data/milling/milling_degradation_selected.csv"
)

print("Dataset shape:", data.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

target = "CycleToFailure"

y = data[target]

X = data.drop(
    columns=[
        target,
        "CycleToFailureNormalized",
        "ToolID"
    ],
    errors="ignore"
)

X = X.select_dtypes(
    include=np.number
)

groups = data["ToolID"]


# ============================================================
# 3. TOOL-WISE TRAIN / TEST SPLIT
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


print("\n========== DATA SPLIT ==========")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

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
    columns=X_train.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)


# ============================================================
# 5. RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 6. PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# 7. EVALUATION
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


print("\n========== DEGRADATION MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.3f} cycles")
print(f"RMSE : {rmse:.3f} cycles")
print(f"R²   : {r2:.3f}")


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(
    ascending=False
)


print("\n========== TOP 20 FEATURES ==========")

print(
    importance.head(20)
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

importance.to_csv(
    "reports/degradation_feature_importance.csv"
)

results = pd.DataFrame({
    "Actual_RUL": y_test.values,
    "Predicted_RUL": predictions
})

results.to_csv(
    "reports/degradation_predictions.csv",
    index=False
)


print("\nSaved:")

print(
    "reports/degradation_feature_importance.csv"
)

print(
    "reports/degradation_predictions.csv"
)