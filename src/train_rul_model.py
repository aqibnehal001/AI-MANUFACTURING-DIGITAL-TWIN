import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "data/milling/milling_cleaned.csv"

data = pd.read_csv(file_path)

print("Dataset shape:", data.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

target = "CycleToFailure"


# ============================================================
# 3. REMOVE TARGET / LEAKAGE / IDENTIFIER COLUMNS
# ============================================================

columns_to_remove = [
    target,
    "CycleToFailureNormalized",
    "ToolID",
    "NumberOfCycle"
]

X = data.drop(
    columns=columns_to_remove,
    errors="ignore"
)

y = data[target]

# Keep only numerical features
X = X.select_dtypes(include=np.number)

print("Number of ML features:", X.shape[1])


# ============================================================
# 4. TOOL-WISE TRAIN / TEST SPLIT
# ============================================================

groups = data["ToolID"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_tools = sorted(
    groups.iloc[train_idx].unique().tolist()
)

test_tools = sorted(
    groups.iloc[test_idx].unique().tolist()
)

print("\n========== TOOL-WISE SPLIT ==========")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("Training tools:", train_tools)
print("Testing tools:", test_tools)


# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

print("\n========== RANDOM FOREST ==========")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# 6. PREDICTION
# ============================================================

predictions = model.predict(X_test)


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


print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE  : {mae:.3f} cycles")
print(f"RMSE : {rmse:.3f} cycles")
print(f"R²   : {r2:.3f}")


# ============================================================
# 8. SAMPLE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual_RUL": y_test.values,
    "Predicted_RUL": predictions
})

print("\n========== SAMPLE PREDICTIONS ==========")

print(
    results.head(15).to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results.to_csv(
    "reports/rul_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print("reports/rul_predictions.csv")