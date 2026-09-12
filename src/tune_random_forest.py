import pandas as pd
import numpy as np

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
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
# 2. PREPARE NUMERICAL FEATURES
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

groups_train = groups.iloc[train_idx]


print("========== TOOL SPLIT ==========")

print(
    "Training tools:",
    sorted(groups_train.unique().tolist())
)

print(
    "Testing tools:",
    sorted(groups.iloc[test_idx].unique().tolist())
)


# ============================================================
# 4. SELECT TOP 20 FEATURES
#    TRAINING DATA ONLY
# ============================================================

# Temporary median filling only for calculating correlations
X_train_for_selection = X_train.copy()

medians = X_train_for_selection.median(
    numeric_only=True
)

X_train_for_selection = (
    X_train_for_selection.fillna(medians)
)

correlation_data = X_train_for_selection.copy()

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
# 5. RANDOM FOREST PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "model",
       RandomForestRegressor(
    random_state=42,
    n_jobs=2
)
    )
])


# ============================================================
# 6. HYPERPARAMETER GRID
# ============================================================

parameter_grid = {

    "model__n_estimators": [
        200,
        300
    ],

    "model__max_depth": [
        None,
        10,
        15
    ],

    "model__min_samples_leaf": [
        1,
        2
    ],

    "model__max_features": [
        "sqrt"
    ]
}


# ============================================================
# ============================================================
# 7. GROUPED CROSS-VALIDATION
# ============================================================

cv = GroupKFold(n_splits=3)

# ============================================================
# 8. GRID SEARCH
# ============================================================

print("\n========== RANDOM FOREST TUNING ==========")

print(
    "Running grouped cross-validation..."
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(
    X_train,
    y_train,
    groups=groups_train
)


# ============================================================
# 9. BEST PARAMETERS
# ============================================================

print("\n========== BEST PARAMETERS ==========")

print(
    search.best_params_
)

print(
    f"Best CV MAE: {-search.best_score_:.3f} cycles"
)


# ============================================================
# 10. FINAL TEST EVALUATION
# ============================================================

best_model = search.best_estimator_

predictions = best_model.predict(
    X_test
)


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


print("\n========== FINAL TEST PERFORMANCE ==========")

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
# 11. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "ToolID": groups.iloc[test_idx].values,
    "NumberOfCycle": data.iloc[test_idx]["NumberOfCycle"].values,
    "Actual_RUL": y_test.values,
    "Predicted_RUL": predictions
})

results.to_csv(
    "reports/tuned_rf_predictions.csv",
    index=False
)


summary = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

summary.to_csv(
    "reports/tuned_rf_results.csv",
    index=False
)


print("\nSaved:")

print(
    "reports/tuned_rf_predictions.csv"
)

print(
    "reports/tuned_rf_results.csv"
)