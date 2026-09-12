import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(
    "data/milling/milling_cleaned.csv"
)

target = "CycleToFailure"


# ============================================================
# 2. FEATURES
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

y = data[target]

# Keep only numerical features
X = X.select_dtypes(include=np.number)


# ============================================================
# 3. TOOL-WISE TRAIN / TEST SPLIT
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
# ============================================================
# HANDLE MISSING FEATURE VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

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

print("\nMissing values handled using training-set medians.")


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
# 4. MODELS
# ============================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ============================================================
# 5. TRAIN + EVALUATE
# ============================================================

results = []

print("\n========== MODEL COMPARISON ==========")

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")


# ============================================================
# 6. COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE"
)

print("\n========== FINAL COMPARISON ==========")

print(
    results_df.to_string(index=False)
)


# ============================================================
# 7. BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n========== BEST MODEL ==========")

print("Model:", best_model["Model"])
print(f"MAE : {best_model['MAE']:.3f}")
print(f"RMSE: {best_model['RMSE']:.3f}")
print(f"R²  : {best_model['R2']:.3f}")


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("reports/model_comparison.csv")