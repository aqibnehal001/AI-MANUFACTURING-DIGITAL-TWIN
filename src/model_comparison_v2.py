import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(
    "data/milling/milling_cleaned.csv"
)

target = "CycleToFailure"


# ============================================================
# 2. REMOVE TARGET / LEAKAGE COLUMNS
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
# 4. HANDLE MISSING VALUES
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


# ============================================================
# 5. FEATURE SELECTION USING TRAINING DATA ONLY
# ============================================================

print("\n========== FEATURE SELECTION ==========")

correlations = X_train.copy()

correlations["TARGET"] = y_train.values

correlation_scores = (
    correlations
    .corr(numeric_only=True)["TARGET"]
    .drop("TARGET")
    .abs()
    .sort_values(ascending=False)
)

print("\nTop 20 training-only features:")

print(
    correlation_scores.head(20)
)


# ============================================================
# 6. TEST DIFFERENT FEATURE COUNTS
# ============================================================

feature_counts = [10, 20, 30, 50, 75, 100, 126]

all_results = []


for number_of_features in feature_counts:

    number_of_features = min(
        number_of_features,
        len(correlation_scores)
    )

    selected_features = (
        correlation_scores
        .head(number_of_features)
        .index
        .tolist()
    )

    X_train_selected = X_train[
        selected_features
    ]

    X_test_selected = X_test[
        selected_features
    ]


    print(
        f"\nTesting {number_of_features} features..."
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_selected,
        y_train
    )

    predictions = model.predict(
        X_test_selected
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


    all_results.append({
        "Features": number_of_features,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


    print(
        f"MAE={mae:.3f} | "
        f"RMSE={rmse:.3f} | "
        f"R²={r2:.3f}"
    )


# ============================================================
# 7. FINAL COMPARISON
# ============================================================

results = pd.DataFrame(all_results)

results = results.sort_values(
    by="MAE"
)

print("\n========== FEATURE COUNT COMPARISON ==========")

print(
    results.to_string(index=False)
)


# ============================================================
# 8. BEST FEATURE COUNT
# ============================================================

best = results.iloc[0]

print("\n========== BEST FEATURE SET ==========")

print(
    "Number of features:",
    int(best["Features"])
)

print(
    f"MAE : {best['MAE']:.3f}"
)

print(
    f"RMSE: {best['RMSE']:.3f}"
)

print(
    f"R²  : {best['R2']:.3f}"
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results.to_csv(
    "reports/feature_count_comparison.csv",
    index=False
)

print(
    "\nSaved to:"
)

print(
    "reports/feature_count_comparison.csv"
)