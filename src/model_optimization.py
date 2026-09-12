import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

data = pd.read_csv(
    "data/milling/milling_cleaned.csv"
)

target = "CycleToFailure"

y = data[target]

groups = data["ToolID"]


# ============================================================
# 2. REMOVE NON-FEATURE COLUMNS
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


print("========== TOOL SPLIT ==========")

print(
    "Training tools:",
    sorted(groups.iloc[train_idx].unique())
)

print(
    "Testing tools:",
    sorted(groups.iloc[test_idx].unique())
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

corr = X_train.copy()

corr[target] = y_train.values

correlations = (
    corr.corr()[target]
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


print("\n========== SELECTED 20 FEATURES ==========")

for feature in selected_features:
    print("-", feature)


X_train = X_train[selected_features]
X_test = X_test[selected_features]


# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        random_state=42,
        loss="huber"
    )
}


# ============================================================
# 7. TRAIN AND EVALUATE
# ============================================================

results = []


print("\n========== MODEL COMPARISON ==========")


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
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

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(
        f"MAE  : {mae:.3f}"
    )

    print(
        f"RMSE : {rmse:.3f}"
    )

    print(
        f"R2   : {r2:.3f}"
    )


# ============================================================
# 8. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    "MAE"
)


print(
    "\n========== FINAL MODEL COMPARISON =========="
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "reports/model_optimization.csv",
    index=False
)


print(
    "\nSaved to:"
)

print(
    "reports/model_optimization.csv"
)