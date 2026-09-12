import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(
    "data/milling/milling_cleaned.csv"
)

target = "CycleToFailure"

# Same test tools that we already established
TEST_TOOLS = [2, 11, 102]

# Everything else is training data
train_data = data[
    ~data["ToolID"].isin(TEST_TOOLS)
].copy()


print("Training rows:", len(train_data))

print(
    "Training tools:",
    sorted(train_data["ToolID"].unique().tolist())
)


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

X = train_data.drop(
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

y = train_data[target]

groups = train_data["ToolID"]


# ============================================================
# 3. IMPUTE
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)


# ============================================================
# 4. SELECT TOP 20 FEATURES
# ============================================================

correlation_data = X.copy()

correlation_data[target] = y.values

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

X = X[selected_features]


print("\n========== SELECTED FEATURES ==========")

for feature in selected_features:
    print("-", feature)


# ============================================================
# 5. GROUPED CROSS-VALIDATION
# ============================================================

cv = GroupKFold(
    n_splits=5
)

fold_results = []


print("\n========== CROSS-VALIDATION ==========")


for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        n_jobs=2
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_val
    )

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    validation_tools = sorted(
        groups.iloc[val_idx].unique().tolist()
    )

    fold_results.append({
        "Fold": fold,
        "MAE": mae,
        "ValidationTools": str(validation_tools)
    })

    print(
        f"Fold {fold}: "
        f"MAE = {mae:.3f} cycles | "
        f"Tools = {validation_tools}"
    )


# ============================================================
# 6. SUMMARY
# ============================================================

results = pd.DataFrame(
    fold_results
)

print("\n========== CV SUMMARY ==========")

print(results.to_string(index=False))

print(
    f"\nMean CV MAE: "
    f"{results['MAE'].mean():.3f} cycles"
)

print(
    f"Std CV MAE: "
    f"{results['MAE'].std():.3f} cycles"
)


# ============================================================
# 7. SAVE
# ============================================================

results.to_csv(
    "reports/cross_validation_results.csv",
    index=False
)

print(
    "\nSaved to:"
)

print(
    "reports/cross_validation_results.csv"
)