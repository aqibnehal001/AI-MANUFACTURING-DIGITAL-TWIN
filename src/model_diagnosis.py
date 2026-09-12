import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    "data/milling/milling_cleaned.csv"
)

target = "CycleToFailure"

groups = data["ToolID"]


# ============================================================
# SAME SPLIT AS OUR MODEL
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(data, data[target], groups=groups)
)

train = data.iloc[train_idx]
test = data.iloc[test_idx]


# ============================================================
# TOOL-WISE RUL DISTRIBUTION
# ============================================================

print("\n========== TRAINING TOOLS ==========")

print(
    train.groupby("ToolID")[target]
    .agg(["count", "min", "max", "mean"])
)


print("\n========== TESTING TOOLS ==========")

print(
    test.groupby("ToolID")[target]
    .agg(["count", "min", "max", "mean"])
)


# ============================================================
# OVERALL DISTRIBUTION
# ============================================================

print("\n========== OVERALL ==========")

print(
    "Training mean RUL:",
    round(train[target].mean(), 2)
)

print(
    "Testing mean RUL:",
    round(test[target].mean(), 2)
)

print(
    "Training min RUL:",
    train[target].min()
)

print(
    "Training max RUL:",
    train[target].max()
)

print(
    "Testing min RUL:",
    test[target].min()
)

print(
    "Testing max RUL:",
    test[target].max()
)