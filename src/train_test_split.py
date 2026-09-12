import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load cleaned dataset
file_path = "data/milling/milling_cleaned.csv"

data = pd.read_csv(file_path)

# Target
target = "CycleToFailure"

# Separate features and target
X = data.drop(columns=[target, "CycleToFailureNormalized", "ToolID"])
y = data[target]

# Tool identity is used ONLY for grouping
groups = data["ToolID"]

# 85% tools for training, 15% tools for testing
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

train_tools = sorted(groups.iloc[train_idx].unique())
test_tools = sorted(groups.iloc[test_idx].unique())

print("========== TRAIN / TEST SPLIT ==========")

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining tools:")
print(train_tools)

print("\nTesting tools:")
print(test_tools)

print("\nNumber of training tools:", len(train_tools))
print("Number of testing tools:", len(test_tools))