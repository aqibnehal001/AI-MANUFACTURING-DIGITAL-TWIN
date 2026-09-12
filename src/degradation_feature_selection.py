import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer


# ============================================================
# 1. LOAD DEGRADATION DATASET
# ============================================================

input_file = "data/milling/milling_degradation.csv"

data = pd.read_csv(input_file)

print("Original dataset shape:", data.shape)


# ============================================================
# 2. DEFINE TARGET AND GROUP
# ============================================================

target = "CycleToFailure"
group_column = "ToolID"


# ============================================================
# 3. TOOL-WISE TRAIN / TEST SPLIT
# ============================================================

groups = data[group_column]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        data,
        data[target],
        groups=groups
    )
)

train_data = data.iloc[train_idx].copy()
test_data = data.iloc[test_idx].copy()


print("\n========== TOOL-WISE SPLIT ==========")

print(
    "Training tools:",
    sorted(train_data[group_column].unique().tolist())
)

print(
    "Testing tools:",
    sorted(test_data[group_column].unique().tolist())
)

print(
    "Training rows:",
    len(train_data)
)

print(
    "Testing rows:",
    len(test_data)
)


# ============================================================
# 4. REMOVE TARGET / IDENTIFIER / LEAKAGE COLUMNS
# ============================================================

columns_to_remove = [
    target,
    "CycleToFailureNormalized",
    "ToolID",
    "NumberOfCycle"
]

X_train = train_data.drop(
    columns=columns_to_remove,
    errors="ignore"
)

X_test = test_data.drop(
    columns=columns_to_remove,
    errors="ignore"
)


# Keep numerical features only

X_train = X_train.select_dtypes(
    include=np.number
)

X_test = X_test[
    X_train.columns
]


print(
    "\nInitial numerical features:",
    X_train.shape[1]
)


# ============================================================
# 5. HANDLE MISSING VALUES
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
# 6. TARGET CORRELATION
#    IMPORTANT: TRAINING DATA ONLY
# ============================================================

y_train = train_data[target]

correlation_data = X_train_imputed.copy()

correlation_data[target] = y_train.values

target_correlations = (
    correlation_data
    .corr(numeric_only=True)[target]
    .drop(target)
    .abs()
    .sort_values(ascending=False)
)


print("\n========== TOP FEATURES BY TARGET CORRELATION ==========")

print(
    target_correlations.head(30)
)


# ============================================================
# 7. KEEP TOP FEATURES
# ============================================================

TOP_FEATURES = 100

top_features = (
    target_correlations
    .head(TOP_FEATURES)
    .index
    .tolist()
)

print(
    "\nFeatures retained after target screening:",
    len(top_features)
)


# ============================================================
# 8. REMOVE HIGHLY CORRELATED FEATURES
# ============================================================

correlation_matrix = (
    X_train_imputed[top_features]
    .corr()
    .abs()
)

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(
            correlation_matrix.shape
        ),
        k=1
    ).astype(bool)
)


redundant_features = [
    column
    for column in upper_triangle.columns
    if any(
        upper_triangle[column] > 0.95
    )
]


selected_features = [
    feature
    for feature in top_features
    if feature not in redundant_features
]


print(
    "\nHighly correlated features removed:",
    len(redundant_features)
)

print(
    "Final selected features:",
    len(selected_features)
)


# ============================================================
# 9. SHOW FINAL FEATURES
# ============================================================

print("\n========== FINAL SELECTED FEATURES ==========")

for feature in selected_features:
    print("-", feature)


# ============================================================
# 10. CREATE REDUCED DATASET
# ============================================================

keep_columns = [
    "ToolID",
    "CycleToFailure"
]

if "CycleToFailureNormalized" in data.columns:
    keep_columns.append(
        "CycleToFailureNormalized"
    )

keep_columns += selected_features


reduced_data = data[
    keep_columns
].copy()


# ============================================================
# 11. SAVE REDUCED DATASET
# ============================================================

output_file = (
    "data/milling/"
    "milling_degradation_selected.csv"
)

reduced_data.to_csv(
    output_file,
    index=False
)


# ============================================================
# 12. SAVE FEATURE LIST
# ============================================================

feature_list = pd.DataFrame({
    "Feature": selected_features,
    "TargetCorrelation": [
        target_correlations[f]
        for f in selected_features
    ]
})

feature_list.to_csv(
    "reports/degradation_selected_features.csv",
    index=False
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n========== FEATURE SELECTION COMPLETE ==========")

print(
    "Original features:",
    X_train.shape[1]
)

print(
    "After target screening:",
    len(top_features)
)

print(
    "After redundancy removal:",
    len(selected_features)
)

print(
    "\nReduced dataset shape:",
    reduced_data.shape
)

print("\nSaved dataset:")
print(output_file)

print("\nSaved feature list:")
print(
    "reports/degradation_selected_features.csv"
)