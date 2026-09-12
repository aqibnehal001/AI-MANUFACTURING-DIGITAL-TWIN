import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

data = pd.read_csv(
    "reports/tuned_rf_predictions.csv"
)


# ============================================================
# 2. ACTUAL vs PREDICTED RUL
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    data["Actual_RUL"],
    data["Predicted_RUL"],
    alpha=0.7
)

# Perfect prediction line
minimum = min(
    data["Actual_RUL"].min(),
    data["Predicted_RUL"].min()
)

maximum = max(
    data["Actual_RUL"].max(),
    data["Predicted_RUL"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted Remaining Useful Life")

plt.tight_layout()

plt.savefig(
    "reports/actual_vs_predicted_rul.png",
    dpi=300
)

plt.show()


# ============================================================
# 3. PREDICTION ERROR
# ============================================================

data["Error"] = (
    data["Predicted_RUL"]
    - data["Actual_RUL"]
)


plt.figure(figsize=(8, 6))

plt.hist(
    data["Error"],
    bins=25
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of RUL Prediction Errors")

plt.tight_layout()

plt.savefig(
    "reports/rul_error_distribution.png",
    dpi=300
)

plt.show()


print("\n========== VISUALIZATION COMPLETE ==========")

print(
    "Saved:"
)

print(
    "reports/actual_vs_predicted_rul.png"
)

print(
    "reports/rul_error_distribution.png"
)