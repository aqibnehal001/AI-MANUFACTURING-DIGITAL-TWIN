import pandas as pd

data = pd.read_csv(
    "reports/log_rf_predictions.csv"
)

print("\n========== LOG MODEL ERROR ANALYSIS ==========")

print(
    "Mean Absolute Error:",
    data["Absolute_Error"].mean()
)

print(
    "Maximum Absolute Error:",
    data["Absolute_Error"].max()
)

print("\n========== 10 WORST LOG PREDICTIONS ==========")

worst = data.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

print(
    worst[
        [
            "ToolID",
            "Actual_RUL",
            "Predicted_RUL",
            "Error",
            "Absolute_Error"
        ]
    ].to_string(index=False)
)

print("\n========== TOOL-WISE ERROR ==========")

tool_error = (
    data.groupby("ToolID")
    .agg(
        Rows=("Actual_RUL", "size"),
        MAE=("Absolute_Error", "mean"),
        ActualMean=("Actual_RUL", "mean"),
        PredictedMean=("Predicted_RUL", "mean")
    )
    .sort_values(
        "MAE",
        ascending=False
    )
)

print(
    tool_error.to_string()
)