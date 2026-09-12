import pandas as pd
import numpy as np
import joblib


# ============================================================
# 1. LOAD FINAL MODEL
# ============================================================

model_package = joblib.load(
    "models/final_rul_model.pkl"
)

model = model_package["model"]
imputer = model_package["imputer"]
selected_features = model_package["selected_features"]


print("Final RUL model loaded successfully!")


# ============================================================
# 2. PREDICTION FUNCTION
# ============================================================

def predict_rul(input_data):

    # Convert input into DataFrame
    df = pd.DataFrame(
        [input_data]
    )

    # The imputer was trained on 126 features.
    # Get those exact feature names from the saved imputer.
    imputer_features = list(
        imputer.feature_names_in_
    )

    # Make sure all 126 imputer features exist.
    # Dashboard may provide only the 20 model features.
    for feature in imputer_features:

        if feature not in df.columns:
            df[feature] = np.nan

    # Keep the exact feature order used by the imputer
    df = df[imputer_features]

    # Apply the same imputer used during training
    df = pd.DataFrame(
        imputer.transform(df),
        columns=imputer_features
    )

    # After imputation, keep only the 20 features
    # used by the final Random Forest model.
    df = df[selected_features]

    # Predict log(RUL + 1)
    prediction_log = model.predict(
        df
    )[0]

    # Convert back to original RUL scale
    predicted_rul = np.expm1(
        prediction_log
    )

    # RUL cannot be negative
    predicted_rul = max(
        0,
        predicted_rul
    )

    return predicted_rul


# ============================================================
# 3. MACHINE HEALTH STATUS
# ============================================================

def get_health_status(rul):

    if rul <= 10:
        return "CRITICAL"

    elif rul <= 30:
        return "WARNING"

    else:
        return "HEALTHY"


# ============================================================
# 4. TEST PREDICTION
# ============================================================

if __name__ == "__main__":

    print(
        "\nRequired model features:"
    )

    for feature in selected_features:
        print(
            "-",
            feature
        )

    print(
        "\nModel is ready for dashboard integration."
    )