import os
import sys
import pandas as pd
from datetime import datetime

from flask import Flask, render_template, request


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Make sure Python can find the "src" folder
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# Now import the prediction functions
from src.predict import predict_rul, get_health_status


app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "milling",
    "milling_cleaned.csv"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "final_selected_features.csv"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "rul_history.csv"
)
TELEMETRY_HISTORY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "telemetry_history.csv"
)


# ============================================================
# LOAD SELECTED MODEL FEATURES
# ============================================================

def load_selected_features():

    try:

        selected_features = (
            pd.read_csv(FEATURE_PATH)
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .tolist()
        )

        return selected_features

    except Exception as e:

        print(
            "ERROR LOADING SELECTED FEATURES:",
            e
        )

        return []


# ============================================================
# LOAD DATASET
# ============================================================

def load_milling_data():

    try:

        data = pd.read_csv(DATA_PATH)

        data.columns = (
            data.columns
            .astype(str)
            .str.strip()
        )

        return data

    except Exception as e:

        print(
            "ERROR LOADING MILLING DATA:",
            e
        )

        return pd.DataFrame()


# ============================================================
# GET VALUES FROM A DATASET ROW
# ============================================================

def load_values_for_row(row_index):

    try:

        data = load_milling_data()

        selected_features = load_selected_features()

        if data.empty:
            return {}

        if not selected_features:
            return {}

        # Keep row index inside dataset range
        row_index = row_index % len(data)

        row = data.iloc[row_index]

        values = {}

        for feature in selected_features:

            if feature in data.columns:

                try:

                    values[feature] = float(
                        row[feature]
                    )

                except (ValueError, TypeError):

                    values[feature] = ""

        print(
            f"Loaded dataset row {row_index} "
            f"with {len(values)} model inputs."
        )

        return values

    except Exception as e:

        print(
            "ERROR LOADING DATASET ROW:",
            e
        )

        return {}


# ============================================================
# DETERMINE NEXT DATASET ROW
# ============================================================

def get_next_dataset_row():

    try:

        data = load_milling_data()

        if data.empty:
            return 0

        # Number of predictions already recorded
        if os.path.exists(HISTORY_PATH):

            history = pd.read_csv(
                HISTORY_PATH
            )

            prediction_count = len(history)

        else:

            prediction_count = 0

        # First dashboard load = row 0
        # After first prediction = row 1
        # After second prediction = row 2
        next_row = prediction_count % len(data)

        return next_row

    except Exception as e:

        print(
            "ERROR DETERMINING DATASET ROW:",
            e
        )

        return 0


# ============================================================
# SAVE RUL PREDICTION HISTORY
# ============================================================

def save_rul_history(rul, health_status):

    try:

        os.makedirs(
            os.path.dirname(HISTORY_PATH),
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        new_record = pd.DataFrame([
            {
                "timestamp": timestamp,

                "rul": round(
                    float(rul),
                    2
                ),

                "health_status": health_status
            }
        ])

        if os.path.exists(HISTORY_PATH):

            new_record.to_csv(
                HISTORY_PATH,
                mode="a",
                header=False,
                index=False
            )

        else:

            new_record.to_csv(
                HISTORY_PATH,
                mode="w",
                header=True,
                index=False
            )

        print(
            f"RUL history saved: "
            f"{round(float(rul), 2)} cycles | "
            f"{health_status}"
        )

    except Exception as e:

        print(
            "ERROR SAVING RUL HISTORY:",
            e
        )
        # ============================================================
# SAVE MACHINE TELEMETRY HISTORY
# ============================================================

def save_telemetry_history(input_data, rul, health_status):

    try:

        os.makedirs(
            os.path.dirname(TELEMETRY_HISTORY_PATH),
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        record = {
            "timestamp": timestamp,
            "rul": round(float(rul), 2),
            "health_status": health_status
        }

        # Add all sensor/model input values
        for feature, value in input_data.items():

            try:
                record[feature] = float(value)

            except (ValueError, TypeError):
                record[feature] = value

        new_record = pd.DataFrame([record])

        # ----------------------------------------------------
        # First telemetry record
        # ----------------------------------------------------

        if not os.path.exists(TELEMETRY_HISTORY_PATH):

            new_record.to_csv(
                TELEMETRY_HISTORY_PATH,
                mode="w",
                header=True,
                index=False
            )

        # ----------------------------------------------------
        # Existing telemetry history
        # ----------------------------------------------------

        else:

            existing = pd.read_csv(
                TELEMETRY_HISTORY_PATH
            )

            # Make sure both dataframes have the same columns
            all_columns = list(
                dict.fromkeys(
                    list(existing.columns) +
                    list(new_record.columns)
                )
            )

            existing = existing.reindex(
                columns=all_columns
            )

            new_record = new_record.reindex(
                columns=all_columns
            )

            combined = pd.concat(
                [existing, new_record],
                ignore_index=True
            )

            combined.to_csv(
                TELEMETRY_HISTORY_PATH,
                index=False
            )

        print(
            "Machine telemetry history saved successfully."
        )

    except Exception as e:

        print(
            "ERROR SAVING TELEMETRY HISTORY:",
            e
        )
        # ============================================================
# LOAD MACHINE TELEMETRY HISTORY
# ============================================================

def load_telemetry_history():

    try:

        if not os.path.exists(TELEMETRY_HISTORY_PATH):
            return []

        telemetry = pd.read_csv(
            TELEMETRY_HISTORY_PATH
        )

        if telemetry.empty:
            return []

        # Keep latest 20 telemetry records
        telemetry = telemetry.tail(20)

        records = telemetry.to_dict(
            orient="records"
        )

        return records

    except Exception as e:

        print(
            "ERROR LOADING TELEMETRY HISTORY:",
            e
        )

        return []


# ============================================================
# LOAD RUL PREDICTION HISTORY
# ============================================================

def load_rul_history():

    try:

        if not os.path.exists(HISTORY_PATH):

            return []

        history = pd.read_csv(
            HISTORY_PATH
        )

        if history.empty:

            return []

        # Keep latest 20 predictions
        history = history.tail(20)

        # Determine original assessment number
        full_history = pd.read_csv(
            HISTORY_PATH
        )

        start_number = max(
            1,
            len(full_history) -
            len(history) +
            1
        )

        records = []

        for position, (_, row) in enumerate(
            history.iterrows(),
            start=start_number
        ):

            records.append(
                {
                    "assessment": position,

                    "timestamp": str(
                        row["timestamp"]
                    ),

                    "rul": float(
                        row["rul"]
                    ),

                    "health_status": str(
                        row["health_status"]
                    )
                }
            )

        return records

    except Exception as e:

        print(
            "ERROR LOADING RUL HISTORY:",
            e
        )

        return []


# ============================================================
# DASHBOARD HOME
# ============================================================

@app.route("/")
def home():

    # --------------------------------------------------------
    # Determine which real dataset row should be displayed
    # --------------------------------------------------------

    next_row = get_next_dataset_row()

    default_values = load_values_for_row(
        next_row
    )

    # --------------------------------------------------------
    # Load prediction history
    # --------------------------------------------------------

    rul_history = load_rul_history()
    telemetry_history = load_telemetry_history()

    prediction = None
    health_status = None

    # --------------------------------------------------------
    # Show latest prediction if available
    # --------------------------------------------------------

    if rul_history:

        latest_record = rul_history[-1]

        prediction = latest_record["rul"]

        health_status = latest_record[
            "health_status"
        ]

    # --------------------------------------------------------
    # Render dashboard
    # --------------------------------------------------------

    return render_template(
        "index.html",

        prediction=prediction,

        health_status=health_status,

        default_values=default_values,

        rul_history=rul_history,

        telemetry_history=telemetry_history
    )


# ============================================================
# AI PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    input_data = {}

    # --------------------------------------------------------
    # Read values submitted by dashboard
    # --------------------------------------------------------

    for key, value in request.form.items():

        if value.strip() != "":

            try:

                input_data[key] = float(
                    value
                )

            except ValueError:

                pass

    print("\n======================================")
    print("AI PREDICTION REQUEST")
    print("======================================")

    print(
        f"Received {len(input_data)} model inputs."
    )

    # --------------------------------------------------------
    # Run trained Random Forest
    # --------------------------------------------------------

    try:

        rul = predict_rul(
            input_data
        )

    except Exception as e:

        print(
            "ERROR DURING RUL PREDICTION:",
            e
        )

        return render_template(
            "index.html",

            prediction=None,

            health_status=None,

            default_values=input_data,

            rul_history=load_rul_history()
        )

    # --------------------------------------------------------
    # Determine machine health
    # --------------------------------------------------------

    health_status = get_health_status(
        rul
    )

    # --------------------------------------------------------
    # Save prediction
    # --------------------------------------------------------

    save_rul_history(
        rul,
        health_status
    )
    save_telemetry_history(
    input_data,
    rul,
    health_status
)

    # --------------------------------------------------------
    # Load updated history
    # --------------------------------------------------------

    rul_history = load_rul_history()
    telemetry_history = load_telemetry_history()

    # --------------------------------------------------------
    # IMPORTANT:
    # After prediction, load NEXT real dataset row.
    #
    # This means the next prediction will not repeatedly
    # use the same first-row sensor values.
    # --------------------------------------------------------

    next_row = get_next_dataset_row()

    default_values = load_values_for_row(
        next_row
    )

    print(
        f"Next dashboard dataset row: {next_row}"
    )

    print(
        f"Predicted RUL: {round(float(rul), 2)}"
    )

    print(
        f"Health: {health_status}"
    )

    print("======================================\n")

    # --------------------------------------------------------
    # Return updated dashboard
    # --------------------------------------------------------

    return render_template(
        "index.html",

        prediction=round(
            float(rul),
            2
        ),

        health_status=health_status,

        default_values=default_values,

        rul_history=rul_history,

        telemetry_history=telemetry_history

        
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )