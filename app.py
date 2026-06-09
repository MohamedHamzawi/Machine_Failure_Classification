from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.feature_engineering import add_engineered_features


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "final_model_pipeline.pkl"
CONFIG_PATH = PROJECT_ROOT / "models" / "final_model_config.pkl"


st.set_page_config(
    page_title="Predictive Maintenance Classifier",
    page_icon="⚙️",
    layout="centered"
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)

    if CONFIG_PATH.exists():
        config = joblib.load(CONFIG_PATH)
    else:
        config = {}

    return model, config


def get_threshold_from_config(config, default=0.5):
    """
    Safely extract the best threshold from the saved config file.
    Works if config is a dictionary and contains a threshold key.
    """

    possible_keys = [
        "best_threshold",
        "threshold",
        "selected_threshold",
        "final_threshold",
    ]

    if isinstance(config, dict):
        for key in possible_keys:
            if key in config:
                return float(config[key])

    return default


model, config = load_artifacts()
best_threshold = get_threshold_from_config(config, default=0.5)


st.title("⚙️ Predictive Maintenance Machine Failure Prediction")

st.write(
    """
    This app predicts whether a machine is likely to fail based on operating
    conditions such as temperature, rotational speed, torque, tool wear, and machine type.
    """
)

st.sidebar.header("Prediction Settings")

threshold = st.sidebar.slider(
    "Failure classification threshold",
    min_value=0.0,
    max_value=1.0,
    value=float(best_threshold),
    step=0.01
)

st.sidebar.write(f"Saved best threshold: `{best_threshold:.2f}`")

st.header("Enter Machine Operating Conditions")

machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

air_temp = st.number_input(
    "Air temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=298.0,
    step=0.1
)

process_temp = st.number_input(
    "Process temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=308.0,
    step=0.1
)

rot_speed = st.number_input(
    "Rotational speed [rpm]",
    min_value=0,
    max_value=5000,
    value=1500,
    step=10
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.1
)

tool_wear = st.number_input(
    "Tool wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)


input_data = pd.DataFrame({
    "Type": [machine_type],
    "Air temperature [K]": [air_temp],
    "Process temperature [K]": [process_temp],
    "Rotational speed [rpm]": [rot_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear],
})


st.subheader("Raw Input Data")
st.dataframe(input_data)


with st.expander("Show engineered features"):
    engineered_data = add_engineered_features(input_data)
    st.dataframe(engineered_data)


if st.button("Predict Machine Failure"):
    failure_probability = model.predict_proba(input_data)[:, 1][0]
    prediction = int(failure_probability >= threshold)

    st.subheader("Prediction Result")

    st.metric(
        label="Failure Probability",
        value=f"{failure_probability:.2%}"
    )

    if prediction == 1:
        st.error("Prediction: Machine failure risk detected.")
    else:
        st.success("Prediction: No machine failure detected.")

    st.write(f"Selected threshold: `{threshold:.2f}`")

    st.info(
        """
        The model receives the raw input data. Engineered features are added automatically
        inside the saved machine learning pipeline before prediction.
        """
    )