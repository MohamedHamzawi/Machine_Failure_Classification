from pathlib import Path
import joblib
import pandas as pd

Required_input_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
def load_model_artifacts(
    model_path = "models/final_model_pipeline.pkl",
    config_path="models/final_model_config.pkl"
):
    """
    Load the saved model pipeline and model configuration.

    The saved pipeline already includes:
    - feature engineering
    - preprocessing
    - trained classifier
    """
    project_root = Path(__file__).resolve().parents[1]
    model_path = Path(model_path)
    config_path = Path(config_path)
    
    if not model_path.is_absolute():
        model_path = project_root / model_path

    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    model = joblib.load(model_path)
    config = joblib.load(config_path)

    return model, config

def validate_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the input dataframe contains the required raw columns.
    """
    
    missing_columns = [col for col in Required_input_columns if col not in input_df.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required input columns: {missing_columns}"
        )
    return input_df[Required_input_columns].copy()

def predict_failure(input_df: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """
    Predict machine failure probabilities and final class labels.
    """
    
    clean_input = validate_input_data(input_df)
    
    failure_proba = model.predict_proba(clean_input)[:,1]
    predicted_failure = (failure_proba >= threshold).astype(int)
    
    results = clean_input.copy()
    results["failure_probability"] = failure_proba
    results["predicted_failure"] = predicted_failure
    
    return results

def predict_single_machine(
    machine_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    model,
    threshold: float
) -> pd.DataFrame:
    """
    Predict failure for a single machine observation.
    """
    input_df = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])
    return predict_failure(input_df, model, threshold)