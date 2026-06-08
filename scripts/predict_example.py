from src.inference import load_model_artifacts, predict_single_machine

def main():
    model, config = load_model_artifacts()
    
    threshold = config["threshold"]
    
    prediction = predict_single_machine(
        machine_type="M",
        air_temperature=298.1,
        process_temperature=308.6,
        rotational_speed=1551,
        torque=42.8,
        tool_wear=120,
        model=model,
        threshold=threshold 
    )
    
    print(prediction)
    
if __name__ == "__main__":
    main()