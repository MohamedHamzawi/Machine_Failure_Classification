import pandas as pd

def get_feature_importance_from_pipeline(pipeline):
    """
    Extract feature importances from a fitted sklearn pipeline.

    Assumes the pipeline has:
    - 'preprocessor' step
    - 'model' step with feature_importances_
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df