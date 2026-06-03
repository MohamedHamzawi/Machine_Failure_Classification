import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

def evaluate_classifier(model,X,y,model_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:,1]

    print(f"====={model_name}=====")
    print(classification_report(y,y_pred,zero_division=0))
    cm = confusion_matrix(y,y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    results ={
        "Model": model_name,
        "Precision": precision_score(y,y_pred,zero_division=0),
        "Recall":recall_score(y,y_pred,zero_division=0),
        "F1_Score":f1_score(y,y_pred,zero_division=0),
        "ROC_AUC":roc_auc_score(y,y_proba),
        "PR_AUC":average_precision_score(y,y_proba)
    }
    
    return results

def evaluate_with_threshold(model,X,y,threshold,model_name):
    y_proba = model.predict_proba(X)[:,1]
    y_pred = (y_proba >= threshold).astype(int)
    
    print(f"===== {model_name} at threshold {threshold:.3f} =====")
    print(classification_report(y, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    results = {
        "Model":model_name,
        "Threshold": threshold,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1 Score": f1_score(y, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, y_proba),
        "PR-AUC": average_precision_score(y, y_proba)
    }
    
    return results

def results_to_dataframe(results_list):
    
    #Convert list of result dictionaries into a pandas DataFrame.
    
    return pd.DataFrame(results_list)