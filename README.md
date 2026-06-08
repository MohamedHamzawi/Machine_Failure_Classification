# Predictive Maintenance Machine Failure Classification

## Project Overview

This project is a machine learning project for **predictive maintenance** using the **AI4I 2020 Predictive Maintenance Dataset**.

The goal is to predict whether a machine will fail based on machine operating conditions such as air temperature, process temperature, rotational speed, torque, tool wear, and machine type.

This is a **binary classification problem**:

* `0` → No machine failure
* `1` → Machine failure

The project follows a professional machine learning workflow including:

* Data understanding
* Exploratory data analysis
* Feature engineering
* Baseline modeling
* Model improvement
* Threshold tuning
* Hyperparameter tuning
* Model interpretation
* Inference
* Reporting

---

## Dataset

The dataset used is the **AI4I 2020 Predictive Maintenance Dataset**.

### Main Features

| Column                    | Description                   |
| ------------------------- | ----------------------------- |
| `Type`                    | Product quality type/category |
| `Air temperature [K]`     | Ambient air temperature       |
| `Process temperature [K]` | Process temperature           |
| `Rotational speed [rpm]`  | Machine rotational speed      |
| `Torque [Nm]`             | Torque applied by the machine |
| `Tool wear [min]`         | Tool wear time                |

### Target

| Column            | Description                                         |
| ----------------- | --------------------------------------------------- |
| `Machine failure` | Binary target indicating whether the machine failed |

### Failure Type Columns

The dataset also contains failure-type columns:

* `TWF`
* `HDF`
* `PWF`
* `OSF`
* `RNF`

These columns were used for dataset understanding and analysis, but they were not used as model input features because they describe failure outcomes and would cause data leakage.

---

## Project Structure

```text
PREDICTIVE MAINTAINANCE PROJECT/
│
├── data/
│   └── raw/ai4i2020.csv
│
├── models/
│   ├── final_model_config.pkl
│   └── final_model_pipeline.pkl
│
├── notebooks/
│   ├── Day2_Exploratory_data_analysis.ipynb
│   ├── Day3_baseline_modeling.ipynb
│   ├── Day4_model_improvement.ipynb
│   ├── Day5_project_optimization.ipynb
│   ├── Day6_Model_tuning_and_interpretation.ipynb
│   └── Day7_Inference_reporting.ipynb
│
├── reports/
│   ├── figures/
│   ├── baseline_models_results.csv
│   ├── day4_final_test_results.csv
│   ├── day4_threshold_comparison.csv
│   ├── day4_validation_model_results.csv
│   ├── day6_tuned_rfmodel_no_threshold.csv
│   ├── sample_predictions.csv
│   ├── threshold_metrics_comparison.csv
│   └── Predictive Maintenance Machine Failure Classification.docx
│
├── scripts/
│   └── predict_example.py
│
├── src/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── interpretation.py
│   ├── modeling.py
│   └── preprocessing.py
│
├── predictive_maintenance.egg-info/
├── venv/
├── README.md
└── requirements.txt
```

---

## Project Workflow

The project workflow is:

```text
Raw data
   ↓
Exploratory data analysis
   ↓
Feature engineering
   ↓
Preprocessing pipeline
   ↓
Baseline model training
   ↓
Model evaluation
   ↓
Threshold tuning
   ↓
Hyperparameter tuning
   ↓
Model interpretation
   ↓
Inference pipeline
   ↓
Reporting
```

---

## Work Completed

## Day 2 — Exploratory Data Analysis

Notebook:

```text
notebooks/Day2_Exploratory_data_analysis.ipynb
```

Completed:

* Loaded and inspected the dataset
* Checked dataset shape, columns, data types, and missing values
* Understood the target column `Machine failure`
* Explored target class imbalance
* Analyzed numerical feature distributions
* Created histograms, boxplots, and scatter plots
* Compared features against the target
* Analyzed relationships between:

  * Air temperature and process temperature
  * Torque and machine failure
  * Rotational speed and machine failure
  * Tool wear and machine failure
* Started connecting EDA observations to the predictive maintenance problem

Important observations:

* The dataset is imbalanced because failure cases are much fewer than non-failure cases.
* Accuracy alone is not enough for evaluating this project.
* Some machine operating conditions show different behavior between failed and non-failed machines.
* The relationship between process temperature and air temperature is important because the two temperatures are naturally connected.

---

## Day 3 — Baseline Modeling

Notebook:

```text
notebooks/Day3_baseline_modeling.ipynb
```

Completed:

* Defined the machine learning problem as binary classification
* Prepared the input features and target
* Built preprocessing steps for:

  * Numerical features
  * Categorical feature `Type`
* Used `ColumnTransformer`
* Trained baseline models
* Compared baseline model performance
* Saved baseline model results to:

```text
reports/baseline_models_results.csv
```

Models used:

* Logistic Regression
* Random Forest
* Extra Trees

Metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC

Important note:

Because the dataset is imbalanced, PR-AUC, precision, recall, and F1-score are more important than accuracy alone.

---

## Day 4 — Model Improvement and Threshold Tuning

Notebook:

```text
notebooks/Day4_model_improvement.ipynb
```

Completed:

* Improved the model evaluation workflow
* Used predicted probabilities instead of only default class predictions
* Applied threshold tuning
* Used precision-recall analysis
* Compared model results at different thresholds
* Saved validation model results
* Saved threshold comparison results
* Saved final test results from the Day 4 model improvement stage

Generated report files:

```text
reports/day4_validation_model_results.csv
reports/day4_threshold_comparison.csv
reports/day4_final_test_results.csv
reports/threshold_metrics_comparison.csv
```

Important concept:

The default classification threshold of `0.5` is not always the best choice for imbalanced classification.

In predictive maintenance, the threshold should be selected based on the cost of errors:

* False Negative: A real failure is missed
* False Positive: A healthy machine is flagged as risky

False negatives are usually more dangerous in maintenance because they may lead to unexpected downtime or equipment damage.

---

## Day 5 — Project Optimization

Notebook:

```text
notebooks/Day5_project_optimization.ipynb
```

Completed:

* Improved the project organization
* Moved reusable code into the `src/` folder
* Added reusable feature engineering code
* Added reusable preprocessing code
* Added reusable modeling and evaluation code
* Refactored the workflow to be cleaner and more professional
* Integrated feature engineering into the machine learning pipeline

Important project improvement:

Feature engineering was moved inside the pipeline using a structure similar to:

```python
Pipeline([
    ("features", FunctionTransformer(add_engineered_features)),
    ("preprocessor", preprocessor),
    ("model", model)
])
```

Why this matters:

The same feature engineering steps are now applied during:

* Training
* Validation
* Testing
* Future inference

This reduces the risk of inconsistency between model training and model usage.

---

## Day 6 — Model Tuning and Interpretation

Notebook:

```text
notebooks/Day6_Model_tuning_and_interpretation.ipynb
```

Completed:

* Used cross-validation for model evaluation
* Used `StratifiedKFold`
* Tuned the Random Forest model
* Used hyperparameter tuning
* Compared tuned model performance with previous results
* Saved tuned model results
* Started model interpretation
* Added interpretation-related code into the project

Generated result file:

```text
reports/day6_tuned_rfmodel_no_threshold.csv
```

Important observation:

The tuned Random Forest had different hyperparameters, but its PR-AUC was very close to the previous Random Forest result.

This suggests that the baseline Random Forest was already strong, or that the current features and model family reached a performance plateau.

Source files used or updated during this stage include:

```text
src/modeling.py
src/evaluation.py
src/interpretation.py
```

---

## Day 7 — Inference and Reporting

Notebook:

```text
notebooks/Day7_Inference_reporting.ipynb
```

Completed:

* Created an inference workflow
* Added inference logic into:

```text
src/inference.py
```

* Created a prediction example script:

```text
scripts/predict_example.py
```

* Generated sample predictions
* Saved sample predictions to:

```text
reports/sample_predictions.csv
```

* Created a project report document:

```text
reports/Predictive Maintenance Machine Failure Classification.docx
```

* Connected the trained model pipeline to a practical prediction workflow
* Organized results and reporting outputs inside the `reports/` folder

Important Day 7 outcome:

The project is no longer only a notebook-based modeling project. It now includes reusable inference code and a script that demonstrates how the trained model can be used for prediction.

This makes the project closer to a real machine learning project workflow.

---

## Source Code Modules

The project uses reusable Python modules inside the `src/` folder.

### `src/feature_engineering.py`

Contains feature engineering logic.

Main responsibility:

```python
add_engineered_features()
```

This function creates additional features used by the model pipeline.

---

### `src/preprocessing.py`

Contains preprocessing logic.

Main responsibility:

```python
build_preprocessor()
```

This builds the preprocessing pipeline for numerical and categorical features.

---

### `src/modeling.py`

Contains model-building logic.

Main responsibility:

```python
build_model_pipeline()
```

This builds the full machine learning pipeline.

---

### `src/evaluation.py`

Contains reusable model evaluation functions.

Main responsibilities include evaluating models using metrics such as:

* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion matrix

---

### `src/interpretation.py`

Contains model interpretation helper code.

This file is used to support interpretation and analysis of model behavior.

---

### `src/inference.py`

Contains inference logic for using the trained model pipeline on new data.

This separates prediction logic from notebooks and makes the project more reusable.

---

## Reports and Outputs

The project saves results inside the `reports/` folder.

Current saved outputs include:

| File                                                         | Purpose                                         |
| ------------------------------------------------------------ | ----------------------------------------------- |
| `baseline_models_results.csv`                                | Baseline model comparison                       |
| `day4_validation_model_results.csv`                          | Validation results from model improvement       |
| `day4_threshold_comparison.csv`                              | Threshold tuning comparison                     |
| `day4_final_test_results.csv`                                | Final test results from Day 4 model improvement |
| `threshold_metrics_comparison.csv`                           | Threshold metric comparison                     |
| `day6_tuned_rfmodel_no_threshold.csv`                        | Tuned Random Forest results                     |
| `sample_predictions.csv`                                     | Example predictions from inference workflow     |
| `Predictive Maintenance Machine Failure Classification.docx` | Project report document                         |

The `reports/figures/` folder is used to store project figures and visual outputs.

---

## Inference Example

A prediction example script is available at:

```text
scripts/predict_example.py
```

This script demonstrates how to use the trained pipeline for prediction.

Expected usage:

```bash
python scripts/predict_example.py
```

The inference workflow is supported by:

```text
src/inference.py
```

---

## Current Project Status

Current status: **Completed through Day 7**

Completed:

* Exploratory data analysis
* Baseline modeling
* Model improvement
* Threshold tuning
* Project code refactoring
* Feature engineering inside the pipeline
* Cross-validation
* Random Forest tuning
* Model interpretation
* Inference workflow
* Prediction example script
* Sample predictions
* Project report document

Still can be improved:

* Add exact final metrics directly into this README
* Add important figures from `reports/figures/`
* Add a short explanation of the selected threshold
* Add final model file name inside the `models/` section
* Clean notebook outputs before publishing
* Add installation instructions if missing
* Push the project to GitHub

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-link>
cd "PREDICTIVE MAINTAINANCE PROJECT"
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

### 5. Run the notebooks

Run the notebooks in order:

```text
Day2_Exploratory_data_analysis.ipynb
Day3_baseline_modeling.ipynb
Day4_model_improvement.ipynb
Day5_project_optimization.ipynb
Day6_Model_tuning_and_interpretation.ipynb
Day7_Inference_reporting.ipynb
```

### 6. Run inference example

```bash
python scripts/predict_example.py
```

---

## Project Summary

This project demonstrates a complete applied machine learning workflow for predictive maintenance.

It starts from data exploration and baseline modeling, then progresses into model improvement, threshold tuning, pipeline optimization, hyperparameter tuning, interpretation, inference, and reporting.

The final structure includes notebooks, reusable source code, result files, a prediction script, and a project report.
