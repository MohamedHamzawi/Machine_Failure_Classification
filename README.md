# Predictive Maintenance System

A machine learning project for predicting machine failures using industrial sensor and operational data.

## Project Goal

The goal of this project is to build a predictive maintenance pipeline capable of identifying potential machine failures before they happen.

This project focuses on:
- data preprocessing
- exploratory data analysis
- feature engineering
- machine learning modeling
- experiment tracking
- model evaluation

---

## Dataset

Dataset used:
AI4I 2020 Predictive Maintenance Dataset

Source:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- MLflow
- Jupyter Notebook

---

### Main Features

| Column | Meaning |
|---|---|
| Type | Product quality type: L, M, or H |
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Process operating temperature |
| Rotational speed [rpm] | Machine rotational speed |
| Torque [Nm] | Torque applied by the machine |
| Tool wear [min] | Accumulated tool usage time |
| Machine failure | Target variable |

### Columns Excluded From Modeling

Some columns were excluded because they are identifiers or can cause data leakage:

| Column | Reason |
|---|---|
| UDI | Unique ID, not useful for prediction |
| Product ID | Identifier, not useful for generalization |
| TWF, HDF, PWF, OSF, RNF | Failure-type labels that leak information about the target |

---

## Project Structure

Current project structure:

```text
Predictive-Maintenance-Project/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_threshold_tuning.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── reports/
│   └── figures/
│
├── models/
│
├── README.md
├── requirements.txt
└── .gitignore
