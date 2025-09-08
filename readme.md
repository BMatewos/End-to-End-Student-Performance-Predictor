#  End-to-End Student Performance Predictor

## Overview
This project demonstrates a **complete machine learning pipeline** for predicting student performance.  
It is designed as an end-to-end ML workflow, from raw data to a trained, ready-to-deploy model.

---

## Features
- **Data Ingestion**: Reading and validating raw data
- **Data Transformation**: Cleaning, preprocessing, feature engineering
- **Model Training**: CatBoost, XGBoost, and other ML models
- **Evaluation**: Accuracy ~ **0.88** on test set
- **Logging & Exception Handling**: Custom utilities for debugging and error tracking
- **Deployment-Ready Structure**: Modularized code (`src/components`, `src/pipeline`, etc.)

---

## Tech Stack
- **Python 3.10+**
- **Libraries**: pandas, numpy, scikit-learn, catboost, xgboost
- **Tools**: Git, GitHub, Conda
- **Project Organization**: Follows modular pipeline design

---

## 📂 Project Structure
End-to-End-Student-Performance-Predictor/
│── artifacts/ # saved models, outputs (ignored in git)
│── notebook/ # Jupyter notebooks (EDA & experiments)
│── src/
│ ├── components/ # data_ingestion, transformation, model_trainer
│ ├── pipeline/ # prediction pipeline
│ ├── utils.py # helper functions
│ ├── logger.py # logging setup
│ └── ...
│── requirements.txt # dependencies
│── setup.py # package setup
│── .gitignore # ignored files (artifacts, logs, DS_Store, etc.)
│── README.md # project documentation


![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/BMatewos/End-to-End-Student-Performance-Predictor)
![License](https://img.shields.io/badge/license-MIT-green)
