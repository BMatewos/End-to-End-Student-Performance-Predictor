#  End-to-End Student Performance Predictor

## Overview
This project demonstrates a **complete machine learning pipeline** for predicting student performance.  
It is designed as an **end-to-end ML workflow**, starting from raw data ingestion to training, evaluation, and finally deployment as a Flask web application with a simple UI for predictions.

---

## Features
- **Data Ingestion**: Reads and validates raw student data  
- **Data Transformation**: Cleans, preprocesses, and performs feature engineering  
- **Model Training**: Trains models such as CatBoost, XGBoost, and others  
- **Evaluation**: Achieves ~ **0.88 accuracy** on the test set  
- **Logging & Exception Handling**: Custom utilities for easier debugging and error tracking  
- **Deployment-Ready Structure**: Modularized code (`src/components`, `src/pipeline`, etc.) for scalability  
- **Flask Web App**: Interactive web interface for entering student details and predicting math scores in real-time  


---

##  Tech Stack
- **Language**: Python 3.10+  
- **Libraries**: pandas, numpy, scikit-learn, catboost, xgboost, flask  
- **Tools**: Git, GitHub, Conda, Jupyter Notebook  
- **Model Serving**: Flask web app with HTML templates  
- **Project Organization**: Modular ML pipeline design (data ingestion → transformation → training → prediction)  




##  Project Structure

```
End-to-End-Student-Performance-Predictor/
├── artifacts/             # saved models, outputs (ignored in git)
├── notebook/              # Jupyter notebooks (EDA & experiments)
├── src/
│   ├── components/        # data_ingestion, transformation, model_trainer
│   ├── pipeline/          # prediction pipeline
│   ├── utils.py           # helper functions
│   ├── logger.py          # logging setup
│   └── ...
├── requirements.txt       # dependencies
├── setup.py               # package setup
├── .gitignore             # ignored files (artifacts, logs, DS_Store, etc.)
└── README.md              # project documentation
```



![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/BMatewos/End-to-End-Student-Performance-Predictor)
![License](https://img.shields.io/badge/license-MIT-green)
