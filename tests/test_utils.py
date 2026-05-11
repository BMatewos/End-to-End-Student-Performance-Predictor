import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.utils import load_object, save_object, evaluate_models


def test_save_and_load_object(tmp_path):
    data = {"a": 1, "b": 2}
    file_path = tmp_path / "artifact.pkl"

    save_object(str(file_path), data)
    loaded = load_object(str(file_path))

    assert loaded == data


def test_evaluate_models_simple():
    X_train = np.arange(20).reshape(10, 2)
    y_train = np.arange(10)
    X_test = np.arange(20, 28).reshape(4, 2)
    y_test = np.arange(10, 14)

    models = {"Linear Regression": LinearRegression()}
    params = {"Linear Regression": {}}

    report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)

    assert "Linear Regression" in report
    assert isinstance(report["Linear Regression"], float)
