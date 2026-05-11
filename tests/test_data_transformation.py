import os
import pandas as pd

from src.components.data_transformation import DataTransformation


def test_initiate_data_transformation_creates_processed_arrays(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    frame = pd.DataFrame(
        {
            "gender": ["female", "male", "female", "male"],
            "race_ethnicity": ["group A", "group B", "group C", "group D"],
            "parental_level_of_education": [
                "bachelor's degree",
                "some college",
                "master's degree",
                "associate's degree",
            ],
            "lunch": ["standard", "free/reduced", "standard", "free/reduced"],
            "test_preparation_course": ["none", "completed", "none", "completed"],
            "writing_score": [72, 60, 78, 58],
            "reading_score": [74, 62, 80, 59],
            "math_score": [70, 65, 75, 60],
        }
    )

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    frame.iloc[:2].to_csv(train_path, index=False)
    frame.iloc[2:].to_csv(test_path, index=False)

    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
        str(train_path), str(test_path)
    )

    assert train_arr.shape[1] == test_arr.shape[1]
    assert train_arr.shape[0] == 2
    assert test_arr.shape[0] == 2
    assert os.path.exists(preprocessor_path)
