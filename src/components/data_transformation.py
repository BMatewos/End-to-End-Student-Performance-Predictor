import sys
import os
from dataclass import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,standardScaler 

from src.exception import CustomException
from src.logger import logging
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts',"preprocessor.pkl")
class DataTransformation:
    def__init__(self):