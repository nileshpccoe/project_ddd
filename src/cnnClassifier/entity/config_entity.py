import os 
import sys
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    zip_file_path: str
    unzip_file_path: str


@dataclass
class DataTransformConfig:
    train_path: str
    test_path: str

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    params_learning_rate: float
    model_saved_path: str
    params_batchsize :int
    train_data:any
    validation_data:any
    test_data:any

@dataclass
class EvaluationConfig:
    folder_path : str
    model_name : str
    hist: any