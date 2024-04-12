from cnnClassifier.components.base_model_prepare import PrepareBaseModel
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.components.data_transform import DataTransform
from cnnClassifier.components.model_evalution import EvaluationModel

from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.entity.config_entity import DataTransformConfig
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dropout, Input, Flatten, Dense, MaxPooling2D


import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dropout, Input, Flatten, Dense, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataclasses import dataclass
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import json
from typing import Iterator
import time
import os
import matplotlib.pyplot as plt
import zipfile
from dataclasses import dataclass





def pipeline():
    # zip_path = "E:/codes/mlops/project_ddd/data/raw/cotton.zip"  # Provide the path to save the zip file (without file name)
    # unzip_path = "E:/codes/mlops/project_ddd/data" # Provide the path to extract the contents of the zip file

    # data_ingestion_config = DataIngestionConfig(zip_path, unzip_path)
    # data_ingestion = DataIngestion(data_ingestion_config)
    # data_ingestion.unzip_file()

    # Define paths
    train_path = "data/cotton/train"
    test_path = "data/cotton/test"

    # Create config
    config = DataTransformConfig(train_path, test_path)

    # Initialize DataIngestion with the config
    data_ingestion = DataTransform(config)
    
  

    # Get data paths
    train_data, validation_data, test_data = data_ingestion.data_path()

    print("Data is Successfully Transform")

    model_saved_path = "artifacts\models\model1.h5"
    learning_rate=0.05
    batchsize=8
    pretrained_model=InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(80, 80, 3)))
    base_model_obj=PrepareBaseModel(PrepareBaseModelConfig(train_data=train_data,validation_data=validation_data,test_data=test_data,params_learning_rate = learning_rate,model_saved_path = model_saved_path,params_batchsize = batchsize))
    base_model=base_model_obj.getbasemodel(pretrained_model)
    hist=base_model_obj.save_model(base_model)

    print("Model is successfully prepared")
    

    folder_path="artifacts/evaluation"
    Model_name="Inception"
    Evaluation_obj=EvaluationModel(EvaluationConfig(hist=hist,folder_path=folder_path,model_name=Model_name))
    Evaluation_obj.save_json()
    Evaluation_obj.save_score()

    print("Scores are evalutated")


if __name__=="__main__":
    pipeline()

