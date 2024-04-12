import os 
import sys
from cnnClassifier import logger
from cnnClassifier.config.config_entity import DataTransformConfig
import zipfile




class DataTransform:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def data_path(self):
        try:
            train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=0.2, shear_range=0.2,
                                               zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
                                               validation_split=0.2)

            train_data = train_datagen.flow_from_directory(self.config.train_path,
                                                           target_size=(80, 80), batch_size=8, class_mode='categorical', subset='training')

            validation_data = train_datagen.flow_from_directory(self.config.train_path,
                                                                target_size=(80, 80), batch_size=8, class_mode='categorical', subset='validation')

            test_datagen = ImageDataGenerator(rescale=1./255, rotation_range=0.2, shear_range=0.2, zoom_range=0.2,
                                              width_shift_range=0.2, height_shift_range=0.2)

            test_data = test_datagen.flow_from_directory(self.config.test_path,
                                                          target_size=(80, 80), batch_size=8, class_mode='categorical')
        except Exception as e:
            raise e
        return train_data, validation_data, test_data

if __name__=="__main__":
        # Define paths
        train_path = "/content/drive/MyDrive/DDD_data/train_data"
        test_path = "/content/drive/MyDrive/DDD_data/test_data"

        # Create config
        config = DataTransformConfig(train_path, test_path)

        # Initialize DataIngestion with the config
        data_trasnform = DataTransform(config)

        # Get data paths
        train_data, validation_data, test_data = data_transform.data_path()    

