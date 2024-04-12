import os
import zipfile
from dataclasses import dataclass
from cnnClassifier.entity.config_entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    # def download_zip(self):
    #     try:
    #         os.makedirs(self.config.zip_file_path, exist_ok=True)
    #         gdd.download_file_from_google_drive(file_id=self.config.drive_link, dest_path=self.config.zip_file_path)
    #     except Exception as e:
    #         raise e

    def unzip_file(self):
        try:
            with zipfile.ZipFile(self.config.zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.unzip_file_path)
        except Exception as e:
            raise e

