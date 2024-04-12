import os
import zipfile
from dataclasses import dataclass
from google_drive_downloader import GoogleDriveDownloader as gdd

@dataclass
class DataIngestionConfig:
    zip_file_path: str
    unzip_file_path: str

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

if __name__ == "__main__":
    zip_path = "E:/codes/mlops/project_ddd/data/raw/cotton.zip"  # Provide the path to save the zip file (without file name)
    unzip_path = "E:/codes/mlops/project_ddd/data" # Provide the path to extract the contents of the zip file
    drive_link = "1OK6L-QJltohBPfuW-vq_4bpkpeqqyQIA"  # Google Drive file ID

    data_ingestion_config = DataIngestionConfig(zip_path, unzip_path)
    data_ingestion = DataIngestion(data_ingestion_config)

    # Download zip file from Google Drive
    # data_ingestion.download_zip()

    # Unzip the downloaded file
    data_ingestion.unzip_file()
