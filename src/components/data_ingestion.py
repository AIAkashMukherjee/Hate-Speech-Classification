from dataclasses import dataclass
from src.config.gcloud_syncer import GCloudSync
import os
import sys
from zipfile import ZipFile
from src.logger.logging import logger
from src.exception.exception import CustomException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config
        self.g_cloud = GCloudSync()

    def verify_file_path(self, file_path):
        if os.path.exists(file_path):
            logger.info(f"File found: {file_path}")
        else:
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

    def get_data_from_g_cloud(self):
        try:
            logger.info("Entered the get_data_from_gcloud method of Data ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            logger.info(f"Syncing from bucket: {self.data_ingestion_config.BUCKET_NAME}, file: {self.data_ingestion_config.ZIP_FILE_NAME}")
            self.g_cloud.sync_folder_from_gcloud(
                gcp_bucket_url=self.data_ingestion_config.BUCKET_NAME,
                filename=self.data_ingestion_config.ZIP_FILE_NAME,
                destination=self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR
            )

            zip_file_path = self.data_ingestion_config.ZIP_FILE_PATH
            logger.info(f"Expected zip file path: {zip_file_path}")
            self.verify_file_path(zip_file_path)

            logger.info("Exited the get_data_from_gcloud method of Data ingestion class")

        except Exception as e:
            raise CustomException(e, sys)

    def unzip_and_clean(self):
        logger.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            zip_file_path = self.data_ingestion_config.ZIP_FILE_PATH
            logger.info(f"Unzipping file: {zip_file_path}")
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logger.info(f"Contents of the directory after unzipping: {os.listdir(self.data_ingestion_config.ZIP_FILE_DIR)}")
            logger.info("Unzipped file successfully")
            logger.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logger.info("Entered the initiate_data_ingestion method of Data ingestion class")

        try:
            self.get_data_from_g_cloud()
            logger.info("Fetched the data from gcloud bucket")
            _, raw_data_file_path = self.unzip_and_clean()
            logger.info("Unzipped file and split into train and valid")

            data_ingestion_artifacts = DataIngestionArtifacts(
                # imbalance_data_file_path= imbalance_data_file_path,
                raw_data_file_path = raw_data_file_path
            )

            logger.info("Exited the initiate_data_ingestion method of Data ingestion class")
            logger.info(f"Data ingestion artifact: {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
