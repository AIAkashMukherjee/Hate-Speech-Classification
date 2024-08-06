import os
import sys
import keras
import pickle
from src.logger.logging import logger
from src.constants import *
from src.exception.exception import CustomException
from keras.utils import pad_sequences # type: ignore
from src.config.gcloud_syncer import GCloudSync
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import DatatransformationCOnfig
from src.entity.artifact_entity import DataIngestionArtifacts

class PredictionPipeline:
    def __init__(self) -> None:
        self.bucket_name = BUCKET_NAME
        self.model_name = MODEL_NAME
        self.model_path = os.path.join('artifacts', 'PredictModel')
        self.gcloud = GCloudSync()
        self.data_transformation = DataTransformation(
            data_transformation_config=DatatransformationCOnfig,
            data_ingestion_artifact=DataIngestionArtifacts
        )

    def get_model_from_gcloud(self):
        """
        Method Name :   get_model_from_gcloud
        Description :   This method to get best model from google cloud storage
        Output      :   best_model_path
        """
        logger.info("Entered the get_model_from_gcloud method of PredictionPipeline class")
        try:
            # Loading the best model from gcloud bucket
            os.makedirs(self.model_path, exist_ok=True)
            logger.info(f"Syncing model from bucket {self.bucket_name} to path {self.model_path}")
            self.gcloud.sync_folder_from_gcloud(self.bucket_name, self.model_name, self.model_path)
            best_model_path = os.path.join(self.model_path, self.model_name)
            logger.info(f"Best model path: {best_model_path}")

            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"Model file not found at {best_model_path}")
            
            logger.info("Exited the get_model_from_gcloud method of PredictionPipeline class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, best_model_path, text):
        """Run prediction on the given text"""
        logger.info("Running the predict function")
        try:
            load_model = keras.models.load_model(best_model_path)
            logger.info(f"Loaded model from {best_model_path}")

            tokenizer_path = 'tokenizer.pickle'
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

            with open(tokenizer_path, 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            logger.info("Loaded tokenizer")

            text = self.data_transformation.concat_data_cleaning(text)
            text = [text]
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=300)
            pred = load_model.predict(padded)

            if pred < 0.5:
                return "hate and abusive"
            else:
                return "no hate"
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self, text):
        logger.info('Entered run_pipeline method')
        try:
            best_model_path = self.get_model_from_gcloud()
            predicted_text = self.predict(best_model_path, text)
            logger.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys)
