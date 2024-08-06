# type: ignore
import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logger
from src.exception.exception import CustomException
from keras.utils import pad_sequences 
from src.constants import *
from src.config.gcloud_syncer import GCloudSync
from sklearn.metrics import confusion_matrix
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifact, DataTransformationArtifacts

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifact,
                 data_transformation_artifacts: DataTransformationArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GCloudSync()

    def get_best_model_from_gcloud(self):
        try:
            logger.info('Getting Model from Gcloud')
            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            self.gcloud.sync_folder_from_gcloud(
                self.model_evaluation_config.BUCKET_NAME,
                self.model_evaluation_config.MODEL_NAME,
                self.model_evaluation_config.BEST_MODEL_DIR_PATH
            )

            best_model_path = os.path.join(
                self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                self.model_evaluation_config.MODEL_NAME
            )

            logger.info("Exited the get_best_model_from_gcloud method of Model Evaluation class")
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys)    

    def evaluate(self):
        try:
            logger.info("Entering into to the evaluate function of Model Evaluation class")
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col=0)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col=0)

            tokenizer_path = 'tokenizer.pickle'
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as handle:
                    tokenizer = pickle.load(handle)
            else:
                raise FileNotFoundError(f"Error: The tokenizer file does not exist at {tokenizer_path}")

            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            x_test = x_test['tweet'].astype(str).squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_LEN)

            accuracy = load_model.evaluate(test_sequences_matrix, y_test)
            logger.info(f"The test accuracy is {accuracy}")

            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = [1 if prediction[0] >= 0.5 else 0 for prediction in lstm_prediction]

            conf_matrix = confusion_matrix(y_test, res)
            logger.info(f"The confusion matrix is {conf_matrix}")

            return accuracy
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        logger.info("Initiate Model Evaluation")
        try:
            trained_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            tokenizer_path = 'tokenizer.pickle'
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as handle:
                    load_tokenizer = pickle.load(handle)
            else:
                raise FileNotFoundError(f"Error: The tokenizer file does not exist at {tokenizer_path}")

            trained_model_accuracy = self.evaluate()

            best_model_path = self.get_best_model_from_gcloud()

            if not os.path.isfile(best_model_path):
                is_model_accepted = True
                logger.info("GCloud storage model is not found. Currently trained model accepted.")
            else:
                best_model = keras.models.load_model(best_model_path)
                best_model_accuracy = self.evaluate()

                is_model_accepted = trained_model_accuracy >= best_model_accuracy

                logger.info("Comparing loss between best_model_loss and trained_model_loss")
                logger.info(f"Trained model {'accepted' if is_model_accepted else 'not accepted'}")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logger.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys)
