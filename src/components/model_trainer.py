
# type: ignore

import sys
import pickle
import pandas as pd
from src.logger.logging import logger
from src.constants import *
from src.exception.exception import CustomException
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer 
from keras.utils import pad_sequences
from src.entity.config_entity import ModelTraninerConfig
from src.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifacts
from src.ml.model import ModelArchitecture



class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifacts,model_trainer_config:ModelTraninerConfig) -> None:
        self.data_transformation_artifacts = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def spliting_data(self,csv_path):
        try:
            logger.info("Entered the spliting_data function")
            logger.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logger.info("Splitting the data into x and y")
            x = df[TWEET]
            y = df[LABEL]

            logger.info("Applying train_test_split on the data")
            x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state = 42)
            print(len(x_train),len(y_train))
            print(len(x_test),len(y_test))
            print(type(x_train),type(y_train))
            logger.info("Exited the spliting the data function")
            return x_train,x_test,y_train,y_test

        except Exception as e:
            raise CustomException(e, sys)   
        

    def tokenizing(self,x_train):
        try:
            logger.info("Applying tokenization on the data")
            x_train = x_train.astype(str).tolist()
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logger.info(f"converting text to sequences:")
            sequences_matrix = pad_sequences(sequences,maxlen=self.model_trainer_config.MAX_LEN)
            logger.info(f" The sequence matrix is: {sequences_matrix}")
            return sequences_matrix,tokenizer
        except Exception as e:
            raise CustomException(e, sys)   
        

    def initate_model_trainer(self):
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logger.info('Enterd into initate model trainer')

            x_train,x_test,y_train,y_test=self.spliting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_arch=ModelArchitecture()
            
            model=model_arch.get_model()

            sequences_matrix,tokenizer =self.tokenizing(x_train)


            logger.info("Entered into model training")
            model.fit(sequences_matrix, y_train, 
                        batch_size=self.model_trainer_config.BATCH_SIZE, 
                        epochs = self.model_trainer_config.EPOCH, 
                        validation_split=self.model_trainer_config.VALIDATION_SPLIT, 
                        )
            logger.info("Model training finished")
            with open('tokenizer.pickle','wb')as handle:
                pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)

            logger.info("saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifact(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH)

            logger.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts
    

        except Exception as e:
            raise CustomException(e,sys)    


