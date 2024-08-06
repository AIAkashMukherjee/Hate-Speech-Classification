import numpy as np 
import pandas as pd
import re
import sys
from nltk.corpus import stopwords
import os
import nltk
import string
from src.logger.logging import logger
from src.exception.exception import CustomException
from src.entity.config_entity import DatatransformationCOnfig
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts


class DataTransformation:
    def __init__(self,data_transformation_config:DatatransformationCOnfig,data_ingestion_artifact:DataIngestionArtifacts) -> None:
        self.dt_config=data_transformation_config
        self.data_ingestion_artifact=data_ingestion_artifact

    def raw_data_cleaninig(self):
        try:
            df=pd.read_csv(self.data_ingestion_artifact.raw_data_file_path)
            df.drop(self.dt_config.drop_columns,axis=self.dt_config.Axis,inplace=self.dt_config.Inplace)

            df['class']=df['class'].replace({0:1})
            logger.info(f"Exited the raw_data_cleaning function and returned the raw_data {df}")
            return df
        except Exception as e:
            raise CustomException(e,sys)    
        
    def concat_data_cleaning(self,words):
        try:
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if words not in stopword]
            words=" ".join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words=" ".join(words)
            

            return words 

        except Exception as e:
            raise CustomException(e,sys)    
        
    def initate_data_transformation(self):
        try:
            logger.info('Entered into initate data transformation')
            df=self.raw_data_cleaninig()
            logger.info("Entered into the concat_data_cleaning function")
            df[self.dt_config.tweet]=df[self.dt_config.tweet].apply(self.concat_data_cleaning)
            os.makedirs(self.dt_config.DataTransformation_Artifact_Dir, exist_ok=True)

            df.to_csv(self.dt_config.Transformed_File_path,index=False,header=True)
            data_transformation_artifact=DataTransformationArtifacts(
                transformed_data_path=self.dt_config.Transformed_File_path
            )
            logger.info("returning the DataTransformationArtifacts")
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e,sys)    
