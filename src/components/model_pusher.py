import sys
from src.logger.logging import logger
from src.exception.exception import CustomException
from src.config.gcloud_syncer import GCloudSync
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifacts


class ModelPusher:
    def __init__(self,model_pusher_config:ModelPusherConfig):
        self.model_pusher_config=model_pusher_config
        self.g_cloud=GCloudSync()

    def initate_model_pusher(self):
        """
            Method Name :   initiate_model_pusher
            Description :   This method initiates model pusher.

            Output      :    Model pusher artifact
        """
        logger.info('Entered into mdoel pusher class')

        try:
            self.g_cloud.sync_folder_to_gcloud(self.model_pusher_config.BUCKET_NAME,self.model_pusher_config.TRAINED_MODEL_PATH,self.model_pusher_config.MODEL_NAME)

            logger.info("Uploaded best model to gcloud storage")

            model_pusher_artifact=ModelPusherArtifacts(
                bucket_name=self.model_pusher_config.BUCKET_NAME
            )
            logger.info("Exited the initiate_model_pusher method of ModelTrainer class")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e,sys)    
