from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    # imbalance_data_file_path: str
    raw_data_file_path: str

@dataclass
class DataTransformationArtifacts:
    transformed_data_path:str

@dataclass
class ModelTrainerArtifact:
    trained_model_path:str
    x_test_path:str    
    y_test_path:str


@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool 

@dataclass
class ModelPusherArtifacts:
    bucket_name: str    