import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


list_of_files = [
    f"src/components/__init__.py",
    f"src/components/data_ingestion.py",
    f"src/components/data_transformation.py",
    f"src/components/model_trainer.py",
    f"src/components/model_evaluation.py",
    f"src/config/__init__.py",
    f"src/config/gcloud_syncer.py",
    f"src/constants/__init__.py",
    f"src/entity/__init__.py",
    f"src/entity/config_entity.py",
    f"src/entity/artifact_entity.py",
    f"src/exception/__init__.py",
    f"src/logger/__init__.py",
    f"src/pipeline/__init__.py",
    f"src/pipeline/train_pipeline.py",
    f"src/pipeline/prediction_pipeline.py",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",
    'notebook/experiments.ipynb',

]


for filepath in list_of_files:
    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")