import json 
import logging 
import requests 
from db.uri import MODELDB_URI, PROJECT_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__ )

def add_project(project_name: str, description: str=None):
    uri = MODELDB_URI().uri + "projects"
    headers = MODELDB_URI().headers 
    data = {
        "project_name": PROJECT_NAME,
        "description": description
    }
    res = requests.post(uri, data=json.dumps(data), headers=headers)
    if res.status_code == 200:
        logger.info(f"response: {res.text}")
    else: 
        logger.info(f"response: {res.status_code}")
        
    
def add_model(model_id: str, model_name: str, description: str=None):
    uri_ = MODELDB_URI().uri + f"projects/name/{PROJECT_NAME}"
    res_get_id = requests.get(uri_)
    project_id = json.loads(res_get_id.text)["project_id"]
    
    uri__ = MODELDB_URI().uri + "models"
    headers = MODELDB_URI().headers
    data = {
        "model_id": model_id,
        "project_id": project_id,
        "model_name": model_name,
        "description": description
    }
    res = requests.post(uri__, data=json.dumps(data), headers=headers)
    if res.status_code == 200:
        logger.info(f"response: {res.text}")
    else:
        logger.info(f"response: {res.status_code} {res.text}")
    
    
def add_expriments(model_id: str,
                   model_version_id: str,
                   parameters: dict,
                   training_dataset: str,
                   validation_dataset: str=None,
                   test_dataset: str=None,
                   evaluations: dict=None, 
                   artifact_file_paths: dict=None):
    uri = MODELDB_URI().uri+"experiments"
    headers = MODELDB_URI().headers 
    data = {
        "model_id": model_id,
        "model_version_id": model_version_id,
        "parameters": parameters,
        "training_dataset": training_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "evaluations": evaluations,
        "artifact_file_paths": artifact_file_paths
    }
    res = requests.post(uri, data=json.dumps(data), headers=headers)
    if res.status_code == 200:
        logger.info(f"response: {res.text}")
    else:
        logger.info(f"response: {res.status_code} {res.text} ")
    
    