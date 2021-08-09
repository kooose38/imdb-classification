import joblib 
from db.uri import PROJECT_NAME
import logging 
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__ )

def make_dump_prep(model, filepath: str="./pred/preprocessing/preprocessing_bert.pkl"):
  joblib.dump(model, filepath)

def load_dump_prep(filepath: str="./pred/preprocessing/preprocessing_bert.pkl"):
  prep = joblib.load(filepath)
  return prep

def upload_s3_bucket(filename: str="./pred/preprocessing/preprocessing_bert.pkl"):
    model_name = "bert"
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(PROJECT_NAME)
    logger.info("upload file to s3 ... ")
    s3_dir = filename.split("/")
    if len(s3_dir) < 4:
        return False
    bucket.upload_file(filename, f"{model_name}/{s3_dir[-2]}/{s3_dir[-1]}")
    logger.info("complete upload files !!!")