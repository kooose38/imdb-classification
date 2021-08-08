import joblib 
import boto3 
from db.uri import PROJECT_NAME
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__ )

def make_dump_prep(model, filepath: str="./pred/preprocessing/preprocessing_transformers.pkl"):
  joblib.dump(model, filepath)

def load_dump_prep(filepath: str="./pred/preprocessing/preprocessing_transformers.pkl"):
  prep = joblib.load(filepath)
  return prep 

def upload_s3_bucket(filename: str="./pred/preprocessing/preprocessing_transformers.pkl"):
    model_name = "transformers"
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(PROJECT_NAME)
    logger.info("upload file to s3 ....")
    s3_dir = filename.split("/")
    if len(s3_dir) < 3:
        return False 
    ################################################################## requests FIX
    bucket.upload_file(filename, f"{model_name}/{s3_dir.split("/")[2]}/{s3_dir.split("/")[3]}")
    word2index = "./pred/word2index.txt"
    bucket.upload_file(filename, f"{model_name}/{s3_dir.split("/")[2]}/{word2index.split("/")[2]}")