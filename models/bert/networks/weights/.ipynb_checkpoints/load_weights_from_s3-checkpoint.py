import requests 
import io
import json 
import torch
import logging 
import time 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__ )

def load_weights_from_s3():
    '''pytorch_model.binの読み込みと取得'''
    
    with open("./networks/weights/weights.json", "r") as f:
      res = json.load(f)
    uri = res["URL"]
    logger.info(f"loading s3 from {uri}")
    start = time.time()
    res = requests.get(uri)
    end = time.time()
    logger.info(f"complete read uri duraion in senconds: {end-start}")
    loaded_model_dict = torch.load(io.BytesIO(res.content))
    return loaded_model_dict
    
        

