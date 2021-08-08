import os 

# EIP を取得していないので常時変更する
IP_ADDRESS = os.getenv("IP_ADDRESS", "13.115.40.91") 
PORT = os.getenv("PORT", "8000")
PROJECT_NAME = os.getenv("POJECT_NAME", "imdb-classification")

class MODELDB_URI:
    uri = f"http://{IP_ADDRESS}:{str(PORT)}/v0.1/api/"
    headers = {'content-type': 'application/json'}