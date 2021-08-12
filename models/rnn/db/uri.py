import os 

PROJECT_NAME = os.getenv("PROJECT_NAME", "imdb-classification")

class MODELDB_URI:
   uri = os.getenv("MODELDB_IP_ADDRESS", "http://<IP_ADDRESS>:8000/v0.1/api/")
   headers = {
      "Content-Type": "application-json"
   }