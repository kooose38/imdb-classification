import joblib 

def make_dump_prep(model, filepath: str="./preprocessing/preprocessing_transformers.pkl"):
  joblib.dump(model, filepath)

def load_dump_prep(filepath: str="./preprocessing/preprocessing_transformers.pkl"):
  prep = joblib.load(filepath)
  return prep 