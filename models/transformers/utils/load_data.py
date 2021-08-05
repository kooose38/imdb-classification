import random 
import pickle 

class LoadTrainDataset:
  def __init__(self):
    self.data = None 
    self.load_data()

  def load_data(self):
    f = open("./utils/data/train.txt","rb")
    data = pickle.load(f)
    self.data = random.sample(data, len(data))

loaded_data = LoadTrainDataset().data 

