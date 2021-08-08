import pickle 
import random 
from utils.create_loader import loader 

class LoadIMDBdata:
  def __init__(self, filename="./utils/data/train.txt"):
    self.filename = filename 
    self.dataset = []
    self.train = []
    self.val = []
    self.test = []
    self.make_loader = loader 

    self._load_data() 

  def _load_data(self):
    with open(self.filename, "rb") as f:
      data = pickle.load(f)
    
    self.dataset = random.sample(data, len(data))
    n_ = len(self.dataset)
    n_train = int(n_*.6)
    n_val = int(n_*.2)

    self.train = self.dataset[:n_train]
    self.val = self.dataset[n_train:n_train+n_val]
    self.test = self.dataset[n_train+n_val:]

  def _create_dataloader(self, data: list, max_length: int):
    text_list, labels_list = [], []
    for d in data:
      for i, dd in enumerate(d.items()):
        if i == 0:
          text_list.append(dd[1])
        elif i == 1:
          labels_list.append(dd[1])
    data_loader = self.make_loader.bert_loader(text_list, labels_list, max_length=max_length)
    return data_loader 

  def transform(self, max_length: int):
    train_loader = self._create_dataloader(self.train, max_length)
    val_loader = self._create_dataloader(self.val, max_length)
    test_loader = self._create_dataloader(self.test, max_length)

    return train_loader, val_loader, test_loader 

dataset = LoadIMDBdata()