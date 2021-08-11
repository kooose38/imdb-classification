from typing import Tuple 
from utils.load_data import loaded_data 
from utils.tokenizer import EnWordToTensor 

class CreateDataLoader:
   def __init__(self):
      self.data = loaded_data
      self.vocab_size = 0
      self.word2index = {}

   def make_loader(self, batch_size: int=32) -> Tuple:
      tokenizer = EnWordToTensor()
      train, val, test = tokenizer.transform(self.data, loader=True, batch_size=batch_size, limit_length=128)
      self.vocab_size = tokenizer.vocab_size 
      self.word2index = tokenizer.word2index 

      return train, val, test

data = CreateDataLoader()
