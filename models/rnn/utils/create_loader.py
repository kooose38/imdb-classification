from torch.utils.data import DataLoader 
import torch 

class MakeDataLoader(DataLoader):
  def __init__(self, batch_size: int=32, shuffle: bool=True):
    self.batch_size = batch_size
    self.shuffle = shuffle 

  def my_word_loader(self, dataset):
    return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    