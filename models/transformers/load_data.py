from utils.load_data import loaded_data 
from utils.tokenize import EnWordToTensor

class CreateDataLoader:
  def __init__(self):
    self.data = loaded_data 
    self.vocab_size = 0

  def make_loader(self, batch_size=32):
    tokenizer = EnWordToTensor()
    train, val, test = tokenizer.transform(self.data, loader=True, batch_size=batch_size, limit_length=256)
    self.vocab_size = tokenizer.vocab_size

    return train, val, test