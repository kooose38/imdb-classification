from utils.tokenizer import BertEnTokenizer
from torch.utils.data import DataLoader
from utils.preprocessing_text import PreProcessingTEXT

class MakeDataLoader(DataLoader):
  def __init__(self, batch_size: int=16, shuffle: bool=True, language="en"):
    self.batch_size = batch_size
    self.shuffle = shuffle 
    self.language = language
    self.tokenizer = BertEnTokenizer()

  def bert_loader(self,
                  text_list: list,
                  labels_list: list,
                  padding: str="max_length", 
                  max_length: int=216,
                  truncation: bool=True,
                  return_tensor: bool=True, 
                  cuda: bool=False
                  ):
    data = []
    for text, label in zip(text_list, labels_list):
      if self.language == "en":
        text = PreProcessingTEXT()._en_preprocessing(text)
      encoding = self.tokenizer(text,
                             padding=padding,
                             max_length=max_length,
                             truncation=truncation,
                             return_tensor=return_tensor,
                             cuda=cuda,
                             labels=label)
      data.append(encoding)

    return DataLoader(data, batch_size=self.batch_size, shuffle=self.shuffle)

loader = MakeDataLoader()
