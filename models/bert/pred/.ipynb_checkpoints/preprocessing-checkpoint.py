from transformers import BertTokenizer
from pred.load_vocab_file import vocab_file
import re

class PreprocessingBertModel:
  def __init__(self):
    self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

  def transform(self, text: str, max_length=256):
    text = re.sub("\n", " ", text)
    text = re.sub("\r", "", text)
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    
    import torch 
    encoding = self.tokenizer(text, max_length=max_length, padding="max_length", return_tensors="pt")
    return encoding

prep  = PreprocessingBertModel()