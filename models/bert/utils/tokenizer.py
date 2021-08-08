from transformers import BertTokenizer
from typing import Union 
import torch 
from utils.load_vocab_file import ModelNameByBertTokenizer

class BasicToken:

  def __call__(self, text: Union[str, list],
               labels: Union[int, list],
               padding: str="",
               max_length: int=216,
               truncation: bool=False,
               return_tensor: bool=True,
               cuda: bool=False,
               ):
    # return input_ids , attention_mask and token_type_ids 
    param = {
        "text": text,
    }

    if padding != "":
      param["padding"] = padding # ["max_length", "longest"]
    if max_length != 0:
      param["max_length"] = max_length
    if truncation:
      param["truncation"] = truncation

    tokenized = self.tokenizer(**param)

    if labels != None:
      tokenized["labels"] = labels
    if cuda:
      tokenized = {k: v.cuda() for k, v in tokenized.items()}
    if return_tensor:
      tokenized = {k: torch.tensor(v, dtype=torch.long) for k, v in tokenized.items()}

    return tokenized

class BertEnTokenizer(BasicToken):
  def __init__(self):
    self.model_name = ModelNameByBertTokenizer().en_model_name 
    self.tokenizer = BertTokenizer(vocab_file=self.model_name, do_lower_case=True)
    self.vocab_size = 0 
    self._get_vocab()

  def _get_vocab(self):
    with open(self.model_name, "r") as f:
      data = f.read().strip().split("\n")
    self.vocab_size = len(data)
    
    