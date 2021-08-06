from networks.attention_net import TransformerBlock
from networks.classifier_net import Classification
from networks.embedding_net import EmbedderAndPositionEncoder
import torch 
import torch.nn as nn 

class TransformerClassification(nn.Module):
  def __init__(self, vocab_size, n_token, embedding_dim, tag_size):
    '''全てを総括したモデル層'''
    super().__init__()

    self.embedd = EmbedderAndPositionEncoder(vocab_size, n_token, embedding_dim)
    self.attnet = TransformerBlock(embedding_dim)
    self.classfier = Classification(embedding_dim, tag_size)

    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

  def create_mask(self, x):
    mask = x != 0
    mask.to(self.device)
    return mask 

  def forward(self, x, attention_flg=False):
    mask = self.create_mask(x)

    x1 = self.embedd(x)
    attnet_output = []
    for _ in range(2):
      x1, weights = self.attnet(x1, mask)
      attnet_output.append(x1)

    x3 = self.classfier(attnet_output[-1])
    
    if attention_flg:
      return x3, weights 
    else:
      return x3 