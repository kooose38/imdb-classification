import torch 
import torch.nn as nn 

class BertLayerNorm(nn.Module):
  def __init__(self, hidden_dim):
    '''BatchNormalizationを行う層'''
    super(BertLayerNorm, self).__init__()

    self.gamma = nn.Parameter(torch.ones(hidden_dim))
    self.beta = nn.Parameter(torch.zeros(hidden_dim))
    self.variance_eps = 1e-12

  def forward(self, x):
    # (batch, token, hidden) -> (batch, token, hidden)
    mu = x.mean(-1, keepdim=True)
    std = (x-mu).pow(2).mean(-1, keepdim=True)
    x = (x-mu)/torch.sqrt(std+self.variance_eps)
    return self.gamma*x+self.beta