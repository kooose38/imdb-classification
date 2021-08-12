import torch 
import torch.nn as nn 

class Embedder(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super(Embedder, self).__init__()
    self.embeddings = nn.Embedding(vocab_size, 
                                   embedding_dim,
                                   padding_idx=0)

  def forward(self, x):
    x = self.embeddings(x)
    return x