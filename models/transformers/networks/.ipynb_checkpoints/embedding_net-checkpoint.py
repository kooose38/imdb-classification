import torch 
import math
import torch.nn as nn 

class EmbedderAndPositionEncoder(nn.Module):
  '''文章トークンの単語埋め込み化'''
  def __init__(self, vocab_size, n_tokens, embedding_dim):
    super(EmbedderAndPositionEncoder, self).__init__()
    self.embedding_dim = embedding_dim
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    pe = torch.zeros(n_tokens, embedding_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pe = pe.to(device)
    # 一意になる位置ベクトル
    for pos in range(n_tokens):
      for i in range(0, embedding_dim, 2):
        pe[pos, i] = math.sin(pos/ (10000** ((2*i)/embedding_dim)))
        pe[pos, i+1] = math.cos(pos/ (10000**((2*i)/embedding_dim)))
    # 次元の追加と勾配の計算をしない
    self.pe = pe.unsqueeze(0) # (1, token, hidden)
    self.pe.requiers_grad = False 

  def forward(self, x):
    # (batch, token) -> (batch, token, hidden)
    embeded = self.embeddings(x)
    posed = math.sqrt(self.embedding_dim)*embeded+self.pe
    return posed