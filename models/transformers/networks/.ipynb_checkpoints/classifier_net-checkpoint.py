import torch 
import torch.nn as nn

class Classification(nn.Module):
  def __init__(self, embedding_dim, tag_size):
    '''Blockからの出力でクラス分類を行う'''
    super().__init__()
    self.embedding_dim = embedding_dim 
    self.fc = nn.Linear(embedding_dim, tag_size)

  def forward(self, x):
    # (batch, hidden)
    # 最初の文字である<CLS>の要素ベクトルをそれぞれ取り出す
    xx = x[:, 0, :]
    xx = xx.view(-1, self.embedding_dim)
    output = self.fc(xx)
    return output