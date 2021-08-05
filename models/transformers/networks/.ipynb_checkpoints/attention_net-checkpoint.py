import torch 
import torch.nn as nn 
import math

class Attention(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    '''Attentionによる重みづけを行う'''
    self.embedding_dim = embedding_dim
    self.query = nn.Linear(embedding_dim, embedding_dim)
    self.key = nn.Linear(embedding_dim, embedding_dim)
    self.value = nn.Linear(embedding_dim, embedding_dim)

    self.out = nn.Linear(embedding_dim, embedding_dim)

  def transform(self, key):
    return key.view(-1, self.embedding_dim, key.size()[1])

  def forward(self, x, mask):
    # Attention重みとしてのキー・クエリ・バリュー
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    
    # query+keyの関係性からtoken同士の関係を計算する
    k = self.transform(k)
    # weights.size == (batch, token, token)
    weights = torch.matmul(q, k)/math.sqrt(self.embedding_dim)
    # <pad>の部分は後のソフトマックス関数で影響値を0に近づける
    # mask.size == (batch, 1, token)
    mask = mask.unsqueeze(1)
    mask = mask.view(-1, 1, x.size()[1])
    weights = weights.masked_fill(mask == 0, -1e9)
    weights = torch.nn.functional.softmax(weights, dim=-1)
    # Attention*value (batch, token, hidden)
    output = torch.matmul(weights, v)
    output = self.out(output)
    return output, weights


class AttentionAfterLinear(nn.Module):
  def __init__(self, embedding_dim, d_ff=1024, dropout=.1):
    '''Attention outputを線形変換する'''
    super().__init__()

    self.fc = nn.Sequential(
        nn.Linear(embedding_dim, d_ff),
        nn.Dropout(dropout),
        nn.Linear(d_ff, embedding_dim)
    )

  def forward(self, x):
    # (batch, token, hidden) -> (batch, token, d_ff) -> (batch, token, hidden)
    output = self.fc(x)
    return output 


class TransformerBlock(nn.Module):
  def __init__(self, embedding_dim, dropout=.1):
    '''Attention+Linear'''
    super().__init__()

    self.norm_1 = nn.LayerNorm(embedding_dim)
    self.norm_2 = nn.LayerNorm(embedding_dim)

    self.attn = Attention(embedding_dim)
    self.fc = AttentionAfterLinear(embedding_dim)

    self.drop_1 = nn.Dropout(dropout)
    self.drop_2 = nn.Dropout(dropout)

  def forward(self, x, mask=None):
    '''
    x: Embedderの出力層
    mask: (batch, token)
    '''
    x1 = self.norm_1(x)
    output, weights = self.attn(x1, mask)
    x2 = x+self.drop_1(output)
    x3 = self.norm_2(x2)
    output = x2+self.drop_2(self.fc(x3))
    # output.size == (batch, token, hidden)
    # weights.size == (batch, token, token)
    return output, weights