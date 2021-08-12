import torch 
import torch.nn as nn 

class RNNClassification(nn.Module):
  def __init__(self, embedding_dim, tag_size):
    super(RNNClassification, self).__init__()
    self.embedding_dim = embedding_dim 

    self.fc = nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim),
        nn.Dropout(.1),
        nn.LayerNorm(embedding_dim),
        nn.GELU(),
        nn.Linear(embedding_dim, tag_size),
        nn.Softmax()
    )

  def forward(self, x, embeddings):
    '''
    x: (batch, hidden) LayerRNNからの出力
    embeddings: (batch, token. hidden) 埋め込み層の出力

    output: (batch, tag_size) バッチ単位でクラス分類
    '''
    embeddings = embeddings[:, -1, :]
    embeddings = embeddings.view(embeddings.size()[0], self.embedding_dim)
    output = x + embeddings 
    output = self.fc(output)
    return output 
