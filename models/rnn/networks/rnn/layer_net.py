import torch 
import torch.nn as nn 

class LayerRNN(nn.Module):
  def __init__(self, n_token, embedding_dim, hidden_dim):
    '''文章の語順に合わせて準伝播する再起型層'''
    super(LayerRNN, self).__init__()
    self.hidden_dim = hidden_dim 
    self.layers = nn.ModuleList([HiddenRNN(embedding_dim, hidden_dim)
                                for _ in range(n_token)])
    
  def _init_hidden(self, x):
    hidden = torch.zeros(x.size()[0], self.hidden_dim)
    hidden = hidden.to(x.device).to(torch.float32)
    hidden = hidden.long()
    return hidden 
    
  def forward(self, x, hidden_flg=False):
    '''
    x: (batch, token, hidden) Embedderからの出力

    output: (batch, hidden) 最後のtokenベクトル
    hidden: (batch, hidden_dim) 全てのtokenを反映した文脈ベクトル
    '''
    hidden = self._init_hidden(x)
    for i, layer in enumerate(self.layers):
      token = x[:, i, :]
      token = token.view(x.size()[0], x.size()[2])
      output, hidden = layer(token, hidden)

    if hidden_flg:
      return output, hidden
    else:
      return output 


class HiddenRNN(nn.Module):
  def __init__(self, embedding_dim, hidden_dim):
    super(HiddenRNN, self).__init__()
    self.hidden_dim = hidden_dim 

    self.i2h = nn.Linear(embedding_dim+hidden_dim, hidden_dim)
    self.i2o = nn.Linear(embedding_dim+hidden_dim, embedding_dim)

    self.dropout = nn.Dropout(.1)
    self.norm = nn.LayerNorm(embedding_dim)

  def forward(self, x, hidden):
    '''
    x: (batch, hidden) token1つ分
    hidden (batch, hidden_dim) token毎に更新されるhidden層
    '''
    combined = torch.cat((x, hidden), 1)
    hidden = self.i2h(combined)
    output = self.i2o(combined)
    output = self.dropout(output)
    output = self.norm(output)
    return output, hidden 
