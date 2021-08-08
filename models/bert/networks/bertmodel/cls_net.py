import torch 
import torch.nn as nn 

class BertClassification(nn.Module):
  '''<cls>の要素ベクトルを取り出す分類の最終層'''
  def __init__(self, config):
    super(BertClassification, self).__init__()
    
    self.fc = nn.Sequential(
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.Tanh()
    )

  def forward(self, x):
    '''
    x: BertEncoderの12回の最後の出力層
    '''
    # (batch, token, hidden) -> (batch, hidden)
    xx = x[:, 0, :] # 各バッチから1トークン分のベクトルを取り出す
    xx = xx.view(x.size()[0], x.size()[2])
    xx = self.fc(xx)
    return xx