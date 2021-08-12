import torch 
import torch.nn as nn 
from networks.rnn.classifier_net import RNNClassification
from networks.rnn.embed_net import Embedder 
from networks.rnn.layer_net import LayerRNN 

class RNN(nn.Module):
  def __init__(self, n_token=128, embedding_dim=256, hidden_dim=100, vocab_size=121855, tag_size=2):
    '''全ての層をまとめた層'''
    super(RNN, self).__init__()
    
    # 単語埋め込み層
    self.embed_net = Embedder(vocab_size, embedding_dim)
    # hidden層の反映（最帰型）
    self.hidden_net = LayerRNN(n_token, embedding_dim, hidden_dim)
    # クラス分類
    self.class_net = RNNClassification(embedding_dim, tag_size)

  def forward(self, x, hidden_flg=False):
    embedd = self.embed_net(x)
    if hidden_flg:
      output, hidden = self.hidden_net(embedd, hidden_flg=True)
    else:
      output = self.hidden_net(embedd)
    output = self.class_net(output, embedd)

    if hidden_flg:
      return output, hidden 
    else:
      return output