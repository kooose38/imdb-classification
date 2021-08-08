import torch 
import torch.nn as nn 
from networks.bertmodel.intermediate_net import BertLayer

class BertEncoder(nn.Module):
  def __init__(self, config):
    '''BertLayersを12回繰り返す'''
    super(BertEncoder, self).__init__()
    self.layers = nn.ModuleList([BertLayer(config) 
                                  for _ in range(config.num_hidden_layers)
    ])

  def forward(self, embeddings, mask, attention_flg=False):
    all_encoder_output = []
    # 12 のアテンション重みと出力層
    for layer_module in self.layers:
      if attention_flg:
        embeddings, weights = layer_module(embeddings, mask, attention_flg)
      else:
        embeddings = layer_module(embeddings, mask)
      # (12, batch, token, hidden)
      all_encoder_output.append(embeddings) 
    if attention_flg:
      return all_encoder_output, weights 
    else:
      return all_encoder_output