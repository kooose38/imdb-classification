import torch 
import torch.nn as nn 
from networks.bertmodel.embedding_net import EmbedderForBert 
from networks.bertmodel.encoder_net import BertEncoder 
from networks.bertmodel.cls_net import BertClassification 

class BertModel(nn.Module):
  def __init__(self, config):
    '''batch毎に768次元ベクトルで表現されるモデル'''
    super(BertModel, self).__init__()

    self.embedd = EmbedderForBert(config)
    self.encode = BertEncoder(config)
    self.decode = BertClassification(config)

  def _create_mask(self, input_ids):
    # <pad>を埋める
    mask = input_ids != 0 
    mask = mask.to(input_ids.device)
    return mask 

  def forward(self, input_ids, token_type_ids, mask=None, attention_flg=False):
    if mask is None:
      mask = self._create_mask(input_ids)
    # 1 layers 
    embeddings = self.embedd(input_ids, token_type_ids)
    # 2 layers 
    if attention_flg:
      output_attn, weights = self.encode(embeddings, mask, attention_flg) # *12count
    else:
      output_attn = self.encode(embeddings, mask)
    # 3 layers
    output = self.decode(output_attn[-1])
    if attention_flg:
      return output, weights 
    else:
      return output