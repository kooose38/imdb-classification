import torch 
import torch.nn as nn 
from networks.bertmodel.norm_net import BertLayerNorm

class BertIntermediate(nn.Module):
  '''Attention層の出力を線形変換する'''
  def __init__(self, config):
        
    super(BertIntermediate, self).__init__()
    # (batch, token, hidden) -> (batch, token, intermediate_size)
    # 0付近が滑らかな活性化関数を使用する
    self.fc = nn.Sequential(
      nn.Linear(config.hidden_size, config.intermediate_size),
      nn.GELU()
    )

  def forward(self, x):
    output = self.fc(x)
    return output 

class BertOutput(nn.Module):
  def __init__(self, config):
    '''Attention層を線形変換後と前の状態の和'''
    super(BertOutput, self).__init__()

    # (batch, token, intermediat_size) -> (batch, token, hidden)
    self.fc = nn.Sequential(
        nn.Linear(config.intermediate_size, config.hidden_size),
        nn.Dropout(config.hidden_dropout_prob)
    )
    self.norm = BertLayerNorm(config.hidden_size)

  def forward(self, output_intermediate, output_attn):
    '''
    output_intermediate: BertIntermediateの出力 (batch, token, intermediate_size)
    output_attn: BertAttentionの出力 (batch, token, hidden)
    '''
    out = self.fc(output_intermediate)
    out = self.norm(out+output_attn)
    return out

##################################################################################
from networks.bertmodel.attention_net import BertAttention

class BertLayer(nn.Module):
  def __init__(self, config):
    '''Attention+Intermediate+Outputをまとめたもの'''
    super(BertLayer, self).__init__()

    self.attnet = BertAttention(config)
    self.intermadiate = BertIntermediate(config)
    self.out = BertOutput(config)

  def forward(self, embeddings, mask, attention_flg=False):
    '''
    embeddings: Embedderの出力 (batch, token, hidden)
    mask: attention_mask (batch, token)
    '''
    if attention_flg:
      output_attn, weights = self.attnet(embeddings, mask, attention_flg)
      output_inter = self.intermadiate(output_attn)
      output = self.out(output_inter, output_attn)
      return output, weights
    else:
      output_attn = self.attnet(embeddings, mask)
      output_inter = self.intermadiate(output_attn)
      output = self.out(output_inter, output_attn)
      return output
###################################################################################