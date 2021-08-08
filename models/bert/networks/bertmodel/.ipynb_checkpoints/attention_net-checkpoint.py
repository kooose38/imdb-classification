import torch 
import torch.nn as nn 
import math 
from networks.bertmodel.norm_net import BertLayerNorm

class BertSelfAttention(nn.Module):
  def __init__(self, config):
    '''key-queryからAttention weightsを求めてvalueとの和をとる'''
    super(BertSelfAttention, self).__init__()
    
    # attentionを分割するサイズ
    self.num_attention_heads = config.num_attention_heads # 12
    self.attention_head_size = int(config.hidden_size/config.num_attention_heads) # 768/12 = 64
    self.all_head_size = self.num_attention_heads*self.attention_head_size # 768

    self.query = nn.Linear(config.hidden_size, self.all_head_size) # 768->768
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def _transform(self, x):
    # (batch, token, hidden) -> (batch, 12, token, hidden/12)
    x = x.view(-1, self.num_attention_heads, x.size()[1], self.attention_head_size)
    return x 

  def _transform_mask(self, mask):
    # paddingが0の部分を(-inf)にすることで、
    # 後のソフトマックス関数により0に近づける目的
    # (batch, token) -> (batch, 1, 1, token)
    mask = mask.view(mask.size()[0], 1, 1, mask.size()[1])
    mask = mask.to(dtype=torch.float32)
    mask = (1.0-mask)*-10000.0
    return mask 

  def forward(self, x, mask, attention_flg=False):
    '''
    x: Embedderの出力層 (batch, token. hidden)
    mask: attention_mask (batch. token)
    '''
    # トークンごとの重み計算のためのキー・クエリ・バリュー
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    q_trans = self._transform(q)
    k_trans = self._transform(k)
    v_trans = self._transform(v)
    # Key*Query = Attention weights * 12 
    # (batch, 12, token, token)
    weights = torch.matmul(q_trans,
                           k_trans.view(-1, self.num_attention_heads, self.attention_head_size, x.size()[1]))
    weights = weights/math.sqrt(self.attention_head_size)
    # maskによる影響値を反映する
    mask = self._transform_mask(mask)
    weights = weights+mask 
    weigths = torch.nn.functional.softmax(weights) #ここで<pad>の部分はほぼ0になる
    weights = self.dropout(weights)
    # (batch, 12, token, hidden/12)
    output = torch.matmul(weights, v_trans)
    output = output.view(-1, x.size()[1], x.size()[2])

    if attention_flg:
      return output, weights 
    else:
      return output



class BertSelfOutput(nn.Module):
    '''BertSelfAttentionの出力(重みづけ）に元の単語埋め込みの和をとる'''
    # 重みづけベクトルのみでは単語を表現するには,
    # 小さい値なので元の埋め込みの和をとることで強く表現可能になる計算処理

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, output_attn, embeddings):
        '''
        output_attn：BertSelfAttentionの出力テンソル
        embeddings：Embeddingsモジュールもしくは前段のBertLayerからの出力
        '''
        output = self.dense(output_attn)
        output = self.dropout(output)
        output = self.LayerNorm(output+embeddings)
        return output
    
    
    
class BertAttention(nn.Module):
  def __init__(self, config):
    '''BertSelfAttention+BertSelfoutputをまとめたもの'''
    super(BertAttention, self).__init__()

    self.attnet = BertSelfAttention(config)
    self.attout = BertSelfOutput(config)

  def forward(self, embeddings, mask, attention_flg=False):
    if attention_flg:
      output_attn, weights = self.attnet(embeddings, mask, attention_flg=True)
      output = self.attout(output_attn, embeddings)
      return output, weights
    else:
      output_attn = self.attnet(embeddings, mask)
      output = self.attout(output_attn, embeddings)
      return output