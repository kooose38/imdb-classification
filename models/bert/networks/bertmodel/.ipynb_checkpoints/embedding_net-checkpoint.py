import torch 
import torch.nn as nn 
from networks.bertmodel.norm_net import BertLayerNorm

class EmbedderForBert(nn.Module):
  '''単語埋め込みと単語の位置を反映したベクトルの作成'''
  def __init__(self, config):
        
    super(EmbedderForBert, self).__init__()
    
    # 通常の単語埋め込み化
    self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
    # 位置ベクトル 512tokenからなる
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    # 文章が一文か二文で構成されているかを特長とする <cls>
    self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
    self.norm = BertLayerNorm(config.hidden_size)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
  
  def _create_pos(self, input_ids):
    # 位置関係によって一意の値をとる
    n_token = input_ids.size()[1]
    pos_ids = torch.arange(n_token, dtype=torch.long, device=input_ids.device)
    pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids) # (batch, token)
    return pos_ids

  def forward(self, input_ids, token_type_ids):
    '''
    input_ids: (batch, token) 入力トークン
    token_type_ids: (batch, token) 文章数をカウントしたもの（０・１）
    '''
    assert input_ids.size()[1] == token_type_ids.size()[1]
    # (batch, token) -> (batch, token, hidden)
    embed = self.embeddings(input_ids) # トークン
    token_type_embed = self.token_type_embeddings(token_type_ids) # 文章数
    pos_ids = self._create_pos(input_ids) 
    pos_ids_embed = self.position_embeddings(pos_ids) # トークンの位置
    # 全ての和をとる
    embeddings = embed+token_type_embed+pos_ids_embed
    embeddings = self.norm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings