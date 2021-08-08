import torch 
import torch.nn as nn 
from networks.bertmodel.bertmodel import BertModel
from networks.weights.load_weights_from_s3 import load_weights_from_s3


class BertForIMDBClassification(nn.Module):
  def __init__(self, config, tag_size=2):
    '''BertModelをクラス分類の出力に改良したモデル '''
    super(BertForIMDBClassification, self).__init__()
    # BertModelの事前学習させたパラメータを使った学習
    # 初期時点で学習済パラメータの読み込みをする  
    bert = BertModel(config)
    bert.eval()
    param_name = []
    for name, param in bert.named_parameters():
      param_name.append(name)

    bert_state_dict = bert.state_dict()
    loaded_model_dict = load_weights_from_s3()
    for i, (k, v) in enumerate(loaded_model_dict.items()):
      name = param_name[i]
      bert_state_dict[name] = v 

      if i+1 >= len(param_name):
        break 
    bert.load_state_dict(bert_state_dict)
    # bert の重みを学習させない
    for param in bert.parameters():
      param.requires_grad = False 

    self.bert = bert 

    self.fc = nn.Linear(config.hidden_size, tag_size)
    # initlize last linear layer weights 
    nn.init.normal_(self.fc.weight, std=.02)
    nn.init.normal_(self.fc.bias, 0)

  def forward(self, input_ids, token_type_ids, mask=None, attention_flg=False):
    if attention_flg:
      output_bert, weights = self.bert(input_ids, token_type_ids, mask, attention_flg=True)
      # 4 layers 
      output = self.fc(output_bert)
      return output, weights 
    else:
      output_bert = self.bert(input_ids, token_type_ids, mask)
      output = self.fc(output_bert)
      return output