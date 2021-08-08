import json 
from attrdict import AttrDict

class LoadConfig:
  with open("./networks/config/bert_config.json", "r") as f:
    _data = json.load(f)
  config = AttrDict(_data)

config = LoadConfig().config