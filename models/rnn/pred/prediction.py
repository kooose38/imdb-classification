class PredictionRNN:
  def __init__(self):
    self.filename = "./utils/data/labels.json"
    self.labels = {}
    self.load_json()

  def load_json(self):
    import json 
    with open(self.filename, "r") as f:
      self.labels = json.load(f)
    f.close()

  def transform(self, output) -> list:
    '''rnnの出力からラベル名と確率を返す'''
    import torch 
    pred_idx = output.argmax(-1).item()
    pred_soft = torch.nn.functional.softmax(output).tolist()[0]
    pred_cate = self.labels[str(pred_idx)]

    return pred_cate, pred_soft 

pred = PredictionRNN()
   