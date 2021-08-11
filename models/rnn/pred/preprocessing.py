class PreprocessingRNN:
  def __init__(self):
    self.txt = "./utils/vocab/word2index.txt"
    self.word2index = {}
    self.load_vocab()
    self.max_length = 128

  def load_vocab(self):
    import pickle 
    with open(self.txt, "rb") as f:
      word2index = pickle.load(f)
    self.word2index = word2index[0]
    f.close()

  def transform(self, text: str):
    '''予測時の前処理'''
    import torch 
    text = text.strip().split()
    text_ = []
    for r in text:
      r = r.replace("\n", " ")
      r = r.replace("\r", "")
      r = r.replace(".", " . ")
      r = r.replace(",", " , ")
      text_.append(r)

    inputs = []
    for r in text_:
      if r in self.word2index:
        idx = self.word2index[r]
      else:
        idx = self.word2index["<unk>"]
      inputs.append(idx)
    inputs.insert(0, self.word2index["<cls>"])

    if self.max_length > len(inputs):
      for _ in range(self.max_length-len(inputs)):
        inputs.append(self.word2index["<pad>"])
    else:
      inputs = inputs[:self.max_length]

    inputs = torch.tensor(inputs, dtype=torch.long)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = inputs.unsqueeze(0).to(device)
    return inputs 

prep = PreprocessingRNN()

