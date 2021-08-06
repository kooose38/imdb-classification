class PreprocessingTransformers:

  def __init__(self, filename: str):
    self.word2index = {}
    self.filename = filename 
    self.vocab_size = 0

    self._load_file()

  def _load_file(self):
    import pickle 
    f = open(self.filename, "rb")
    data = pickle.load(f)
    for d in data:
      for w, i in d.items():
        if w not in self.word2index:
          self.word2index[w] = int(i)
    self.vocab_size = len(self.word2index)
    
    f.close()

  def fit(self):
    pass 
  
  def transform(self, text: str, max_length=256):
    import torch 
    import re
    text = re.sub("\n", " ", text)
    text = re.sub("\r", "", text)
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")

    text = text.strip().split()
    inputs = []
    for r in text:
      if r in self.word2index:
        idx = self.word2index[r]
      else:
        idx = self.word2index["<unk>"]
      inputs.append(idx)
    inputs.insert(0, self.word2index["<cls>"])

    if max_length > len(inputs):
      for _ in range(max_length-len(inputs)):
        inputs.append(self.word2index["<pad>"])
    else:
      inputs = inputs[:max_length]

    inputs = torch.tensor(inputs, dtype=torch.long)
    inputs = inputs.unsqueeze(0)

    return inputs 

# if __name__ = "__main__":
prep = PreprocessingTransformers("./preprocessing/word2index.txt")