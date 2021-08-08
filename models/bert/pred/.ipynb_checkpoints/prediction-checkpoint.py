from transformers import BertTokenizer
from pred.load_vocab_file import vocab_file

class PredictionBertModel:
  def __init__(self):
    self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

  def transform(self, input_ids: list) -> list:
    text_list = self.tokenizer.convert_ids_to_tokens(input_ids)
    new_text = []
    for t in text_list:
      t = t.replace("##", "")
      t = t.replace("[PAD]", "")
      new_text.append(t)
    return new_text

pred = PredictionBertModel()