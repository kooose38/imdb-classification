def _highlight(word: str, attn: float) -> str:
    "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)

def mk_html(index, sample, preds, attn_weight1, word2index):
  index2word = {}
  for w, i in word2index.items():
    index2word[i] = w 

  sentence = sample["input_ids"][index]
  labels = sample["labels"][index]
  pred = preds[index]
  # batchを指定して<cls>と各トークンの重みを取り出す
  attns1 = attn_weight1[index, 0, :]

  if labels == 0:
    label_str = "Negative"
  else:
    label_str = "Positive"
  if pred == 0:
    pred_str = "Negative"
  else:
    pred_str = "Positive"

  assert sentence.size()[0] == attns1.size()[0]

  html = f"正解: {label_str}<br>推論: {pred_str}<br>"

  html += "Attention-layers<br>"
  for word, attn in zip(sentence, attns1):
    text = index2word[word.item()]
    html += _highlight(text, attn)
  html += "<br>"

  return html 