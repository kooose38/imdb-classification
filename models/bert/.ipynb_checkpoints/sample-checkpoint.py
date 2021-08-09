import json 
from pred.make_pkl import load_dump_prep


def _highlight(word: str, attn: float) -> str:
    "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)

def load_json():
    with open("./utils/data/labels.json", "r") as f:
      labels = json.load(f)
    return labels

def mk_html(index, sample, preds, attn_weight1):

  sentence = sample["input_ids"][index]
  labels = sample["labels"][index].item()
  pred = preds[index].item()
  pred_f = load_dump_prep("./pred/prediction/prediction_bert.pkl")
  sentence_text = pred_f.transform(sentence)

  assert len(setntence_text) == attn_weight1.size()[-1]

  corr = load_json()
  label_str = corr[str(labels)]
  pred_str = corr[str(pred)]

  html = f"正解: {label_str}<br>推論: {pred_str}<br>"

  for i in range(attn_weight1.size()[1]):
    # 12のアテンションを絞る
    # 分類器で先頭からLossを算出しているのでアテンションもそれに合わせる
    attns = attn_weight1[index, i, 0, :]
    attns /= attns.max()
    

    html += "BertのAttentionを可視化: "+str(i)+"<br>"
    for word, attn in zip(sentence_text, attns):
      if word == "[SEP]":
        break 
      html += _highlight(word, attn)
    html += "<br>"
    
  all_attens = attns*0
  for i in range(12):
    all_attens += attn_weight1[index, i, 0, :]
  all_attens /= all_attens
  
  html += "BertのAttentionを可視化_ALL <br>"
  for word, attn in zip(sentence_text, all_attens):
    if word == "[SEP]":
        break 
    html += _highlight(word, attn)
  html += "<br>"

  return html 