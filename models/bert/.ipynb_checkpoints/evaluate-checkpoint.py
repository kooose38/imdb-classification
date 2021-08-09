from tqdm import tqdm
import torch 

def evaluate(test, model, criterion):
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  model.to(device)

  test_loss = 0 
  acc = 0
  n_test = 0 
  model.eval()
  for data in tqdm(test):
    inputs = data["input_ids"].to(device)
    mask = data["attention_mask"].to(device)
    token_type_ids = data["token_type_ids"].to(device)
    labels = data["labels"].to(device)

    with torch.no_grad():
      output = model(inputs, token_type_ids, mask)

    loss = criterion(output, labels)
    pred = output.argmax(-1)
    acc += torch.sum(pred == labels).item()
    test_loss += loss.item()
    n_test += labels.size()[0]

  print(f"test loss: {test_loss/n_test:4f} accuracy: {acc/n_test:4f}")