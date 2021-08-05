import torch 

def evaluate(test, model, criterion):
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  model.to(device)

  accuracy = 0 
  test_loss = 0 
  n_test = 0 

  for data in test:
    inputs = data["input_ids"].to(device)
    labels = data["labels"].to(device)

    with torch.no_grad():
      output = model(inputs)
    loss = criterion(output, labels)
    test_loss += loss.item()
    y = output.argmax(-1)
    acc = torch.sum(y == labels).item()
    accuracy += acc 
    n_test += labels.size()[0]

  print(f"test Loss: {test_loss:4f} accuracy: {accuracy/n_test:4f}")