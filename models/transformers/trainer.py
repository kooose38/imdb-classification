import torch 
import torch.onnx as onnx 

def trainer(train, val, model, criterion, optimizer, num_epochs):
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"device: {device}")
  model.to(device)

  best_model = None 
  best_val_loss = 10000.0

  for epoch in range(num_epochs):
    train_loss = 0 
    model.train()
    for data in train:
      inputs = data["input_ids"].to(device)
      labels = data["labels"].to(device)

      optimizer.zero_grad()
      output = model(inputs, attention_flg=False)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
    print(f"{epoch+1}/{num_epochs} | train | Loss: {train_loss:4f}")

    val_loss = 0 
    model.eval()
    for data in val:
      inputs = data["input_ids"].to(device)
      labels = data["labels"].to(device)

      with torch.no_grad():
        output = model(inputs, attention_flg=False)
      loss = critetion(output, labels)
      val_loss += loss.item()
    print(f"{epoch+1}/{num_epochs} | val | Loss: {val_loss:4f}")
    if best_val_loss > val_loss:
      best_model = model 
      best_val_loss = val_loss 

  print(f"best validation Loss: {best_val_loss:4f}")

  filepath = "./onnx/transformers_imdb.onnx"
  dummy = torch.rand(1, 256)
  onnx.export(best_model, dummy, filepath)
  print(f"saving file :{filepath}")

  return best_model 