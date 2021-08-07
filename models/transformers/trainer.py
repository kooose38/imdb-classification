import torch 
import torch.onnx as onnx 
import logging
from torch.utils.tensorboard import SummaryWriter
import time 
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__ )


def trainer(train, val, model, criterion, optimizer, num_epochs):
  writer = SummaryWriter(log_dir="tensorboard/")
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print(f"device: {device}")
  model.to(device)

  best_model = None 
  best_val_loss = 10000.0
    
  logger.info("start training ...")
  start = time.time()
    
  for epoch in range(num_epochs):
    train_loss = 0 
    model.train()
    for i, data in enumerate(train):
      inputs = data["input_ids"].to(device)
      labels = data["labels"].to(device)

      optimizer.zero_grad()
      output = model(inputs, attention_flg=False)
      loss = criterion(output, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
        
      writer.add_scaler("data/total_loss", float(loss.item()), (epoch+1)*i)
      writer.add_scaler("loss/train", float(loss.item()/(i+1)), (epoch+1)*i)
        
    print(f"{epoch+1}/{num_epochs} | train | Loss: {train_loss:4f}")
    now = time.time()
    logger.info(f"{epoch+1}/{num_epochs} duration in senconds: {now-start}")

    val_loss = 0 
    model.eval()
    for data in val:
      inputs = data["input_ids"].to(device)
      labels = data["labels"].to(device)

      with torch.no_grad():
        output = model(inputs, attention_flg=False)
      loss = critetion(output, labels)
      val_loss += loss.item()
      
      writer.add_scaler("data/total_loss_val", float(loss.item()), (epoch+1)*i)
      writer.add_scaler("loss/val", float(loss.item()/(i*1)), (epoch+1)*i)  
    
    print(f"{epoch+1}/{num_epochs} | val | Loss: {val_loss:4f}")
    now_val = time.time()
    logger.info(f"{epoch+1}/{num_epoch} duration in seconds: {now_val-start}")
    
    if best_val_loss > val_loss:
      logger.info("updata best model")
      best_model = model 
      best_val_loss = val_loss 

  print(f"best validation Loss: {best_val_loss:4f}")
  
  model_id = str(uuid.uuid4())[:6]
  filepath = f"./onnx/transformers_imdb_{model_id}.onnx"
  dummy = torch.rand(1, 256)
  onnx.export(best_model, 
              dummy,
              filepath,
             verbose=True,
             input_names=["input"],
             output_names=["output"])
  logger.info(f"saving best model file output .onnx:{filepath}")
  
  filepath = "./onnx/transformers_imdb.pth"
  touch.save(best_model.state_dict(), filepath)
  logger.info(f"saving best model weigths path: {filepath}")

  return best_model 
