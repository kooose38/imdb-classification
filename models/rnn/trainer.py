import time 
import os 
import boto3 
from torch.utils.tensorboard import SummaryWriter
import uuid 
import torch 
import torch.onnx as onnx 
import logging 
from db.crud import add_model, add_experiments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__ )

def trainer(train, val, net, criterion, optimizer, num_epochs, description=None):
  writer = SummaryWriter(log_dir="tensorboard/")
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  logger.info(f"device {device}")
  net.to(device)

  best_model = None 
  best_val_loss = 10000.0
  start = time.time()
  logger.info("start training ....")

  for e in range(num_epochs):
    net.train()
    train_loss = 0 
    n_train = 0 
    for i, data in enumerate(train):
      inputs = data["input_ids"].to(device)
      labels = data["labels"].to(device)

      optimizer.zero_grad()
      output = net(inputs)
      loss = criterion(output, labels)
      train_loss += loss.item()
      n_train += labels.size()[0]
      loss.backward()
      optimizer.step()

      writer.add_scalar("data/total_loss", float(loss.item()), (e+1)*i)
      writer.add_scalar("loss/train", float(loss.item()/(i+1)), (e+1)*i)

    print(f"{e+1}/{num_epochs}  | train | loss: {train_loss/n_train:4f}")
    now = time.time()
    logger.info(f"duration in seconds {now-start}")

    net.eval()
    val_loss = 0 
    n_val = 0
    acc = 0
    for data in val:
      inputs = data["input_ids"].to(device)
      labels = data["labels"].to(device)
      
      with torch.no_grad():
        output = net(inputs)
      loss = criterion(output, labels)
      val_loss += loss.item()
      n_val += labels.size()[0]
      pred = output.argmax(-1)
      acc += torch.sum(pred == labels).item()

      writer.add_scalar("data/total_val_loss", float(loss.item()), (e+1)*i)
      writer.add_scalar("loss/val", float(loss.item()/(i+1)), (e+1)*i)

    print(f"{e+1}/{num_epochs} | val | loss: {val_loss/n_val:4f}")
    print(f"{e+1}/{num_epochs} | val | accuracy: {acc/n_val:4f}")
    now = time.time()
    logger.info(f"duration in seconds {now-start}")

    if best_val_loss > val_loss:
      best_model = net 
      best_val_loss = val_loss 

  print(f"best validation loss: {best_val_loss:4f}")

  model_id = str(uuid.uuid4())[:6]
  model_name = "rnn"
  filename_onnx = f"./models/{model_name}_imdb_{model_id}.onnx"
  filename_pth = f"./models/{model_name}_imdb_{model_id}.pth"

  try:
    start_upload = time.time()

    saving_model_local(best_model,
                       filename_onnx,
                       filename_pth,
                       device)
    
    add_model_db(best_val_loss,
                 model_id,
                 model_name,
                 filename_onnx,
                 filename_pth,
                 optimizer,
                 description)
    
    upload_s3_model_file(filename_onnx,
                         filename_pth,
                         model_name)

    end_upload = time.time()
    logger.info(f"upload tasks duration in seconds {start_upload-end_upload}")

  finally:
    return best_model 


def saving_model_local(net, filename_onnx, filename_pth, device):
  os.makedirs("models", exist_ok=True)
  dummy = torch.rand(1, 128).long().to(device)
  onnx.export(net,
              dummy,
              filename_onnx,
              verbose=True,
              input_names=["input_ids"],
              output_names=["output"])
  logger.info(f"sucessfully saving model format onnxruntime >> filename is {filename_onnx}")

  torch.save(net.state_dict(), filename_pth)
  logger.info(f"successfully saving model weights format pth >> filename is {filename_pth} ")


def add_model_db(val_loss,
                 model_id,
                 model_name,
                 filename_onnx,
                 filename_pth,
                 optimizer,
                 description):
  model_version_id = "0.1"
  train_dataset = "./utils/data/train.txt"
  val_dataset = "./utils/data/train.txt"
  test_dataset = "./utils/data/train.txt"

  evaluations = {
      "validation_loss": val_loss 
  }

  parameters = {}
  for k, v in optimizer.param_groups[0]:
    if k not in ["params"]:
      parameters[k] = v 

  artifact_file_paths = {
      "local_path": {
          "onn_model": filename_onnx,
          "pth_model": filename_pth
      },
      "s3_path": {
          "onnx_model": f"{model_name}/model/{filename_onnx.split('/')[-1]}",
          "pth_model": f"{model_name}/model/{filename_pth.split('/')[-1]}"
      }
  }

  add_model(model_id, model_name, description=description)
  add_experiments(model_id,
                  model_version_id,
                  parameters,
                  train_dataset,
                  val_dataset,
                  test_dataset,
                  evaluations,
                  artifact_file_paths)


def upload_s3_model_file(filename_onnx, filename_pth, model_name):
  s3 = boto3.resource("s3")
  bucket = s3.Bucket("imdb-classificaton")
  logger.info("upload s3 ...")
  bucket.upload_file(filename_onnx, f"{model_name}/model/{filename_onnx.split('/')[-1]}")
  bucket.upload_file(filename_pth, f"{model_name}/model/{filename_pth.split('/')[-1]}")
  logger.info("finished upload s3 tasks ")
