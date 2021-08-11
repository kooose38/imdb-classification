import torch 
import logging 
from tqdm import tqdm 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__ )

def evaluate(test, net, criterion):
  n_test = 0 
  test_loss = 0 
  test_acc = 0 

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  logger.info(f"device: {device}")
  net.to(device)

  net.eval()
  for data in tqdm(test):
    inputs = data["input_ids"].to(device)
    labels = data["labels"].to(device)

    with torch.no_grad():
      output = net(inputs)

    loss = criterion(output, labels).item()
    test_loss += loss 
    n_test += labels.size()[0]
    pred = output.argmax(-1)
    test_acc += torch.sum(pred == labels).item()

  print(f"test loss: {test_loss/n_test:4f}")
  print(f"test accuracy: {test_acc/n_test:4f}")