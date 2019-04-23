from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
import os


data_path = os.getcwd()
data_set = MNIST(data_path,  train=True, transform=tfs.Compose([tfs.ToTensor()]), download=True)
target_dl = DataLoader(data_set, batch_size=64, shuffle=True)