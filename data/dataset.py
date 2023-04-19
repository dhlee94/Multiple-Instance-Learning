import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
import cv2

class ImageDataset(Dataset):
    def __init__(self, Data_path):
        self.Data_path = Data_path
        self.transform = transform

    def __len__(self):
        return len(self.Data_path)
    
    def __getitem__(self, idx):
        input = torch.from_numpy(np.load(self.Data_path[idx]).item()['image'])
        target = torch.from_numpy(np.load(self.Data_path[idx]).item()['label'])        
        
        return image, target