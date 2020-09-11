
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
#from collections import OrderedDict
import json
import PIL
from PIL import Image

from torch.utils.data import Dataset
import os
import natsort


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def Transfer_Res():
    
    model = models.resnet18(pretrained=True)

    model.conv1 = nn.Conv2d(3,64, kernel_size=(7,7), stride=(2, 2), padding=(3, 3), bias=False)

    model.fc = nn.Sequential(nn.Dropout(p=0.5),
                            nn.Linear(512,1000),
                            nn.BatchNorm1d(1000),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(1000, 2)
                            )


    
    return model



def transforms_():
    
    data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
    return data_transforms

