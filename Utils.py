
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



class CNNClassifier(nn.Module):

    def __init__(self, channels=3 , num_classes=2):
        super(CNNClassifier, self).__init__()

        self.conv1 = torch.nn.Conv2d(channels, 32, kernel_size=7, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.drop_layer_conv = nn.Dropout(p=0.25)
        self.drop_layer_dense = nn.Dropout(p=0.5)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 9 * 9, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop_layer_conv(x)

        # Block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.drop_layer_conv(x)
        x = x.view(-1, 64 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.drop_layer_dense(x)
        x = self.fc2(x)
        return x

