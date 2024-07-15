import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import csv
import numpy as np
import cv2

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class ObstacleDetection(nn.Module):
    def __init__(self, class_num=1):
        super(ObstacleDetection, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, class_num),
            nn.Sigmoid()
        )

        self.regresor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return self.classifier(x), self.regresor(x)

class ObstacleDataset(Dataset):
    def __init__(self, data_dir, path):
        self.data_dir = data_dir
        self.path = path
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(os.path.join(self.data_dir, self.path), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                #print(row)
                img = cv2.imread(os.path.join(self.data_dir, row['image_path']))
                img = cv2.resize(img, (224, 224))
                img = torch.from_numpy(img).float().cuda() / 255.
                img = img.permute(2, 0, 1)
                cls = torch.tensor([float(row['obstacle_notation'])], dtype=torch.float32)
                #print(type(row['obstacle_dist']), row['obstacle_dist'])
                dis = torch.tensor([float(row['obstacle_dist']) if row['obstacle_dist'] != 'nan' else 999], dtype=torch.float32)
                data.append((img, cls, dis))
            #print(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    # TODO: Violate the single responsibility principle
    def data_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)