# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch

import torchvision
# +
# # !unzip -q food-11.zip


# +
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
train_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])
# -

from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
batch_size = 128
train_set = DatasetFolder('food-11/training/labeled',
                         loader=lambda x:Image.open(x),extensions='jpg',
                         transform = train_tfm)
valid_set = DatasetFolder('food-11/validation',
                         loader=lambda x:Image.open(x),extensions='jpg',
                         transform = test_tfm)
test_set = DatasetFolder('food-11/testing',
                        loader= lambda x:Image.open(x),extensions='jpg',
                        transform = test_tfm)
train_loader = DataLoader(train_set,batch_size=batch_size,
                          shuffle=True,num_workers=8)
valid_loader = DataLoader(valid_set,batch_size=batch_size,
                          shuffle=True,num_workers=8)
test_loader = DataLoader(test_set,batch_size=batch_size,
                         shuffle=True,num_workers=8)

import torch.nn as nn
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.cnn_layers= nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),
            
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),
            
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4,4,0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,11)
        )
    def forward(self,x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x


import zz

# zz.summary(Classifier(),(3,128,128))
import zz
zz.darkplt()
import d2l.torch as d2l

# +
import matplotlib.pyplot as plt
device = 'cuda'
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.0003,weight_decay=1e-5)
n_epochs = 80
do_semi = False
a_loss = d2l.Animator(xlabel = 'epoch',xlim=[0,n_epochs],ylim=[0,3],
                 legend=['train loss','test loss'],first=True)
a_accs = d2l.Animator(xlabel = 'epoch',xlim=[0,n_epochs],ylim=[0,1],
                 legend=['train acc','test acc'])


for epoch in range(n_epochs):
    if do_semi:
        pass
    model.train()
    train_loss = []
    train_accs = []
    for imgs,labels in train_loader:
#         print(imgs,labels)
        imgs,labels = imgs.cuda(),labels.cuda()
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(),max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss = sum(train_loss)/len(train_loss)
    train_acc = sum(train_accs)/len(train_accs)
    model.eval()
    valid_loss = []
    valid_accs = []
    for imgs,labels in valid_loader:
        imgs,labels = imgs.cuda(),labels.cuda()
        with torch.no_grad():
            logits = model(imgs)
        loss = criterion(logits,labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss)/len(valid_loss)
    valid_acc = sum(valid_accs)/len(valid_accs)
    a_loss.add(epoch,[train_loss,valid_loss])
    a_accs.add(epoch,[train_acc,valid_acc])

# -

model.eval()
predictions = []
for imgs,labels in tqdm(test_loader):
    imgs,labels = imgs.cuda(),labels.cuda()
    with torch.no_grad():
        logits = model(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

with open('predict.csv','w') as f:
    f.write("Id,Category\n")
    for i,pred in enumerate(predictions):
        f.write(f'{i},{pred}\n')



!
