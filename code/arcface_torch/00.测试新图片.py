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

import torchvision.datasets as dset
from torchvision import transforms as tsf
from torch.utils.data import DataLoader

data = DataLoader(dset.ImageFolder('image',transform =  tsf.Compose([tsf.ToTensor(),tsf.Resize([112,112])])))

from backbones import get_model
net = get_model('r100',fp16=False).eval().cuda()
net.load_state_dict(torch.load('/private/data/backbone.pth'))
for i in data:
    i = (i[0]-0.5)*2
    print(i.shape)
    torch.save(net(i.cuda()),'save/js.pt')

torch.Variable


