# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# !pip install -e .

# %%
# from zztorch.gpumanageTorch import GPUManager

# %%
# zztorch.zz()
# print(dir(zztorch))

# %%
from zztorch import summary,d,draw
d

# %%
from zztorch import summary

# %%
from zztorch import summary
import torch
from torch import nn
### generator 在GAN中, 线性 + batchnorm, 最终的大小?
class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.get_generator_block(z_dim, hidden_dim),
            self.get_generator_block(hidden_dim, hidden_dim * 2),
            self.get_generator_block(hidden_dim * 2, hidden_dim * 4),
            self.get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )
    
    def get_generator_block(self,input_dim,output_dim):
        return nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )
    ### 进来的 noise 直接是 64维度的, 然后不断扩大到 im_dim = 784 维度了
    def forward(self, noise):  
        return self.gen(noise)
summary(Generator(), (10,))

# %%
from zztorch import getMNIST
trainL, testL = getMNIST(20)


# %%
trainL

# %%
testL

# %%
import torch 
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
import math
batch_size = 20
trans = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.5,),(0.5,))]
)
train = DataLoader(
    MNIST('./data',train=True, download=True, transform=trans), 
    batch_size=batch_size, 
    shuffle = True,
    pin_memory=True,
    num_workers=10
)
from tqdm.auto import tqdm
def load(train):
    for i in range(3):
        for i in tqdm(train):
            i[0][0] +=1
    #     print(i[0][0])
# load(train)


# %%
### 失败 ###
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
train = DataLoaderX(train)
# load(train)
# for i in train:
    
#     print('z',end=' ')

# %%

# %%

# %%
