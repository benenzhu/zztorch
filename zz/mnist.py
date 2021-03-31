import torch 
from torchvision.datasets import MNIST,CIFAR10,FashionMNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
import math
from IPython import display

import zz
from d2l.torch import predict_ch3

def getMNIST(batch_size):
    trans = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5,),(0.5,))]
    )
    train = DataLoader(
        MNIST('~/data',train=True, download=True, transform=trans), 
        batch_size=batch_size, 
        shuffle = True,
        # pin_memory=True,
        num_workers=4
    )
    test = DataLoader(
        MNIST('~/data',train=False, download=True, transform=trans), 
        batch_size=batch_size, 
        shuffle = True,
        # pin_memory=True,
        num_workers=4
    )
    
    return train,test
    
    
def getCIFAR10(batch_size=64,shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], 
                            [0.24580306, 0.24236229, 0.2603115])
    ])
    trainset = CIFAR10(
        root='~/data', 
        train=True, 
        download=True, 
        transform=transform
    )
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4,
        prefetch_factor=5
    )
    testset = CIFAR10(
        root='~/data', 
        train=False, 
        download=True, 
        transform=transform
    )
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4,
        prefetch_factor=5
    )
    classes = ('plane', 'car', 'bird', 'cat', 'deer', \
               'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,testloader,classes


def getFashionMNIST(batch_size, resize=None):
    trans = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5,),(0.5,))
        ]
    )
    if resize:
        trans.insert(0, transforms.Resize(resize))
    train = DataLoader(
        FashionMNIST('~/data',train=True, download=True, transform=trans), 
        batch_size = batch_size, 
        shuffle = True,
        # pin_memory = True,
        num_workers = 8
    )
    test = DataLoader(
        FashionMNIST('~/data',train=False, download=True, transform=trans), 
        batch_size = batch_size, 
        shuffle = True,
        # pin_memory = True,
        num_workers = 8
    )

    return train,test


def get_dataloader_workers():  
    """Use 4 processes to read the data."""
    return 4

def testReadTime(train_iter,use_gpu=False):
    from tqdm.auto import tqdm
    timer = zz.Timer()
    for X, y in tqdm(train_iter):
        if use_gpu: X.to('cuda'),y.to('cuda')
    timer.print()
    