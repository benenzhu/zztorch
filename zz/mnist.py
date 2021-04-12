import zz
from d2l.torch import predict_ch3
from IPython import display
import math
import numpy as np
import torch
from torch import randperm
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import os
device = 'cuda'
d = 'cuda'


def get_mnist(batch_size, shuffle=False,fast=True):
    if not fast:
        trans = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
        trainLoader = DataLoader(
            MNIST('~/data', train=True, download=True, transform=trans),
            batch_size=batch_size,
            shuffle=True,
            # pin_memory=True,
            num_workers=4
        )
        testLoader = DataLoader(
            MNIST('~/data', train=False, download=True, transform=trans),
            batch_size=batch_size,
            shuffle=True,
            # pin_memory=True,
            num_workers=4
        )
        return trainLoader, testLoader
    else:
        return trainLoader, testLoader


def get_cifar10(batch_size=64, shuffle=False, fast=True, label=False):
    if not fast:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        num_workers = min(batch_size, 4)
        trainset = CIFAR10(root='~/data', train=True,
                           download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers, prefetch_factor=5)
        testset = CIFAR10(root='~/data', train=False,
                          download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, prefetch_factor=5)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
        if label:
            return trainloader, testloader, classes
        else:
            return trainloader, testloader
    else:
        return trainloader, testloader, classes


def get_cifar10_mlleaks_fast(batch_size=128):
    Processed = 'CIFAR10_pre.pt'
    ss = 10520
    if not os.path.exists(Processed):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4948052, 0.48568845, 0.44682974],
                                 [0.24580306, 0.24236229, 0.2603115])
        ])
        trainset = CIFAR10(root='~/data', train=True,
                           download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=1000000)
        testset = CIFAR10(root='~/data', train=False,
                          download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=1000000)
        saves = []
        X, y = zz.test_iter(trainloader)
        saves.append(X)
        saves.append(y)
        X, y = zz.test_iter(testloader)
        saves.append(X)
        saves.append(y)
        torch.save(saves, Processed)
    a, b, c, d = torch.load(Processed)
    a, b, c, d = g(a,b,c,d)
    trainLoader = zziter(a[0:ss], b[0:ss], batch_size)
    testLoader = zziter(a[ss:2 * ss], b[ss: 2*ss], batch_size)
    shadowTrain = zziter(a[2 * ss:3 * ss], b[2 * ss: 3 * ss], batch_size)
    shadowTest = zziter(a[3 * ss:4 * ss], b[3 * ss: 4 * ss], batch_size)

    return trainLoader, testLoader, shadowTrain, shadowTest


def get_cifar10_mlleaks(batch_size=100, shuffle=False):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974],
                             [0.24580306, 0.24236229, 0.2603115])
    ])
    num_workers = min(batch_size, 1)
    idx = torch.arange(60000)
    # print(idx)
    ss = 10520
    trainset = CIFAR10(root='~/data', train=True,
                       download=True, transform=transform)
    # trainloader = DataLoader( trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=5)
    # testset = CIFAR10( root='~/data', train=False, download=True, transform=transform)
    # testloader = DataLoader( testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=5)
    train, test, strain, stest = Subset(trainset, idx[0:ss]), Subset(trainset, idx[ss: ss*2]), \
        Subset(trainset, idx[ss*2:ss*3]), Subset(trainset, idx[ss*3: ss*4])
    def getL(dset): return DataLoader(dset, batch_size=batch_size,
                                      num_workers=num_workers)  # , prefetch_factor=5)
    train, test, strain, stest = getL(train), getL(
        test), getL(strain), getL(stest)
    return train, test, strain, stest


def train(net, trainiter, epochs=50, learning_rate=0.01):
    from tqdm.auto import tqdm
    net = g(net)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    temp_loss = 0.0
    for _ in tqdm(range(epochs)):
        temp_loss = 0.0
        for X, y in trainIter:
            X ,y = g(X, y)
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            temp_loss += loss.item()
        print(f"loss:{temp_loss:.8f}")


def getFashionMNIST(batch_size, resize=None):
    trans = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
         ]
    )
    if resize:
        trans.insert(0, transforms.Resize(resize))
    train = DataLoader(
        FashionMNIST('~/data', train=True, download=True, transform=trans),
        batch_size=batch_size,
        shuffle=True,
        # pin_memory = True,
        num_workers=8
    )
    test = DataLoader(
        FashionMNIST('~/data', train=False, download=True, transform=trans),
        batch_size=batch_size,
        shuffle=True,
        # pin_memory = True,
        num_workers=8
    )

    return train, test


def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4


def testReadTime(train_iter, use_gpu=False):
    from tqdm.auto import tqdm
    timer = zz.Timer()
    for X, y in tqdm(train_iter):
        if use_gpu:
            X ,y = g(X, y)
    timer.print()


def get_fast_CIFAR10(batch_size=64, shuffle=False):
    # Processed  = '~/data/CIFAR10/Target_trans.pt'
    Processed = 'CIFAR10_pre.pt'

    if not os.path.exists(Processed):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
        ])
        trainset = CIFAR10(root='~/data', train=True,
                           download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=1000000)
        testset = CIFAR10(root='~/data', train=False,
                          download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=1000000)
        X, y = zz.test_iter(trainloader)
        tX, ty = zz.test_iter(testloader)
        torch.save([X,y,tX,ty], Processed)
    a, b, c, d = torch.load(Processed)
    a,b,c,d = g(a,b,c,d)
    # print(a[1])
    trainLoader, testLoder = zziter(a, b, batch_size), zziter(c, d, batch_size)
    return trainLoader, testLoder


class zziter:
    import numpy as np

    def __init__(self, X, y, batch_size, shuffle=False):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            y = torch.from_numpy(y).long()
        assert X.shape[0] == y.shape[0]
        self.X, self.y = g(X, y)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        index = 0
        if self.shuffle:
            indices = np.arange(len(self.X))
            np.random.shuffle(indices)
        start_idx = None
        for start_idx in range(0, len(self.X) - self.batch_size + 1, self.batch_size):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield self.X[excerpt], self.y[excerpt]

        if start_idx is not None and start_idx + self.batch_size < len(self.X):
            excerpt = indices[start_idx + self.batch_size:] if self.shuffle else slice(
                start_idx + self.batch_size, len(self.X))
            yield self.X[excerpt], self.y[excerpt]

    def __len__(self):
        return int(self.X.shape[0]/self.batch_size)


def iterate_minibatches(inputs, targets, batch_size=100, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(
            start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]
