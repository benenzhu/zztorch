import torch 
from torch import nn 
from torch import nn


class Generator(nn.Module):
    def __init__(self,z_dim =10 , im_dim = 784, hidden_dim = 128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.get
        )
    def get_generator_block(self, input_dim , output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim), 
            nn.BatchNorm1d(output_dim), 
            nn.ReLU(inplace=True)
        )
    def foward(self, noise):
        return self.gen(noise)

class LinearGAN():
    def __init__():
        gen = Generator
