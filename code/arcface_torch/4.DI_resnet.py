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
import random
import random
import torch
import torch.nn as nn 
import torch.backends.cudnn as cudnn 
import torch.optim as optim
from torchvision.utils import make_grid
from backbones import get_model
from privacy_pipeline import Pipeline
import torch.nn.functional as F
import math
from IPython.display import clear_output
from IPython import display
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from IPython.display import Image as showIm
from IPython.display import display
import numpy as np
import torchvision.models as models
net = models.resnet50(pretrained=True).eval().cuda()


class DeepInversionFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature
    def close(self):
        self.hook.remove()


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


image_resolution = 224
do_flip = True
# net = get_model('r100',fp16=False).eval().cuda()
# net.load_state_dict(torch.load('/private/data/backbone.pth'));

r_feature_layers = [] 
for module in net.modules():
    if isinstance(module,nn.BatchNorm2d):
        r_feature_layers.append(DeepInversionFeatureHook(module))


# +
def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


# -

bs = 16
# emb_target = torch.load('save/emb_average.pt').cuda()
# emb_target = emb_target.expand(bs,-1).cuda()
# cos_loss = torch.cosine_similarity(emb_target,emb_out)
# norm_loss = ((torch.norm(emb_target,dim=1)-torch.norm(emb_out,dim=1))**2)
best_loss = 1e4
emb_target = torch.LongTensor([random.randint(0, 999) for _ in range(bs)]).to('cuda')


# +
stat = [[],[],[],[],[]]
# emb_target = torch.load(emb_file).cuda()
# emb_target = emb_target.expand(bs,-1).cuda()
#####################
lr = 0.25
jitter = 15
epochs = 2000
c_cos,c_norm,c_bn,c_l2var = 1,0.000,0.01,0.0001
flip_prob = 0.5
bn_first_scale = 10
bn_latter_scale = 0 ### not implemented
clamp = True
symmetry = False
bs = 16
start_noise = True
emb_file = 'save/emb_1.pt' # 'save/emb_average.pt'
#####################
if not start_noise:inpu = torch.zeros((bs,3,image_resolution,image_resolution),requires_grad=True,device='cuda',dtype=torch.float)
else:inpu = torch.randn((bs,3,image_resolution,image_resolution),requires_grad=True,device='cuda',dtype=torch.float)
optimizer = optim.Adam([inpu],lr=lr,betas=[0.5,0.9],eps=1e-8)
# optimizer = optim.Adam([inputs], lr=lr, betas=(0.9,0.999), eps = 1e-8)
lr_scheduler = lr_cosine_policy(lr,100,epochs)
for epoch in range(epochs):
    lr_scheduler(optimizer, epoch,epoch)
    off1 = random.randint(-jitter,jitter)
    off2 = random.randint(-jitter,jitter)
    inputs = inpu
    inputs = torch.roll(inputs, shifts=(off1,off2),dims=(2,3))

    flip = random.random() > flip_prob
    if flip and do_flip:
        inputs[:] = torch.flip(inputs, dims=(3,))
    optimizer.zero_grad()
    net.zero_grad()
    emb_out = net(inputs)
    cos_loss = nn.CrossEntropyLoss()(emb_out,emb_target)
    norm_loss = torch.ones(10).cuda() # ((torch.norm(emb_target,dim=1)-torch.norm(emb_out,dim=1))**2)
    var_l1_loss, var_l2_loss = get_image_prior_losses(inputs)
    rescale = [bn_first_scale] + [1. for _ in range(len(r_feature_layers) - 1)]
    r_feature_loss = sum([mod.r_feature * rescale[idx] for (idx,mod) in enumerate(r_feature_layers)])
    
#     l2_loss = torch.norm(inputs.view(bs,-1),dim=1).mean()
    
    loss =  cos_loss * c_cos + \
                norm_loss * c_norm + \
                r_feature_loss * c_bn + \
            var_l2_loss * c_l2var
    
    
    loss.sum().backward()
    optimizer.step()
    with torch.no_grad():
        stat[0].append(cos_loss.sum().item())
        stat[1].append(norm_loss.sum().item())
        stat[2].append(var_l2_loss.sum().item())
        stat[3].append(r_feature_loss.sum().item())
        stat[4].append(loss.sum().item())
        if symmetry : inputs[:] = (inputs + torch.flip(inputs,[3]))/2
        if clamp: inputs.clamp(-1,1)
    if epoch %25 ==24:
#         with torch.no_grad():
#         inputs = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))
        clear_output(wait=True)
        #display.set_matplotlib_formats('svg')
        plt.figure(dpi=70)
        plt.subplot(221),plt.plot(stat[0]),plt.subplot(222),plt.plot(stat[1]),plt.subplot(223),plt.plot(stat[2]),plt.subplot(224),plt.plot(stat[3]),plt.show()
        plt.subplot(221), plt.plot(stat[4]), plt.show()
        save_image(inputs.data.clone(),f'img/{epoch:04d}.png',normalize=True,nrow=8)
        display(showIm(filename=f'img/{epoch:04d}.png'))
# -

f'{epoch:04d}.png'

startNoise = False

a = torch.arange(25).view(5,5)

emb_target,emb_out.shape

a[:,1]=(a[:,1]+a[:,2]+a[:,3])/3

cos_loss,r_feature_loss


