# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from time import time
from PIL import Image
import torch
import torch.nn as nn
from IPython.display import clear_output
import matplotlib.pyplot as plt
from backbones import get_model
from privacy_pipeline import Pipeline
import torch.nn.functional as F
# %%
DEVICE = torch.device("cuda")

# 获取 & 载入模型
net = get_model('r100',fp16=False).eval().to(DEVICE)
net.load_state_dict(torch.load('/private/data/backbone.pth'))

# #这里的batch_size是干啥的?
batch_size = 80
image_dim = 112

def tensorToImage(im):return (im+1)/2

from dataset import MXFaceDataset,DataLoaderX
# 载入数据集
train_set = MXFaceDataset('/private/data/ms1m-retinaface-t1',local_rank=0)
a = DataLoaderX(local_rank=0,dataset=train_set,batch_size=1)

# %%
with torch.no_grad():
    for i in a:
        l = i[0]
        break
    emb_target = net(l)
    # print(emb_target)
# %%

plt.imshow(tensorToImage(l[0]).permute(1, 2, 0).cpu().detach().numpy())

# %%
pipeline = Pipeline(emb_target,
                    net,
                    DEVICE,
                    dim=image_dim,
                    batch_size=batch_size,
                    sym_part=1,
                    multistart=True,
                    gauss_amplitude=0.02
                    )

# %%
(emb_target**2).sum()

# %%
cosines_target = []
facenet_sims = []

# %%
with torch.no_grad():
    for i in range(15000):
        start = time()
        
        recovered_face, cos_target = pipeline()
        
        cosines_target.append(cos_target)
        
        
        if i % 10 == 0:
            clear_output(wait=True)
            recovered_face = np.transpose(recovered_face.cpu().detach().numpy(),(1,2,0))
            recovered_face = recovered_face - np.min(recovered_face)
            recovered_face = recovered_face / np.max(recovered_face)
            
            plt.figure(dpi=130)
            plt.imshow(recovered_face)
            plt.title(f"iterations {i*pipeline.batch_size}   cos_target={round(cos_target,3)}   norm={round(pipeline.norm,3)}")
            plt.show()
            
            plt.plot(cosines_target)
            plt.grid()
            plt.title("cos with a target embedding vs iters")
            plt.show()


# %%






























