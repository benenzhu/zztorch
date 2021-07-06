# -*- coding: utf-8 -*-
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

# import cv2
import numpy as np
import torch
from backbones import get_model
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# img = np.random.randint(0,255,size=(112,112,3),dtype=np.uint8)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img = np.transpose(img,(2,0,1))

#导入网络到 net, 然后得到embedding
with torch.no_grad():
	net = get_model('r100',fp16=False).eval().cuda()
	net.load_state_dict(torch.load('/private/data/backbone.pth'))

from dataset import MXFaceDataset, DataLoaderX

train_set = MXFaceDataset('/private/data/ms1m-retinaface-t1',local_rank = 0)
a = DataLoaderX(local_rank=0,dataset= train_set, batch_size = 64,pin_memory=False)

with torch.no_grad():
    xl,yl = [],[]
    for i in tqdm(a):
        emb=net(i[0])
        xl.append(emb.data.cpu())
        yl.append(i[1].data.cpu())

# + jupyter={"outputs_hidden": true} tags=[]
tiao,cnt = 22,18
with torch.no_grad():
	for i in a:
		if tiao>0:
			tiao -=1
			continue
		cnt-=1
		if cnt<0:break
		i[0] = i[0].cpu()
		i[0] = (i[0]+1)/2
		plt.imshow(i[0][0].cpu().permute(1,2,0))
		plt.show()
		print(i[1])
# -

ans = 2
l = []
tag = 0
a = DataLoaderX(local_rank=0,dataset= train_set, batch_size =1,pin_memory=True)
with torch.no_grad():
    for i in a:
        if i[1].item() == tag: 
            tag+=1
            l.append(i[0])
            print(len(l) ,end = ', ')
            if(len(l)==50):
                a = torch.stack(l)
                a = (a + 1) / 2
                import torchvision
                print(a.shape)
                grid_img = torchvision.utils.make_grid(a.squeeze().cpu(), nrow=5)
                plt.figure(dpi=500)
                plt.imshow(grid_img.permute(1,2,0))
                plt.show()
#                 l = []
                break


# +

a = torch.stack([l[0],l[2],l[3],l[4],l[5],l[8],l[17],l[27]]).squeeze()
grid_img = torchvision.utils.make_grid(a.squeeze().cpu(), nrow=5,normalize=True)
b = net(a.cuda())
# plt.figure(dpi=500)
# plt.imshow(grid_img.permute(1,2,0))
# plt.show()
torch.save([a,b],'save/multi.pt')
# -

dir(a)



with torch.no_grad():
	for i in range(1,len(l)):
		print((l[0]*l[i]).sum())

# +
ans = 0

for i in a:
	if i[1].item() == 0: ans+=1
	else:break
ans
	# if last>0: last-=1
	# else:
	# 	break
	# 	# continue
	# print(i[1].item())
# -

len(train_set.imgidx)

train_set.imgidx

train_set.imgrec

len(train_set)

len(train_set.imgrec)








