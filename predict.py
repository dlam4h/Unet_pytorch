# -*- coding: UTF-8 -*-
import torch
from model import Unet, Unet_bn
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
import glob

use_gpu = torch.cuda.is_available()
print(use_gpu)
if use_gpu:
    torch.cuda.set_device(0)

model = Unet_bn()
model_weight = '2018-08-04 06-23-38 Unet_bn.pt'
if use_gpu:
    model.cuda()
    model.load_state_dict(torch.load(model_weight))
else:
    model.load_state_dict(torch.load(model_weight,map_location='cpu'))

data_path = './test_data/test'
save_path = './test_data/test'
img_type = 'jpg'

img_list = glob.glob(data_path+"/*."+img_type)

for imgname in img_list:
    midname = imgname.split('\\')[-1]
    img = Image.open(data_path+"/"+midname)
    img = np.array(img,dtype=np.float32)
    img = np.transpose(img,(2,0,1))
    img = torch.from_numpy(img).unsqueeze(0)
    if use_gpu:
        img = img.cuda()
        outputs = model(img)
        outputs = F.sigmoid(outputs).squeeze(0).squeeze().cpu().detach().numpy()
    else:
        outputs = model(img)
        outputs = F.sigmoid(outputs).squeeze(0).squeeze().data.numpy()
    dlam = Image.fromarray((outputs * 255).astype(np.uint8))
    # dlam.show()

    # print(save_path+'/'+midname.split('.')[0]+'.tif')
    dlam.save(save_path+'/'+midname.split('.')[0]+'.tif')
    print(midname.split('.')[0]+'.tif')

# outputs = model(Variable(img,volatile=True))
# image = outputs.detach().numpy()
# a=Image.fromarray(int(image[0]))
# a.save('a.jpg')
