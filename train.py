# -*- coding: UTF-8 -*-
import torch
import datetime
import os
from logger import LogWay
from config import opt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Unet, Unet_bn
from dataset import My_Dataset
import torch.nn.functional as F

use_gpu = torch.cuda.is_available()
print(use_gpu)
if use_gpu:
    torch.cuda.set_device(0)

def adjusting_rate(optimizer,learning_rate,epoch):
    dlam = 5
    if epoch%dlam == 0:
        lr = learning_rate*0.1**(epoch//dlam)
        for parm_group in optimizer.param_groups:
            parm_group['lr']=lr

def train():
    model = Unet(input_channel = opt.input_channel, cls_num = opt.cls_num)
    model_name = 'Unet_bn'
    train_logger = LogWay(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + '.txt')
    train_data = My_Dataset(opt.train_images, opt.train_masks)
    train_dataloader = DataLoader(train_data,batch_size = opt.batch_size, shuffle = True, num_workers = 0)

    if opt.cls_num == 1:
        criterion = torch.nn.BCELoss()
    else:
        criterion = torch.nn.NLLLoss()
    if use_gpu:
        model.cuda()
        if opt.cls_num == 1:
            criterion = torch.nn.BCELoss().cuda()
        else:
            criterion = torch.nn.NLLLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    for epoch in range(opt.epoch):
        loss_sum=0
        for i,(data,target) in enumerate(train_dataloader):
            data = Variable(data)
            target = Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            outputs = model(data)

            if opt.cls_num == 1:
                outputs = F.sigmoid(outputs).view(-1)
                mask_true = target.view(-1)
                loss = criterion(outputs,mask_true)
            else:
                outputs = F.LogSoftmax(outputs, dim=1)
                loss = criterion(outputs, target)

            loss_sum = loss_sum + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch:{} batch:{} loss:{}".format(epoch+1,i,loss.item()))
        info = 'Time:{}    Epoch:{}    Loss_avg:{}\n'.format(str(datetime.datetime.now()), epoch+1, loss_sum/(i+1))
        train_logger.add(info)
        adjusting_rate(optimizer,opt.learning_rate,epoch+1)
        realepoch = epoch + 1
        if(realepoch%10==0):
            save_name = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')+' '+model_name+str(realepoch)+'.pt'
            torch.save(model.state_dict(),save_name)

if __name__ == '__main__':
    train()