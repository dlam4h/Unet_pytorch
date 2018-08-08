# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

class Unet(nn.Module):
    def __init__(self, input_channel = 3, cls_num = 1):
        super(Unet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxPool1 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.maxPool2 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.maxPool3 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.maxPool4 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv9 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
 

        self.upsampconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0)
        self.conv11 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
 
        self.upsampconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        self.conv13 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
 
        self.upsampconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.conv15 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv16 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
 
        self.upsampconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.conv17 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv18 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
 
        self.conv19 = nn.Conv2d(64, cls_num, 1, stride=1, padding=0)
        self.init_weights()

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_copy1_2 = x
        x = self.maxPool1(x)
 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x_copy3_4 = x
        x = self.maxPool2(x)
 
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x_copy5_6 = x
        x = self.maxPool3(x)
 
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.dropout(x, 0.5)
        x_copy7_8 = x
        x = self.maxPool4(x)
 
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.dropout(x, 0.5)

        x = F.relu(self.upsampconv1(x))
        x = torch.cat((x, x_copy7_8), 1)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
 
        x = F.relu(self.upsampconv2(x))
        x = torch.cat((x, x_copy5_6), 1)
        x = F.relu(self.conv13(x))
        x = F.relu(self.conv14(x))
 
        x = F.relu(self.upsampconv3(x))
        x = torch.cat((x, x_copy3_4), 1)
        x = F.relu(self.conv15(x))
        x = F.relu(self.conv16(x))
 
        x = F.relu(self.upsampconv4(x))
        x = torch.cat((x, x_copy1_2), 1)
        x = F.relu(self.conv17(x))
        x = F.relu(self.conv18(x))
        x = self.conv19(x)

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

class Unet_bn(nn.Module):
    def __init__(self, input_channel = 3, cls_num = 1):
        super(Unet_bn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxPool1 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.maxPool2 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(256)
        self.maxPool3 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv8_bn = nn.BatchNorm2d(512)
        self.maxPool4 = nn.MaxPool2d(2, stride=2, padding=0)
 
        self.conv9 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
        self.conv9_bn = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.conv10_bn = nn.BatchNorm2d(1024)
 

        self.upsampconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0)
 
        self.conv11 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        self.conv11_bn = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv12_bn = nn.BatchNorm2d(512)
 
        self.upsampconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
 
        self.conv13 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.conv13_bn = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv14_bn = nn.BatchNorm2d(256)
 
        self.upsampconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
 
        self.conv15 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv15_bn = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv16_bn = nn.BatchNorm2d(128)
 
        self.upsampconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
 
        self.conv17 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv17_bn = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv18_bn = nn.BatchNorm2d(64)
 
        self.conv19 = nn.Conv2d(64, cls_num, 1, stride=1, padding=0)
        self.conv19_bn = nn.BatchNorm2d(cls_num)
        self.init_weights()

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x_copy1_2 = x
        x = self.maxPool1(x)
 
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x_copy3_4 = x
        x = self.maxPool2(x)
 
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x_copy5_6 = x
        x = self.maxPool3(x)
 
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.dropout(x, 0.5)
        x_copy7_8 = x
        x = self.maxPool4(x)
 
        x = F.relu(self.conv9_bn(self.conv9(x)))
        x = F.relu(self.conv10_bn(self.conv10(x)))
        x = F.dropout(x, 0.5)

        x = F.relu(self.upsampconv1(x))
        x = torch.cat((x, x_copy7_8), 1)
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
 
        x = F.relu(self.upsampconv2(x))
        x = torch.cat((x, x_copy5_6), 1)
        x = F.relu(self.conv13_bn(self.conv13(x)))
        x = F.relu(self.conv14_bn(self.conv14(x)))
 
        x = F.relu(self.upsampconv3(x))
        x = torch.cat((x, x_copy3_4), 1)
        x = F.relu(self.conv15_bn(self.conv15(x)))
        x = F.relu(self.conv16_bn(self.conv16(x)))
 
        x = F.relu(self.upsampconv4(x))
        x = torch.cat((x, x_copy1_2), 1)
        x = F.relu(self.conv17_bn(self.conv17(x)))
        x = F.relu(self.conv18_bn(self.conv18(x)))

        x = self.conv19_bn(self.conv19(x))
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)