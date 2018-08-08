# -*- coding: UTF-8 -*-
class DefaultConfig(object):
    env = 'default'
    train_images = './data/images' #训练图片
    train_masks = './data/manual1' #训练标签
    epoch = 50 #迭代次数
    batch_size = 3 #批处理大小
    lr_decay = 0.95   
    weight_decay = 5 * 10 ** -4
    momentum = 0.9
    crop_jiange = 100 #分割数据集的间隔
    img_rsize = 512 #输入网络图片的分辨率
    img_csize = 512 #
    input_channel = 3 #输入图片的通道数
    cls_num = 1 #输出图片的通道数
    
    learning_rate = 0.1 #初始学习率
opt =DefaultConfig()
