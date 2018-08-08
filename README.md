# Unet_pytorch for retina images
## 使用Unet对视网膜血管图片进行分割  
数据集 原图为RGB图，分辨率为3504×2336，自动分块成统一大小的图片，标签为二值图  
支持多分类  
## 使用说明  
config.py 参数设置  
dataset.py 数据集加载（根据需要修改）  
logger.py 日志  
model.py 模型定义  
predict.py 加载模型分割图片  
train.py 训练文件
