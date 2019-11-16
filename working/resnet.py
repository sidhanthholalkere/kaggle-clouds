import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#from lib.net.sync_bn.nn import BatchNorm2dSync as SynchronizedBatchNorm2d
#BatchNorm2d = SynchronizedBatchNorm2d
BatchNorm2d = nn.BatchNorm2d


IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]


###############################################################################
PRETRAIN_FILE = \
    '/root/share/project/kaggle/2019/steel/data/pretrain_model/resnet34-333f7ec4.pth'

CONVERSION=[
 'block0.0.weight',	(64, 3, 7, 7),	 'conv1.weight',	(64, 3, 7, 7),
 'block0.1.weight',	(64,),	 'bn1.weight',	(64,),
 'block0.1.bias',	(64,),	 'bn1.bias',	(64,),
 'block0.1.running_mean',	(64,),	 'bn1.running_mean',	(64,),
 'block0.1.running_var',	(64,),	 'bn1.running_var',	(64,),
 'block1.1.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv1.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn1.bn.weight',	(64,),	 'layer1.0.bn1.weight',	(64,),
 'block1.1.conv_bn1.bn.bias',	(64,),	 'layer1.0.bn1.bias',	(64,),
 'block1.1.conv_bn1.bn.running_mean',	(64,),	 'layer1.0.bn1.running_mean',	(64,),
 'block1.1.conv_bn1.bn.running_var',	(64,),	 'layer1.0.bn1.running_var',	(64,),
 'block1.1.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.0.conv2.weight',	(64, 64, 3, 3),
 'block1.1.conv_bn2.bn.weight',	(64,),	 'layer1.0.bn2.weight',	(64,),
 'block1.1.conv_bn2.bn.bias',	(64,),	 'layer1.0.bn2.bias',	(64,),
 'block1.1.conv_bn2.bn.running_mean',	(64,),	 'layer1.0.bn2.running_mean',	(64,),
 'block1.1.conv_bn2.bn.running_var',	(64,),	 'layer1.0.bn2.running_var',	(64,),
 'block1.2.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv1.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn1.bn.weight',	(64,),	 'layer1.1.bn1.weight',	(64,),
 'block1.2.conv_bn1.bn.bias',	(64,),	 'layer1.1.bn1.bias',	(64,),
 'block1.2.conv_bn1.bn.running_mean',	(64,),	 'layer1.1.bn1.running_mean',	(64,),
 'block1.2.conv_bn1.bn.running_var',	(64,),	 'layer1.1.bn1.running_var',	(64,),
 'block1.2.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.1.conv2.weight',	(64, 64, 3, 3),
 'block1.2.conv_bn2.bn.weight',	(64,),	 'layer1.1.bn2.weight',	(64,),
 'block1.2.conv_bn2.bn.bias',	(64,),	 'layer1.1.bn2.bias',	(64,),
 'block1.2.conv_bn2.bn.running_mean',	(64,),	 'layer1.1.bn2.running_mean',	(64,),
 'block1.2.conv_bn2.bn.running_var',	(64,),	 'layer1.1.bn2.running_var',	(64,),
 'block1.3.conv_bn1.conv.weight',	(64, 64, 3, 3),	 'layer1.2.conv1.weight',	(64, 64, 3, 3),
 'block1.3.conv_bn1.bn.weight',	(64,),	 'layer1.2.bn1.weight',	(64,),
 'block1.3.conv_bn1.bn.bias',	(64,),	 'layer1.2.bn1.bias',	(64,),
 'block1.3.conv_bn1.bn.running_mean',	(64,),	 'layer1.2.bn1.running_mean',	(64,),
 'block1.3.conv_bn1.bn.running_var',	(64,),	 'layer1.2.bn1.running_var',	(64,),
 'block1.3.conv_bn2.conv.weight',	(64, 64, 3, 3),	 'layer1.2.conv2.weight',	(64, 64, 3, 3),
 'block1.3.conv_bn2.bn.weight',	(64,),	 'layer1.2.bn2.weight',	(64,),
 'block1.3.conv_bn2.bn.bias',	(64,),	 'layer1.2.bn2.bias',	(64,),
 'block1.3.conv_bn2.bn.running_mean',	(64,),	 'layer1.2.bn2.running_mean',	(64,),
 'block1.3.conv_bn2.bn.running_var',	(64,),	 'layer1.2.bn2.running_var',	(64,),
 'block2.0.conv_bn1.conv.weight',	(128, 64, 3, 3),	 'layer2.0.conv1.weight',	(128, 64, 3, 3),
 'block2.0.conv_bn1.bn.weight',	(128,),	 'layer2.0.bn1.weight',	(128,),
 'block2.0.conv_bn1.bn.bias',	(128,),	 'layer2.0.bn1.bias',	(128,),
 'block2.0.conv_bn1.bn.running_mean',	(128,),	 'layer2.0.bn1.running_mean',	(128,),
 'block2.0.conv_bn1.bn.running_var',	(128,),	 'layer2.0.bn1.running_var',	(128,),
 'block2.0.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.0.conv2.weight',	(128, 128, 3, 3),
 'block2.0.conv_bn2.bn.weight',	(128,),	 'layer2.0.bn2.weight',	(128,),
 'block2.0.conv_bn2.bn.bias',	(128,),	 'layer2.0.bn2.bias',	(128,),
 'block2.0.conv_bn2.bn.running_mean',	(128,),	 'layer2.0.bn2.running_mean',	(128,),
 'block2.0.conv_bn2.bn.running_var',	(128,),	 'layer2.0.bn2.running_var',	(128,),
 'block2.0.shortcut.conv.weight',	(128, 64, 1, 1),	 'layer2.0.downsample.0.weight',	(128, 64, 1, 1),
 'block2.0.shortcut.bn.weight',	(128,),	 'layer2.0.downsample.1.weight',	(128,),
 'block2.0.shortcut.bn.bias',	(128,),	 'layer2.0.downsample.1.bias',	(128,),
 'block2.0.shortcut.bn.running_mean',	(128,),	 'layer2.0.downsample.1.running_mean',	(128,),
 'block2.0.shortcut.bn.running_var',	(128,),	 'layer2.0.downsample.1.running_var',	(128,),
 'block2.1.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv1.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn1.bn.weight',	(128,),	 'layer2.1.bn1.weight',	(128,),
 'block2.1.conv_bn1.bn.bias',	(128,),	 'layer2.1.bn1.bias',	(128,),
 'block2.1.conv_bn1.bn.running_mean',	(128,),	 'layer2.1.bn1.running_mean',	(128,),
 'block2.1.conv_bn1.bn.running_var',	(128,),	 'layer2.1.bn1.running_var',	(128,),
 'block2.1.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.1.conv2.weight',	(128, 128, 3, 3),
 'block2.1.conv_bn2.bn.weight',	(128,),	 'layer2.1.bn2.weight',	(128,),
 'block2.1.conv_bn2.bn.bias',	(128,),	 'layer2.1.bn2.bias',	(128,),
 'block2.1.conv_bn2.bn.running_mean',	(128,),	 'layer2.1.bn2.running_mean',	(128,),
 'block2.1.conv_bn2.bn.running_var',	(128,),	 'layer2.1.bn2.running_var',	(128,),
 'block2.2.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.2.conv1.weight',	(128, 128, 3, 3),
 'block2.2.conv_bn1.bn.weight',	(128,),	 'layer2.2.bn1.weight',	(128,),
 'block2.2.conv_bn1.bn.bias',	(128,),	 'layer2.2.bn1.bias',	(128,),
 'block2.2.conv_bn1.bn.running_mean',	(128,),	 'layer2.2.bn1.running_mean',	(128,),
 'block2.2.conv_bn1.bn.running_var',	(128,),	 'layer2.2.bn1.running_var',	(128,),
 'block2.2.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.2.conv2.weight',	(128, 128, 3, 3),
 'block2.2.conv_bn2.bn.weight',	(128,),	 'layer2.2.bn2.weight',	(128,),
 'block2.2.conv_bn2.bn.bias',	(128,),	 'layer2.2.bn2.bias',	(128,),
 'block2.2.conv_bn2.bn.running_mean',	(128,),	 'layer2.2.bn2.running_mean',	(128,),
 'block2.2.conv_bn2.bn.running_var',	(128,),	 'layer2.2.bn2.running_var',	(128,),
 'block2.3.conv_bn1.conv.weight',	(128, 128, 3, 3),	 'layer2.3.conv1.weight',	(128, 128, 3, 3),
 'block2.3.conv_bn1.bn.weight',	(128,),	 'layer2.3.bn1.weight',	(128,),
 'block2.3.conv_bn1.bn.bias',	(128,),	 'layer2.3.bn1.bias',	(128,),
 'block2.3.conv_bn1.bn.running_mean',	(128,),	 'layer2.3.bn1.running_mean',	(128,),
 'block2.3.conv_bn1.bn.running_var',	(128,),	 'layer2.3.bn1.running_var',	(128,),
 'block2.3.conv_bn2.conv.weight',	(128, 128, 3, 3),	 'layer2.3.conv2.weight',	(128, 128, 3, 3),
 'block2.3.conv_bn2.bn.weight',	(128,),	 'layer2.3.bn2.weight',	(128,),
 'block2.3.conv_bn2.bn.bias',	(128,),	 'layer2.3.bn2.bias',	(128,),
 'block2.3.conv_bn2.bn.running_mean',	(128,),	 'layer2.3.bn2.running_mean',	(128,),
 'block2.3.conv_bn2.bn.running_var',	(128,),	 'layer2.3.bn2.running_var',	(128,),
 'block3.0.conv_bn1.conv.weight',	(256, 128, 3, 3),	 'layer3.0.conv1.weight',	(256, 128, 3, 3),
 'block3.0.conv_bn1.bn.weight',	(256,),	 'layer3.0.bn1.weight',	(256,),
 'block3.0.conv_bn1.bn.bias',	(256,),	 'layer3.0.bn1.bias',	(256,),
 'block3.0.conv_bn1.bn.running_mean',	(256,),	 'layer3.0.bn1.running_mean',	(256,),
 'block3.0.conv_bn1.bn.running_var',	(256,),	 'layer3.0.bn1.running_var',	(256,),
 'block3.0.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.0.conv2.weight',	(256, 256, 3, 3),
 'block3.0.conv_bn2.bn.weight',	(256,),	 'layer3.0.bn2.weight',	(256,),
 'block3.0.conv_bn2.bn.bias',	(256,),	 'layer3.0.bn2.bias',	(256,),
 'block3.0.conv_bn2.bn.running_mean',	(256,),	 'layer3.0.bn2.running_mean',	(256,),
 'block3.0.conv_bn2.bn.running_var',	(256,),	 'layer3.0.bn2.running_var',	(256,),
 'block3.0.shortcut.conv.weight',	(256, 128, 1, 1),	 'layer3.0.downsample.0.weight',	(256, 128, 1, 1),
 'block3.0.shortcut.bn.weight',	(256,),	 'layer3.0.downsample.1.weight',	(256,),
 'block3.0.shortcut.bn.bias',	(256,),	 'layer3.0.downsample.1.bias',	(256,),
 'block3.0.shortcut.bn.running_mean',	(256,),	 'layer3.0.downsample.1.running_mean',	(256,),
 'block3.0.shortcut.bn.running_var',	(256,),	 'layer3.0.downsample.1.running_var',	(256,),
 'block3.1.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv1.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn1.bn.weight',	(256,),	 'layer3.1.bn1.weight',	(256,),
 'block3.1.conv_bn1.bn.bias',	(256,),	 'layer3.1.bn1.bias',	(256,),
 'block3.1.conv_bn1.bn.running_mean',	(256,),	 'layer3.1.bn1.running_mean',	(256,),
 'block3.1.conv_bn1.bn.running_var',	(256,),	 'layer3.1.bn1.running_var',	(256,),
 'block3.1.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.1.conv2.weight',	(256, 256, 3, 3),
 'block3.1.conv_bn2.bn.weight',	(256,),	 'layer3.1.bn2.weight',	(256,),
 'block3.1.conv_bn2.bn.bias',	(256,),	 'layer3.1.bn2.bias',	(256,),
 'block3.1.conv_bn2.bn.running_mean',	(256,),	 'layer3.1.bn2.running_mean',	(256,),
 'block3.1.conv_bn2.bn.running_var',	(256,),	 'layer3.1.bn2.running_var',	(256,),
 'block3.2.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.2.conv1.weight',	(256, 256, 3, 3),
 'block3.2.conv_bn1.bn.weight',	(256,),	 'layer3.2.bn1.weight',	(256,),
 'block3.2.conv_bn1.bn.bias',	(256,),	 'layer3.2.bn1.bias',	(256,),
 'block3.2.conv_bn1.bn.running_mean',	(256,),	 'layer3.2.bn1.running_mean',	(256,),
 'block3.2.conv_bn1.bn.running_var',	(256,),	 'layer3.2.bn1.running_var',	(256,),
 'block3.2.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.2.conv2.weight',	(256, 256, 3, 3),
 'block3.2.conv_bn2.bn.weight',	(256,),	 'layer3.2.bn2.weight',	(256,),
 'block3.2.conv_bn2.bn.bias',	(256,),	 'layer3.2.bn2.bias',	(256,),
 'block3.2.conv_bn2.bn.running_mean',	(256,),	 'layer3.2.bn2.running_mean',	(256,),
 'block3.2.conv_bn2.bn.running_var',	(256,),	 'layer3.2.bn2.running_var',	(256,),
 'block3.3.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.3.conv1.weight',	(256, 256, 3, 3),
 'block3.3.conv_bn1.bn.weight',	(256,),	 'layer3.3.bn1.weight',	(256,),
 'block3.3.conv_bn1.bn.bias',	(256,),	 'layer3.3.bn1.bias',	(256,),
 'block3.3.conv_bn1.bn.running_mean',	(256,),	 'layer3.3.bn1.running_mean',	(256,),
 'block3.3.conv_bn1.bn.running_var',	(256,),	 'layer3.3.bn1.running_var',	(256,),
 'block3.3.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.3.conv2.weight',	(256, 256, 3, 3),
 'block3.3.conv_bn2.bn.weight',	(256,),	 'layer3.3.bn2.weight',	(256,),
 'block3.3.conv_bn2.bn.bias',	(256,),	 'layer3.3.bn2.bias',	(256,),
 'block3.3.conv_bn2.bn.running_mean',	(256,),	 'layer3.3.bn2.running_mean',	(256,),
 'block3.3.conv_bn2.bn.running_var',	(256,),	 'layer3.3.bn2.running_var',	(256,),
 'block3.4.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.4.conv1.weight',	(256, 256, 3, 3),
 'block3.4.conv_bn1.bn.weight',	(256,),	 'layer3.4.bn1.weight',	(256,),
 'block3.4.conv_bn1.bn.bias',	(256,),	 'layer3.4.bn1.bias',	(256,),
 'block3.4.conv_bn1.bn.running_mean',	(256,),	 'layer3.4.bn1.running_mean',	(256,),
 'block3.4.conv_bn1.bn.running_var',	(256,),	 'layer3.4.bn1.running_var',	(256,),
 'block3.4.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.4.conv2.weight',	(256, 256, 3, 3),
 'block3.4.conv_bn2.bn.weight',	(256,),	 'layer3.4.bn2.weight',	(256,),
 'block3.4.conv_bn2.bn.bias',	(256,),	 'layer3.4.bn2.bias',	(256,),
 'block3.4.conv_bn2.bn.running_mean',	(256,),	 'layer3.4.bn2.running_mean',	(256,),
 'block3.4.conv_bn2.bn.running_var',	(256,),	 'layer3.4.bn2.running_var',	(256,),
 'block3.5.conv_bn1.conv.weight',	(256, 256, 3, 3),	 'layer3.5.conv1.weight',	(256, 256, 3, 3),
 'block3.5.conv_bn1.bn.weight',	(256,),	 'layer3.5.bn1.weight',	(256,),
 'block3.5.conv_bn1.bn.bias',	(256,),	 'layer3.5.bn1.bias',	(256,),
 'block3.5.conv_bn1.bn.running_mean',	(256,),	 'layer3.5.bn1.running_mean',	(256,),
 'block3.5.conv_bn1.bn.running_var',	(256,),	 'layer3.5.bn1.running_var',	(256,),
 'block3.5.conv_bn2.conv.weight',	(256, 256, 3, 3),	 'layer3.5.conv2.weight',	(256, 256, 3, 3),
 'block3.5.conv_bn2.bn.weight',	(256,),	 'layer3.5.bn2.weight',	(256,),
 'block3.5.conv_bn2.bn.bias',	(256,),	 'layer3.5.bn2.bias',	(256,),
 'block3.5.conv_bn2.bn.running_mean',	(256,),	 'layer3.5.bn2.running_mean',	(256,),
 'block3.5.conv_bn2.bn.running_var',	(256,),	 'layer3.5.bn2.running_var',	(256,),
 'block4.0.conv_bn1.conv.weight',	(512, 256, 3, 3),	 'layer4.0.conv1.weight',	(512, 256, 3, 3),
 'block4.0.conv_bn1.bn.weight',	(512,),	 'layer4.0.bn1.weight',	(512,),
 'block4.0.conv_bn1.bn.bias',	(512,),	 'layer4.0.bn1.bias',	(512,),
 'block4.0.conv_bn1.bn.running_mean',	(512,),	 'layer4.0.bn1.running_mean',	(512,),
 'block4.0.conv_bn1.bn.running_var',	(512,),	 'layer4.0.bn1.running_var',	(512,),
 'block4.0.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.0.conv2.weight',	(512, 512, 3, 3),
 'block4.0.conv_bn2.bn.weight',	(512,),	 'layer4.0.bn2.weight',	(512,),
 'block4.0.conv_bn2.bn.bias',	(512,),	 'layer4.0.bn2.bias',	(512,),
 'block4.0.conv_bn2.bn.running_mean',	(512,),	 'layer4.0.bn2.running_mean',	(512,),
 'block4.0.conv_bn2.bn.running_var',	(512,),	 'layer4.0.bn2.running_var',	(512,),
 'block4.0.shortcut.conv.weight',	(512, 256, 1, 1),	 'layer4.0.downsample.0.weight',	(512, 256, 1, 1),
 'block4.0.shortcut.bn.weight',	(512,),	 'layer4.0.downsample.1.weight',	(512,),
 'block4.0.shortcut.bn.bias',	(512,),	 'layer4.0.downsample.1.bias',	(512,),
 'block4.0.shortcut.bn.running_mean',	(512,),	 'layer4.0.downsample.1.running_mean',	(512,),
 'block4.0.shortcut.bn.running_var',	(512,),	 'layer4.0.downsample.1.running_var',	(512,),
 'block4.1.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv1.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn1.bn.weight',	(512,),	 'layer4.1.bn1.weight',	(512,),
 'block4.1.conv_bn1.bn.bias',	(512,),	 'layer4.1.bn1.bias',	(512,),
 'block4.1.conv_bn1.bn.running_mean',	(512,),	 'layer4.1.bn1.running_mean',	(512,),
 'block4.1.conv_bn1.bn.running_var',	(512,),	 'layer4.1.bn1.running_var',	(512,),
 'block4.1.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.1.conv2.weight',	(512, 512, 3, 3),
 'block4.1.conv_bn2.bn.weight',	(512,),	 'layer4.1.bn2.weight',	(512,),
 'block4.1.conv_bn2.bn.bias',	(512,),	 'layer4.1.bn2.bias',	(512,),
 'block4.1.conv_bn2.bn.running_mean',	(512,),	 'layer4.1.bn2.running_mean',	(512,),
 'block4.1.conv_bn2.bn.running_var',	(512,),	 'layer4.1.bn2.running_var',	(512,),
 'block4.2.conv_bn1.conv.weight',	(512, 512, 3, 3),	 'layer4.2.conv1.weight',	(512, 512, 3, 3),
 'block4.2.conv_bn1.bn.weight',	(512,),	 'layer4.2.bn1.weight',	(512,),
 'block4.2.conv_bn1.bn.bias',	(512,),	 'layer4.2.bn1.bias',	(512,),
 'block4.2.conv_bn1.bn.running_mean',	(512,),	 'layer4.2.bn1.running_mean',	(512,),
 'block4.2.conv_bn1.bn.running_var',	(512,),	 'layer4.2.bn1.running_var',	(512,),
 'block4.2.conv_bn2.conv.weight',	(512, 512, 3, 3),	 'layer4.2.conv2.weight',	(512, 512, 3, 3),
 'block4.2.conv_bn2.bn.weight',	(512,),	 'layer4.2.bn2.weight',	(512,),
 'block4.2.conv_bn2.bn.bias',	(512,),	 'layer4.2.bn2.bias',	(512,),
 'block4.2.conv_bn2.bn.running_mean',	(512,),	 'layer4.2.bn2.running_mean',	(512,),
 'block4.2.conv_bn2.bn.running_var',	(512,),	 'layer4.2.bn2.running_var',	(512,),
 'logit.weight',	(1000, 512),	 'fc.weight',	(1000, 512),
 'logit.bias',	(1000,),	 'fc.bias',	(1000,),

]

def load_pretrain(net, skip=[], pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=True):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(conversion).reshape(-1,4)
    for key,_,pretrain_key,_ in conversion:
        if any(s in key for s in
            ['.num_batches_tracked',]+skip):
            continue

        #print('\t\t',key)
        if is_print:
            print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
                key, str(state_dict[key].shape),
                pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
            ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    #---
    if 1:
        state_dict['block0.0.weight'], state_dict['block0.0.bias'] = \
            absorb_rgb_normalisation_to_conv(state_dict['block0.0.weight'], state_dict['block0.0.bias'])


    #---
    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')




def absorb_rgb_normalisation_to_conv( weight, bias, rgb_mean=IMAGE_RGB_MEAN, rgb_std=IMAGE_RGB_STD ):
    out,c,h,w = weight.shape
    u = torch.from_numpy( np.array(rgb_mean,np.float32).reshape(1,3,1,1)).to(weight.device)
    s = torch.from_numpy( np.array(rgb_std ,np.float32).reshape(1,3,1,1)).to(weight.device)

    norm_weight = weight/s
    norm_bias   =-u*weight/s
    norm_bias   = norm_bias.sum(dim=[1,2,3]) + bias.to(weight.device)

    return norm_weight,norm_bias


###############################################################################
class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, stride=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x




#############  resnext50 pyramid feature net #######################################
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py


# bottleneck type C
class BasicBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, is_shortcut=False):
        super(BasicBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,    channel, kernel_size=3, padding=1, stride=stride)
        self.conv_bn2 = ConvBn2d(   channel,out_channel, kernel_size=3, padding=1, stride=1)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = self.conv_bn2(z)

        if self.is_shortcut:
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z




class ResNet34(nn.Module):

    def __init__(self, num_class=1000 ):
        super(ResNet34, self).__init__()


        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=True),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block0[0].bias.data.fill_(0.0)

        self.block1  = nn.Sequential(
             nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
             BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,),
          * [BasicBlock( 64, 64, 64, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.block2  = nn.Sequential(
             BasicBlock( 64,128,128, stride=2, is_shortcut=True, ),
          * [BasicBlock(128,128,128, stride=1, is_shortcut=False,) for i in range(1,4)],
        )
        self.block3  = nn.Sequential(
             BasicBlock(128,256,256, stride=2, is_shortcut=True, ),
          * [BasicBlock(256,256,256, stride=1, is_shortcut=False,) for i in range(1,6)],
        )
        self.block4 = nn.Sequential(
             BasicBlock(256,512,512, stride=2, is_shortcut=True, ),
          * [BasicBlock(512,512,512, stride=1, is_shortcut=False,) for i in range(1,3)],
        )
        self.logit = nn.Linear(512,num_class)



    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    net = ResNet34()
    load_pretrain(net, is_print=True)

    #---
    if 0:
        print(net)
        print('')

        print('*** print key *** ')
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        #keys = sorted(keys)
        for k in keys:
            if any(s in k for s in [
                'num_batches_tracked'
                # '.kernel',
                # '.gamma',
                # '.beta',
                # '.running_mean',
                # '.running_var',
            ]):
                continue

            p = state_dict[k].data.cpu().numpy()
            print(' \'%s\',\t%s,'%(k,tuple(p.shape)))
        print('')
        exit(0)

    #---
    if 1:

        net = net.cuda().eval()

        image_dir ='/root/share/data/imagenet/dummy/256x256'
        for f in [
            'great_white_shark','screwdriver','ostrich','blad_eagle','english_foxhound','goldfish',
        ]:
            image_file = image_dir +'/%s.jpg'%f
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            #image = cv2.resize(image,dsize=(224,224))
            #image = image[16:16+224,16:16+224]

            image = image[:,:,::-1]
            image = image.astype(np.float32)/255
            #image = (image -IMAGE_RGB_MEAN)/IMAGE_RGB_STD
            input = image.transpose(2,0,1)


            input = torch.from_numpy(input).float().cuda().unsqueeze(0)

            logit = net(input)
            proability = F.softmax(logit,-1)

            probability = proability.data.cpu().numpy().reshape(-1)
            argsort = np.argsort(-probability)

            print(f, image.shape)
            print(probability[:5])
            for t in range(5):
                print(t, '%5d'%argsort[t], probability[argsort[t]])
            print('')

            pass

    print('\nsucess!')

##############################################################################3


'''
great_white_shark (256, 256, 3)
[1.7603808e-09 4.4697329e-10 9.1863245e-01 8.0922067e-02 4.3487264e-04]
0     2 0.91863245
1     3 0.08092207
2     4 0.00043487264
3     5 2.7358706e-06
4   391 1.987833e-06

screwdriver (256, 256, 3)
[1.6742923e-08 6.9742768e-08 2.0134678e-07 5.2668611e-08 1.1299638e-07]
0   784 0.6263062
1   845 0.37200174
2   813 0.00014389477
3   772 0.00010553924
4   623 9.4616575e-05

ostrich (256, 256, 3)
[2.4325630e-08 1.1584205e-08 3.1243427e-08 5.4277020e-08 6.9122883e-08]
0     9 0.98532593
1   138 0.012045922
2   134 0.00094926864
3    23 0.0002838216
4    99 0.0002217516

blad_eagle (256, 256, 3)
[1.0544086e-09 1.0391727e-09 3.3691472e-10 2.5403588e-10 5.3890319e-09]
0    22 0.9996056
1   146 0.00015696019
2    21 0.00012857928
3    23 7.8184166e-05
4   127 8.167957e-06

english_foxhound (256, 256, 3)
[2.2772488e-06 1.6688903e-06 1.3474945e-06 1.4425848e-06 8.0607111e-07]
0   167 0.47306925
1   166 0.30223784
2   162 0.17834836
3   164 0.008347732
4   163 0.0064110514

goldfish (256, 256, 3)
[7.3288182e-05 9.9819857e-01 5.0156007e-07 1.3255191e-06 1.9134752e-06]
0     1 0.99819857
1    88 0.0003081657
2    29 0.00020368102
3    90 0.00014414938
4   392 9.927917e-05
'''







































