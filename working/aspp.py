from resnet import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class ASPPConv(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(ASPPConv, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.module(x)
        return x

class ASPPPool(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPPPool, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = self.module(x)
        x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel=256,rate=[6,12,18], dropout_rate=0):
        super(ASPP, self).__init__()

        self.atrous0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.atrous1 = ASPPConv(in_channel, out_channel, rate[0])
        self.atrous2 = ASPPConv(in_channel, out_channel, rate[1])
        self.atrous3 = ASPPConv(in_channel, out_channel, rate[2])
        self.atrous4 = ASPPPool(in_channel, out_channel)

        self.combine = nn.Sequential(
            nn.Conv2d(5 * out_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):

        x = torch.cat([
            self.atrous0(x),
            self.atrous1(x),
            self.atrous2(x),
            self.atrous3(x),
            self.atrous4(x),
        ],1)
        x = self.combine(x)
        return x


#------
def resize_like(x, reference, mode='bilinear'):
    if x.shape[2:] !=  reference.shape[2:]:
        if mode=='bilinear':
            x = F.interpolate(x, size=reference.shape[2:],mode='bilinear',align_corners=False)
        if mode=='nearest':
            x = F.interpolate(x, size=reference.shape[2:],mode='nearest')
    return x

def fuse(x, mode='cat'):
    batch_size,C0,H0,W0 = x[0].shape

    for i in range(1,len(x)):
        _,_,H,W = x[i].shape
        if (H,W)!=(H0,W0):
            x[i] = F.interpolate(x[i], size=(H0,W0), mode='bilinear', align_corners=False)

    if mode=='cat':
        return torch.cat(x,1)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size, stride, padding, dilation, groups=in_channel, bias=bias)
        self.bn   = nn.BatchNorm2d(in_channel)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


#JPU
class JointPyramidUpsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(JointPyramidUpsample, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channel[0], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel[1], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel[2], out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        #-------------------------------

        self.dilation0 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation1 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation2 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dilation3 = nn.Sequential(
            SeparableConv2d(3*out_channel, out_channel, kernel_size=3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x0 = self.conv0(x[0])
        x1 = self.conv1(x[1])
        x2 = self.conv2(x[2])

        x0 = resize_like(x0, x2, mode='nearest')
        x1 = resize_like(x1, x2, mode='nearest')
        x = torch.cat([x0,x1,x2], dim=1)

        d0 = self.dilation0(x)
        d1 = self.dilation1(x)
        d2 = self.dilation2(x)
        d3 = self.dilation3(x)
        x = torch.cat([d0,d1,d2,d3], dim=1)
        return x

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

class aspp(nn.Module):
    #def load_pretrain(self, skip=['logit.'], is_print=True):
    #    load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)

    def __init__(self, num_class=4):
        super(aspp, self).__init__()

        e = ResNet34()
        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None  #dropped

        self.jpu = JointPyramidUpsample([512,256,128],128)
        #self.aspp = ASPP(512, 128, rate=[6,12,18], dropout_rate=0.1)
        self.aspp = ASPP(512, 128, rate=[4,8,12], dropout_rate=0.1)
        self.logit = nn.Conv2d(128,num_class,kernel_size=1)


    def forward(self, x):
        batch_size,C,H,W = x.shape

        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x = self.jpu([x4,x3,x2])
        x = self.aspp(x)
        logit = self.logit(x)

        #---
        probability_mask  = torch.sigmoid(logit)
        probability_label = F.adaptive_max_pool2d(probability_mask,1).view(batch_size,-1)
        return probability_mask
