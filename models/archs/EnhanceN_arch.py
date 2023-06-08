"""
## ACMMM 2022
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.archs.CDC import cdcconv
from models.archs.arch_util import Refine



class ProcessBlock(nn.Module):
    def __init__(self, nc, nc_out):
        super(ProcessBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1))
        # self.cdc =cdcconv(nc,nc)
        # self.fuse = nn.Conv2d(2*nc,nc_out,1,1,0)

    def forward(self, x):
        x_conv = self.conv(x)+x
        # x_cdc = self.cdc(x)
        # x_out = self.fuse(torch.cat([x_conv,x_cdc],1))

        return x_conv



class DualBlock(nn.Module):
    def __init__(self, nc, nc_out):
        super(DualBlock,self).__init__()
        self.relu = nn.ReLU()
        self.norm = nn.Sequential(nn.InstanceNorm2d(nc,affine=True),
                                  nn.Conv2d(nc,nc,1,1,0), nn.LeakyReLU(0.1))
        self.prcessblock = InvBlock(nc,nc_out)
        self.fuse1 = nn.Conv2d(2*nc,nc,1,1,0)
        self.fuse2 = nn.Conv2d(2*nc,nc,1,1,0)
        self.post = nn.Sequential(nn.Conv2d(2*nc,nc,3,1,1),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv2d(nc,nc,3,1,1))

    def forward(self, x):
        x_norm = self.norm(x)
        x_p = self.relu(x)
        x_n = self.relu(-x)
        x_p = self.prcessblock(x_p)
        x_n = -self.prcessblock(x_n)
        x_p1 = self.fuse1(torch.cat([x_norm,x_p], 1))
        x_n1 = self.fuse2(torch.cat([x_norm,x_n], 1))
        x_out = self.post(torch.cat([x_p1,x_n1],1))


        return x_out+x


# class INBlock(nn.Module):
#     def __init__(self, nc, nc_out):
#         super(INBlock, self).__init__()
#         self.norm = nn.InstanceNorm2d(nc,affine=True)
#         self.post = nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1),
#                                   nn.LeakyReLU(0.1),
#                                   nn.Conv2d(nc, nc_out, 3, 1, 1))
#
#     def forward(self, x):
#         x_norm = self.norm(x)
#         x_out = self.post(x_norm)
#
#         return x_out



class DualProcess(nn.Module):
    def __init__(self, nc):
        super(DualProcess,self).__init__()
        self.conv1 = DualBlock(nc,nc//2)
        self.conv2 = DualBlock(nc,nc//2)
        self.conv3 = DualBlock(nc,nc//2)
        self.conv4 = DualBlock(nc,nc//2)
        self.conv5 = DualBlock(nc,nc//2)
        self.cat = nn.Conv2d(5 * nc, nc, 1, 1, 0)
        self.refine = Refine(nc,3)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        xout = self.cat(torch.cat([x1,x2,x3,x4,x5],1))
        xfinal = self.refine(xout)

        return xfinal,xout



class InteractNet(nn.Module):
    def __init__(self, nc):
        super(InteractNet,self).__init__()
        self.extract = nn.Conv2d(3,nc,3,1,1)
        self.dualprocess = DualProcess(nc)

    def forward(self, x):
        x_pre = self.extract(x)
        xout,feature = self.dualprocess(x_pre)

        return torch.clamp(xout+0.00001,0.0,1.0),feature





class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = cdcconv(self.split_len2, self.split_len1)
        self.G = cdcconv(self.split_len1, self.split_len2)
        self.H = ProcessBlock(self.split_len1, self.split_len2)

        #in_channels = 3
        # self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        if not rev:
            # invert1x1conv
            # x, logdet = self.flow_permutation(x, logdet=0, rev=False)

            # split to 1 channel and 2 channel.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            out = torch.cat((y1, y2), 1)
        else:
            # split.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

            x = torch.cat((y1, y2), 1)
            out = x
            # inv permutation
            # out, logdet = self.flow_permutation(x, logdet=0, rev=True)

        return out