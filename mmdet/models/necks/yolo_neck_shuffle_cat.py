# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import NECKS


class FastGlobalAvgPool2d(nn.Module):
    """
    TResNet: High Performance GPU-Dedicated Architecture
    https://github.com/mrT23/TResNet/blob/master/src/models/tresnet/layers/avg_pool.py
    Input: [N, C, H, W]
    Output: [N, C] or [N, C, 1, 1]
    args:
    flatten: True, [N, C] ; False, [N, C, 1, 1]
    """

    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class SE(nn.Module):
    def __init__(self, c, reduction=16):
        super(SE, self).__init__()
        self.conv1 = nn.Conv2d(c, c // reduction, kernel_size=1, bias=True)
        self.ac1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c // reduction, c, kernel_size=1, bias=True)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        return 2 * self.ac2(self.conv2(self.ac1(self.conv1(x))))


class Shuffle_Cat(nn.Module):
    def __init__(self, c, c1, c2, g=16, reduction=16, conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        """
        args:
            c: 输出维度
            c1: low 输入维度
            c2: high 输入维度
        """
        super(Shuffle_Cat, self).__init__()
        self.GlobalAvgPool2d = FastGlobalAvgPool2d(flatten=False) # [N, C, 1, 1]
        self.g = g
        # self.c = c
        # self.c1 = c1
        # self.c2 = c2
        c_ = c1 + c2
        # self.conv1 = nn.Conv2d(c_, c_ // reduction, kernel_size=1, bias=True, groups=2)
        # self.ac1 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(c_ // reduction, c_, kernel_size=1, bias=True, groups=2)
        # self.ac2 = nn.Sigmoid()

        self.low_se = SE(c1, reduction)
        self.high_se = SE(c2, reduction)

        # self.conv3 = DWConv(c_, c, k=3)
        # 将原先的2个GCONV合成一个
        # self.conv3 = nn.Sequential(nn.Conv2d(c_, c_, 1, 1, groups=self.g, bias=False),
        #                            nn.BatchNorm2d(c_),
        #                            nn.LeakyReLU(inplace=True))
        # self.conv4 = nn.Sequential(nn.Conv2d(c_, c, 1, 1, groups=self.g, bias=False),
        #                            nn.BatchNorm2d(c),
        #                            nn.LeakyReLU(inplace=True))

        # self.conv3 = nn.Sequential(nn.Conv2d(c_, c, 1, 1, groups=self.g, bias=False),
        #                            nn.BatchNorm2d(c),
        #                            nn.LeakyReLU(inplace=True))

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.low_conv = ConvModule(c1, c1, 3, padding=1, **cfg)
        self.hight_conv = ConvModule(c2, c2, 1, **cfg)

        self.conv3 = ConvModule(c_, c, 1, groups=self.g, **cfg)

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % groups == 0)
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        # low: [N, C1, H, W], high: [N, C2, H, W]
        x1, x2 = x
        x1 = self.low_conv(x1)
        x2 = self.hight_conv(x2)

        x1 *= self.low_se(self.GlobalAvgPool2d(x1))
        x2 *= self.high_se(self.GlobalAvgPool2d(x2))

        x = torch.cat([x1, x2], dim=1)
        # [N, C1+C2, 1, 1]
        # x_gp = 2 * self.ac2(self.conv2(self.ac1(self.conv1(self.GlobalAvgPool2d(x)))))
        # x *= x_gp

        x = self.channel_shuffle(x, groups=self.g)
        # x = self.conv4(self.conv3(x))
        x = self.conv3(x)
        return x



class DetectionBlock(nn.Module):
    """Detection block in YOLO neck.

    Let out_channels = n, the DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
        1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n.
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(DetectionBlock, self).__init__()
        double_out_channels = out_channels * 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(in_channels, out_channels, 1, **cfg)
        self.conv2 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv3 = ConvModule(double_out_channels, out_channels, 1, **cfg)
        self.conv4 = ConvModule(
            out_channels, double_out_channels, 3, padding=1, **cfg)
        self.conv5 = ConvModule(double_out_channels, out_channels, 1, **cfg)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.conv2(tmp)
        tmp = self.conv3(tmp)
        tmp = self.conv4(tmp)
        out = self.conv5(tmp)
        return out


@NECKS.register_module()
class ShuffleCatNeck(nn.Module):
    """The neck of YOLOV3.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(ShuffleCatNeck, self).__init__()
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # in_channels = [1024, 512, 256]
        # out_channels = [512, 256, 128]

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        self.detect1 = DetectionBlock(in_channels[0], out_channels[0], **cfg)
        in_c, out_c = [], []
        for i in range(1, self.num_scales):
            # in_c, out_c = self.in_channels[i], self.out_channels[i]
            in_c.append(self.in_channels[i])
            out_c.append(self.out_channels[i])
            self.add_module(f'conv{i}', ConvModule(in_c[i-1], out_c[i-1], 1, **cfg))

            self.add_module(f'shuffle_cat{i}', Shuffle_Cat(out_c[i-1]+in_c[i-1],
                                                           out_c[i-1], in_c[i-1]))

            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(out_c[i-1]+in_c[i-1], out_c[i-1], **cfg))

    def forward(self, feats):
        assert len(feats) == self.num_scales
        # p3-p4-p5
        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1]) # p5
        outs.append(out)

        # p4 p3
        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)

            # Cat with low-lvl feats
            # tmp: low; x: high
            tmp = F.interpolate(tmp, scale_factor=2)
            # print('tmp', tmp.shape)
            # print('x', x.shape)
            # tmp = torch.cat((tmp, x), 1)
            # 使用shuffle cat替换普通的cat
            shuffle_cat = getattr(self, f'shuffle_cat{i+1}')
            tmp = shuffle_cat((tmp, x))

            # tmp = Shuffle_Cat()
            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        return tuple(outs)

    def init_weights(self):
        """Initialize the weights of module."""
        # init is done in ConvModule
        pass
