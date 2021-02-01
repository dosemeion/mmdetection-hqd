import torch
# import numpy as np

from mmdet.models.necks.yolo_neck_shuffle_cat import ShuffleCatNeck

device = torch.device('cuda:0')
x1 = torch.rand(1, 1024, 20, 20).to(device)
x2 = torch.rand(1, 512, 40, 40).to(device)
x3 = torch.rand(1, 256, 80, 80).to(device)

in_channels = [1024, 512, 256]
out_channels = [512, 256, 128]

model = ShuffleCatNeck(3, in_channels, out_channels).to(device)
y = model([x3, x2, x1])
print(y[0].shape)