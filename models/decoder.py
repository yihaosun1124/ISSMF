import torch
import torch.nn as nn
import torch.nn.functional as F
from models.carafe import build_carafe
import numpy as np


class Decoder(nn.Module):

    def __init__(self, num_classes, backbone):

        super(Decoder, self).__init__()

        if backbone == 'resnet' or backbone == 'drn':

            low_level_inplanes = 256

        elif backbone == 'xception':

            low_level_inplanes = 128

        elif backbone == 'mobilenet':

            low_level_inplanes = 24

        else:

            raise NotImplementedError

        self.carafe1 = build_carafe(128, up_factor=2)

        self.carafe2 = build_carafe(128, up_factor=2)

        self.carafe3 = build_carafe(128, up_factor=2)

        self.conv1 = nn.Conv2d(256, 128, 1, bias=False)

        self.conv2 = nn.Conv2d(256, 128, 1, bias=False)

        self.gn1 = nn.GroupNorm(16, 128)

        self.gn2 = nn.GroupNorm(16, 128)

        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),

                                       nn.GroupNorm(16, 128),

                                       nn.ReLU(),

                                       nn.Dropout(0.5),

                                       nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),

                                       nn.GroupNorm(16, 128),

                                       nn.ReLU(),

                                       nn.Dropout(0.1),

                                       nn.Conv2d(128, num_classes, kernel_size=1, stride=1))

        self._init_weight()

    def forward(self, x, low_level_feat1, low_level_feat2):

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu(x)

        low_level_feat2_size = np.array(list(low_level_feat2.shape[2:]))
        x = nn.AdaptiveAvgPool2d(low_level_feat2_size // 2)(x)
        f3 = self.carafe1(x)
        f3 = F.interpolate(f3, size=tuple(low_level_feat2_size), mode='bilinear', align_corners=True)

        low_level_feat2 = self.conv2(low_level_feat2)
        low_level_feat2 = self.gn2(low_level_feat2)
        low_level_feat2 = self.relu(low_level_feat2)
        low_level_feat2 = low_level_feat2 + f3

        low_level_feat1_size = np.array(list(low_level_feat1.size()[2:]))
        low_level_feat2 = nn.AdaptiveAvgPool2d(low_level_feat1_size // 2)(low_level_feat2)
        f2 = self.carafe2(low_level_feat2)
        f2 = F.interpolate(f2, size=tuple(low_level_feat1_size), mode='bilinear', align_corners=True)

        f1 = low_level_feat1 + f2

        f3 = nn.AdaptiveAvgPool2d(low_level_feat1_size // 2)(f3)
        f3 = self.carafe2(f3)
        f3 = F.interpolate(f3, size=tuple(low_level_feat1_size), mode='bilinear', align_corners=True)

        output = f1 + f2 + f3

        output = self.last_conv(output)

        return output

    def _init_weight(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()

            elif isinstance(m, nn.GroupNorm):

                m.weight.data.fill_(1)

                m.bias.data.zero_()


def build_decoder(num_classes, backbone):

    return Decoder(num_classes, backbone)