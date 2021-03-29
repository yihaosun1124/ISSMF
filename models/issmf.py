import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from models.aspp import build_aspp

from models.decoder import build_decoder

from models.backbone import build_backbone

from models.carafe import build_carafe


class ISSMF(nn.Module):

    def __init__(self, backbone='xception', output_stride=16, num_classes=2):

        super(ISSMF, self).__init__()

        if backbone == 'drn':

            output_stride = 8


        self.backbone = build_backbone(backbone, output_stride)

        self.aspp = build_aspp(backbone, output_stride)

        self.decoder = build_decoder(num_classes, backbone)

        self.carafe = build_carafe(num_classes, up_factor=4)


    def forward(self, input):

        x, low_level_feat1, low_level_feat2 = self.backbone(input)

        x = self.aspp(x)

        x = self.decoder(x, low_level_feat1, low_level_feat2)

        input_size = np.array(list(input.size()[2:]))

        x = nn.AdaptiveAvgPool2d(input_size // 4)(x)

        x = self.carafe(x)

        x = F.interpolate(x, size=tuple(input_size), mode='bilinear', align_corners=True)

        return x

    def get_1x_lr_params(self):

        modules = [self.backbone]

        for i in range(len(modules)):

            for m in modules[i].named_modules():

                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.GroupNorm):

                    for p in m[1].parameters():

                        if p.requires_grad:

                            yield p

    def get_10x_lr_params(self):

        modules = [self.aspp, self.decoder]

        for i in range(len(modules)):

            for m in modules[i].named_modules():

                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.GroupNorm):

                    for p in m[1].parameters():

                        if p.requires_grad:

                            yield p


if __name__ == "__main__":

    model = ISSMF(backbone='xception', output_stride=16, num_classes=2)

    model.eval()

    input = torch.rand(1, 3, 512, 512)

    output = model(input)




