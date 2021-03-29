from ..backbone import xception


def build_backbone(backbone, output_stride):

    return xception.AlignedXception(output_stride)
