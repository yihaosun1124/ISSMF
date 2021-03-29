import argparse


def get_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--in_channels', type=int, default=3, help='input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--n_classes', type=int, default=2, help='classes num')
    parser.add_argument('--is_train', type=bool, default=True, help='is train')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--crop_size', type=int, default=512, help='crop size')

    opt = parser.parse_args()
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)

    return opt
