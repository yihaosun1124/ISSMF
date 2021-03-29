import torch

from PIL import Image

from torch.utils.data import DataLoader

import numpy as np

from util.make_dataset import MyDataSet

import time

from models.issmf import ISSMF


def convert(img):

    for i in range(img.shape[0]):

        for j in range(img.shape[1]):

            if img[i][j] == 1:

                img[i][j] = 255

            else:

                img[i][j] = img[i][j]

    return img


def visualize(crop_size):

    model = torch.load('result/train/model.pkl')

    model.eval()

    val_data_path = 'datasets/index/test.txt'

    val_dataset = MyDataSet(data_path=val_data_path, is_train=False, crop_size=crop_size)

    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1)

    begin = time.time()

    for i, img in enumerate(val_loader):

        img = img.cuda()

        output = model(img)

        predict = torch.max(output, 1)[1].cpu().numpy()

        for p in predict:

            p = p.astype(np.uint8)

            p = convert(p)

            p = Image.fromarray(p)

            p.save('result/test/{0}_predict.png'.format(i + 1))

            print('visualizing {0}'.format(i + 1))

    end = time.time()

    print((end - begin) / len(val_loader))


if __name__ == '__main__':
    visualize(crop_size=512)
