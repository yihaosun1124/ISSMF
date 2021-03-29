from torch.utils.data import DataLoader

import numpy as np

import torch

from models.issmf import ISSMF

from util.lr_scheduler import LR_Scheduler

from util.make_dataset import MyDataSet

from util.calculate_weights import calculate_weigths_labels

from util.options import get_options

from tqdm import tqdm

import os

if __name__ == '__main__':

    opt = get_options()

    train_data_path = 'datasets/index/train.txt'

    train_dataset = MyDataSet(data_path=train_data_path, is_train=True, crop_size=opt.crop_size)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size)

    model = ISSMF(num_classes=opt.n_classes)

    train_params = [{'params': model.get_1x_lr_params(), 'lr': opt.lr},
                    {'params': model.get_10x_lr_params(), 'lr': opt.lr * 10}]

    classes_weight_path = 'util/classes_weights.npy'

    if os.path.isfile(classes_weight_path):

        weight = np.load('util/classes_weights.npy')

    else:

        weight = calculate_weigths_labels(train_loader, opt.n_classes)

    weight = torch.from_numpy(weight.astype(np.float32))

    criterion = torch.nn.CrossEntropyLoss(weight=weight, size_average=True, ignore_index=255)

    optimizer = torch.optim.Adam(train_params, weight_decay=4e-5)

    scheduler = LR_Scheduler(mode='poly', base_lr=opt.lr, num_epochs=opt.epoch, iters_per_epoch=len(train_loader))

    # Using cuda

    if len(opt.gpu_ids) > 0:
        
        model = model.cuda()

        criterion = criterion.cuda()

    for e in range(opt.epoch):

        train_loss = 0.0

        model.train()

        tbar = tqdm(train_loader)

        num_img_tr = len(train_loader)

        for i, sample in enumerate(tbar):

            image, target = sample['image'], sample['label']

            if len(opt.gpu_ids) > 0:
                image, target = image.cuda(), target.cuda()

            scheduler(optimizer, i, e)

            optimizer.zero_grad()

            output = model(image)

            loss = criterion(output, target.long())

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        torch.save(model, 'result/train/model.pkl')