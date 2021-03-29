from torch.utils.data import Dataset

from torchvision import transforms

from PIL import Image

import numpy as np

from util import my_transform as tr


class MyDataSet(Dataset):

    def __init__(self, data_path, is_train, crop_size):

        super().__init__()

        self.data_path = [x.rstrip() for x in open(data_path)]

        self.is_train = is_train

        self.base_size = crop_size

        self.crop_size = crop_size

    def __len__(self):

        return len(self.data_path)

    def __getitem__(self, item):

        if self.is_train:

            img_path = 'datasets/image/' + self.data_path[item]

            label_path = 'datasets/mask/' + self.data_path[item]

            img = Image.open(img_path).convert('RGB')

            label = Image.open(label_path).convert('L')

            sample = {'image': img, 'label': label}

            sample = self.transform_tr(sample)

            return sample

        else:

            img_path = 'datasets/image/' + self.data_path[item]

            img = Image.open(img_path)

            mean = (0.5, 0.5, 0.5)

            std = (0.5, 0.5, 0.5)

            img = np.array(img).astype(np.float32)

            img /= 255.0

            img -= mean

            img /= std

            img = transforms.ToTensor()(img)

            img = img.float()

            return img

    def transform_tr(self, sample):

        composed_transforms = transforms.Compose([

            tr.RandomHorizontalFlip(),

            # tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size, fill=255),

            tr.FixedResize(512),

            tr.RandomGaussianBlur(),

            tr.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

            tr.ToTensor()])

        return composed_transforms(sample)
