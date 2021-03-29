This is an implementation of ISSMF (Integrated Semantic and Spatial Information of Multi-level Features) developed with Pytorch 1.4.0.

The architecture of ISSMF is shown as below:
![Architecture of ISSMF](https://www.hualigs.cn/image/60618a9fe1d66.jpg)

The network can be trained to perform image segmentation on arbitrary imaging data.

To train on your own datasets, please organize your datasets as shown below:

    datasets

    ├── image
    ├    ├── test
    ├    ├── train
    ├    └── val
    ├── index
    ├    ├── test.txt
    ├    ├── train.txt
    ├    └── val.txt
    └── mask
    ├    ├── test
    ├    ├── train
    ├    └── val

'datasets/image' includes the origin images,

'datasets/mask' includes the labels.

'datasets/index' includes the paths of the datasets such as:

train.txt:

    train/001.png
    train/002.png
    train/003.png
    ...

test.txt:

    test/001.png
    test/002.png
    test/003.png
    ...

val.txt:

    val/001.png
    val/002.png
    val/003.png
    ...