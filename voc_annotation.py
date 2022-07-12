import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    dataset_path = "dataset"
    train_percent = 0.9

    segfilepath = os.path.join(dataset_path, 'SegmentationClass')
    temp_seg = os.listdir(segfilepath)
    num = len(temp_seg)
    list = range(num)

    tr = int(num * train_percent)
    train = random.sample(list, tr)

    ftrain = open(os.path.join(dataset_path, 'train.txt'), 'w')
    fval = open(os.path.join(dataset_path, 'val.txt'), 'w')

    for i in list:
        name = temp_seg[i][:-4] + '\n'
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    ftrain.close()
    fval.close()



if __name__ == "__main__":
    main()