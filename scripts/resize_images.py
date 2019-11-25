import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import cv2
import sys
import fnmatch
import os
import progressbar
import numpy as np

res = (500, 500)


def preprocess_dataset(path):
    os.mkdir("resized")
    ch = fnmatch.filter(os.listdir(path), '*.png')
    print("len(ch) : ", len(ch))

    for i in progressbar.progressbar(range(len(ch))):
        img = cv2.imread(os.path.join(path, ch[i]))
        cv2.imwrite(os.path.join("resized", str(i) + ".png"), cv2.resize(img, dsize=res))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        preprocess_dataset(sys.argv[1])
    else:
        print("Usage : resize_images.py data_path")
