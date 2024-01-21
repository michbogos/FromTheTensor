#~/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train
#~/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test
#~/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val

import os
from tqdm import tqdm
from PIL import Image
import random

classes = os.listdir("/home/michael/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train")

files = []

BASE = "/home/michael/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"

with open("/home/michael/Downloads/imagenet-object-localization-challenge/ILSVRC/ImageSets/CLS-LOC/train_cls.txt") as f:
    for line in tqdm(f, total=1281167):
        files.append(line.split(" ")[0])

images = [Image.open(BASE+random.choice(files)+".JPEG") for i in range(1000)]