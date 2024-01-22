from PIL import Image
import numpy as np
import glob
import random
from tinygrad import Tensor
from tinygrad import nn
import json
from tinygrad import dtypes
from tqdm import trange
from tinygrad import TinyJit

BASE = "/home/michael/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/"


imagenet_classes = {v[0]: int(k) for k,v in json.load(open("/home/michael/Documents/FromTheTensor/datasets/imagenet-class-index.json")).items()}

train_files = glob.glob(str(BASE+"train/*/*"))
random.shuffle(train_files)
val_files = train_files[:10000]
train_files = train_files[10000:]

def image_load(fn):
  import torchvision.transforms.functional as F
  img = F.resize(Image.open(fn).convert('RGB'), 256, Image.BILINEAR)
  img = F.center_crop(img, 224)
  ret = np.array(img)
  return ret


def fetch_batch(bs, val=False):
  files = val_files if val else train_files
  samp = np.random.randint(0, len(files), size=(bs))
  files = [files[i] for i in samp]
  x_train = [image_load(x) for x in files]
  y_train = [imagenet_classes[x.split("/")[-2]] for x in files]
  return Tensor(np.swapaxes(x_train, 1, -1), dtype=dtypes.float32), Tensor(y_train, dtype=dtypes.float32)

class Model:
    def __init__(self):
        self.layers = [
           nn.Conv2d(3, 96, (11, 11), 4),
           Tensor.relu,
           lambda x: x.max_pool2d(kernel_size=(3, 3), stride=2),
           nn.BatchNorm2d(96),
           nn.Conv2d(96, 256, (5, 5), 2, padding=2),
           Tensor.relu,
           lambda x: x.max_pool2d(kernel_size=(3, 3), stride=2),
           nn.BatchNorm2d(256),
           nn.Conv2d(256, 384, (3, 3), padding=1),
           Tensor.relu,
           nn.Conv2d(384, 384, (3, 3), padding=1),
           Tensor.relu,
           nn.Conv2d(384, 256, (3, 3), padding=1),
           Tensor.relu,
           lambda x: x.max_pool2d(kernel_size=(3, 3), stride=2),
           nn.BatchNorm2d(256),
           lambda x: x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]),
           nn.Linear(1024, 1024),
           Tensor.dropout,
           Tensor.relu,
           nn.Linear(1024, 1000),
        ]
    @TinyJit    
    def __call__(self, x:Tensor):
       return x.sequential(self.layers)

net = Model()

for i in trange(1000):
   X,Y = fetch_batch(1024)
   res= net(X)