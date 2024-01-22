#~/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train
#~/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/test
#~/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val

import os
from tqdm import tqdm
from PIL import Image
import random
from tinygrad import nn
import numpy as np
import functools
import json
import glob
from tinygrad.helpers import diskcache_put, diskcache_get
import hashlib, pickle
from tqdm import trange
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.nn.optim import Adam, SGD
from tinygrad.nn.state import get_parameters
from tinygrad import dtypes
from tinygrad.nn.state import safe_save, get_state_dict
from memory_profiler import profile
import torchvision.transforms.functional as F
from pathos.multiprocessing import ProcessingPool as Pool
import tracemalloc
import linecache


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


BASE = "/home/michael/Downloads/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/"


imagenet_classes = {v[0]: int(k) for k,v in json.load(open("/home/michael/Documents/FromTheTensor/datasets/imagenet-class-index.json")).items()}

train_files = glob.glob(str(BASE+"train/*/*"))
random.shuffle(train_files)
val_files = train_files[:10000]
train_files = train_files[10000:]

# def image_load(fn):
#   import torchvision.transforms.functional as F
#   ret = []
#   with Image.open(fn) as img:
#     img1 = img.convert("RGB")
#     img2 = F.resize(img1, 256, Image.BILINEAR)
#     img3 = F.center_crop(img2, 224)
#     ret = np.array(img3, dtype=np.float32)
#     img.close()
#     img1.close()
#   return ret

def image_load(fn):
  import torchvision.transforms.functional as F
  img = F.resize(Image.open(fn).convert('RGB'), 256, Image.BILINEAR)
  img = F.center_crop(img, 224)
  ret = np.array(img)
  return ret

def iterate(bs=1024, val=True, shuffle=True):
  cir = imagenet_classes
  files = val_files if val else train_files
  order = list(range(0, len(files)))
  if shuffle: random.shuffle(order)
  p = Pool(12)
  for i in range(0, len(files), bs):
    X = p.map(image_load, [files[i] for i in order[i:i+bs]])
    Y = [cir[files[i].split("/")[-2]] for i in order[i:i+bs]]
    yield (Tensor(np.array(X).swapaxes(1, -1), dtype=dtypes.float32), Tensor(np.array(Y)))

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

X, Y = Tensor.zeros(1, 1, 1, 1), Tensor.zeros(1, 1, 1, 1)

@TinyJit
def eval():
  return ((net(X).argmax(axis=1)==Y).mean()*100).realize()

@TinyJit
def step() -> Tensor:
   with Tensor.train():
        optimizer.zero_grad()
        loss = net(X).sparse_categorical_crossentropy(Y)
        loss.backward()
        optimizer.step()
        return loss.realize()

tracemalloc.start()

net = Model()

for i in trange(1000):
   X,Y = fetch_batch(1024)
   res= net(X)
   snapshot = tracemalloc.take_snapshot()
   display_top(snapshot)
params = get_parameters(net)
optimizer = Adam(params=params)
# test_acc = 0
# for i in (t:=trange(10000)):
#   X,Y = fetch_batch(1024, val=False)
#   loss = step()
#   t.set_description(f"Step:{i}, Loss:{loss.item()}, Test_Acc:{test_acc}")
#   if i%500==499:
#     X,Y = fetch_batch(1024, val=True)
#     test_acc = eval().item()
#     state_dict = get_state_dict(net)
#     safe_save(state_dict, f"AlexNet-{i}-{test_acc:5.2f}.safetensors")
#   snapshot = tracemalloc.take_snapshot()
#   display_top(snapshot)