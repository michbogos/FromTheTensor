from typing import Any
import numpy as np
import pickle
from tinygrad.nn import Conv2d
from tinygrad import nn
from tinygrad import Tensor
from tinygrad.nn.optim import SGD, Adam
from tinygrad.nn.state import get_parameters
from tinygrad import TinyJit
from tinygrad import dtypes
from tqdm import trange
import matplotlib.pyplot as plt
from efficientnet import EfficientNet
from typing import List, Callable

BS = 128

dicts = []

for i in range(5):
    with open(f"./datasets/cifar-10-batches-py/data_batch_{i+1}", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dicts.append(dict)
        print(dict.keys())

x_train = Tensor(np.reshape(np.concatenate([dicts[i][b"data"] for i in range(5)], dtype=np.float32), (50000, 3, 32, 32))).cast(dtypes.float)
print(x_train[0][0].dtype)
y_train = Tensor(np.concatenate([dicts[i][b"labels"] for i in range(5)]))
x_eval = None
y_eval = None

with open(f"./datasets/cifar-10-batches-py/test_batch", "rb") as fo:
    dict = pickle.load(fo, encoding='bytes')
    x_eval = Tensor(np.reshape(np.asarray(dict[b"data"], dtype=np.float32), (10000, 3,32,32))).cast(dtypes.float)
    y_eval = Tensor(np.asarray(dict[b"labels"]))

class NaiveConvnet():
    def __init__(self):
        self.layers=[
            Conv2d(3, 24, (7, 7)),
            nn.BatchNorm2d(24),
            Tensor.relu,
            Conv2d(24, 96, (7,7)),
            nn.BatchNorm2d(96),
            Tensor.relu,
            Conv2d(96, 144, (7, 7)),
            nn.BatchNorm2d(144),
            Tensor.relu,
            Conv2d(144, 256, (7, 7)),
            nn.BatchNorm2d(256),
            Tensor.relu,
            Conv2d(256, 512, (7, 7)),
            nn.BatchNorm2d(512),
            Tensor.relu,
            lambda x: x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])),
            nn.Linear(2048, 10),
            Tensor.relu
        ]
    def __call__(self, x) -> Tensor:
        return x.sequential(self.layers)

class NaiveResNet():
    def __init__(self):
        self.layers = []
        self.layers.append(Conv2d(3, 64, (7, 7), stride=2, padding=2))
        self.layers.append(Tensor.tanh)
        self.layers.append(Tensor.max_pool2d)
        for i in range(1):
            self.layers.append(Conv2d(64, 64, (3, 3), padding=1))
            self.layers.append(Tensor.tanh)
        self.layers.append(Conv2d(64, 128, (3, 3), padding=1))
        self.layers.append(Tensor.tanh)

        self.layers.append(Conv2d(128, 128, (3, 3), stride=2, padding=1))
        self.layers.append(Tensor.tanh)
        for i in range(1):
            self.layers.append(Conv2d(128, 128, (3, 3), padding=1))
            self.layers.append(Tensor.tanh)
        self.layers.append(Conv2d(128, 256, (3, 3), padding=1))
        self.layers.append(Tensor.tanh)

        self.layers.append(Conv2d(256, 256, (3, 3), stride=1, padding=1))
        self.layers.append(Tensor.tanh)
        for i in range(1):
            self.layers.append(Conv2d(256, 256, (3, 3), padding=1))
            self.layers.append(Tensor.tanh)
        self.layers.append(Conv2d(256, 512, (3, 3), padding=1))
        self.layers.append(Tensor.tanh)

        self.layers.append(Conv2d(512, 512, (3, 3), stride=1, padding=1))
        self.layers.append(Tensor.tanh)
        for i in range(1):
            self.layers.append(Conv2d(512, 512, (3,3), padding=1))
            self.layers.append(Tensor.tanh)
        self.layers.append(Tensor.avg_pool2d)
        self.layers.append(lambda x: x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])))
        self.layers.append(nn.Linear(512*2*2, 10))
    
    def __call__(self, x) -> Tensor:
        return x.sequential(self.layers)

class LeNet():
    def __init__(self):
        self.layers = [
            nn.Conv2d(3, 6, (5, 5)),
            Tensor.tanh,
            Tensor.max_pool2d,
            nn.Conv2d(6, 16, (5, 5)),
            Tensor.tanh,
            Tensor.max_pool2d,
            lambda x: x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])),
            nn.Linear(16*5*5, 120),
            Tensor.tanh,
            nn.Linear(120, 84),
            Tensor.tanh,
            nn.Linear(84, 10),
            Tensor.softmax
        ]
    def __call__(self, x:Tensor):
        res = x.sequential(self.layers)
        return res

class DumbNet():
    def __init__(self):
        self.layers = [
            nn.Conv2d(3, 32, (3, 3)),
            Tensor.relu,
            nn.BatchNorm2d(32),
            Tensor.max_pool2d,
            nn.Conv2d(32, 64, (5, 5)),
            Tensor.relu,
            nn.BatchNorm2d(64),
            lambda x: x.max_pool2d(kernel_size=(3, 3)),
            nn.Conv2d(64, 64, (3,3)),
            Tensor.relu,
            nn.BatchNorm2d(64),
            lambda x: x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])),
            nn.Linear(64, 64),
            Tensor.relu,
            nn.Linear(64, 10),
        ]
    def __call__(self, x:Tensor) -> Tensor:
        return x.sequential(self.layers)

class DumbestNet():
    def __init__(self):
        self.layers = [
            lambda x:x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])),
            nn.Linear(1024*3, 10),
            Tensor.sigmoid
        ]
    def __call__(self, x:Tensor) -> Tensor:
        return x.sequential(self.layers)

class Model7:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(3, 48, (7, 7)),
      nn.BatchNorm2d(48),
      Tensor.relu,

      nn.Conv2d(48, 92, (7, 7)),
      nn.BatchNorm2d(92),
      Tensor.relu,

      nn.Conv2d(92, 144, (7, 7)),
      nn.BatchNorm2d(144),
      Tensor.relu,

      nn.Conv2d(144, 192, (7, 7)),
      nn.BatchNorm2d(192),
      Tensor.relu,

      lambda x:x.reshape(x.shape[0], 192*8*8),
      nn.Linear(192*8*8, 10),
      nn.LayerNorm(10)]

  def __call__(self, x:Tensor):return x.sequential(self.layers)

class BatchNorm(nn.BatchNorm2d):
  def __init__(self, num_features):
    super().__init__(num_features, track_running_stats=False, eps=1e-12, momentum=0.85, affine=True)
    self.weight.requires_grad = False
    self.bias.requires_grad = True

class ConvGroup:
  def __init__(self, channels_in, channels_out):
    self.conv1 = nn.Conv2d(channels_in,  channels_out, kernel_size=3, padding=1, bias=False)
    self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)

    self.norm1 = BatchNorm(channels_out)
    self.norm2 = BatchNorm(channels_out)

  def __call__(self, x):
    x = self.conv1(x)
    x = x.max_pool2d(2)
    x = x.float()
    x = self.norm1(x)
    x = x.cast(dtypes.default_float)
    x = x.quick_gelu()
    residual = x
    x = self.conv2(x)
    x = x.float()
    x = self.norm2(x)
    x = x.cast(dtypes.default_float)
    x = x.quick_gelu()

    return x + residual

class SpeedyResNet:
  def __init__(self):
    self.net = [
      nn.Conv2d(3, 12, (3,3)),
      nn.Conv2d(12, 32, kernel_size=1, bias=False),
      lambda x: x.quick_gelu(),
      ConvGroup(32, 64),
      ConvGroup(64, 256),
      ConvGroup(256, 512),
      lambda x: x.max((2,3)),
      nn.Linear(512, 10, bias=False),
      lambda x: x.mul(1./9)
    ]

  def __call__(self, x, training=True):
    # pad to 32x32 because whitening conv creates 31x31 images that are awfully slow to compute with
    # TODO: remove the pad but instead let the kernel optimize itself
    return x.sequential(self.net)

net = SpeedyResNet()
params = get_parameters(net)
optimizer = Adam(params=params)

# for i in range(100):
#     plt.imshow(x_train[i].numpy().swapaxes(2, 0))
#     print(y_train[i].item())
#     plt.show()

@TinyJit
def step(samples):
    with Tensor.train():
        optimizer.zero_grad()
        loss = net(x_train[samples]).sparse_categorical_crossentropy(y_train[samples])
        loss.backward()
        optimizer.step()
        return loss.realize()


def eval():
    return ((net(x_eval[:100]).argmax(axis=1)==y_eval[:100]).mean()*100).realize()

test_acc = 0
for i in (t:=trange(10000)):
    samples = Tensor.randint(128, high=50000)
    loss = step(samples)
    if i%1000 == 100:
        test_acc = eval().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")