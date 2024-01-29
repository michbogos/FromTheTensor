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
from tinygrad.nn.state import get_state_dict, safe_save

BS = 128

dicts = []

dtypes.default_float = dtypes.float16

for i in range(5):
    with open(f"./datasets/cifar-10-batches-py/data_batch_{i+1}", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        dicts.append(dict)
        print(dict.keys())

x_train = Tensor(np.reshape(np.concatenate([dicts[i][b"data"] for i in range(5)], dtype=np.float32), (50000, 3, 32, 32))).cast(dtypes.default_float)
print(x_train[0][0].dtype)
y_train = Tensor(np.concatenate([dicts[i][b"labels"] for i in range(5)]))
x_eval = None
y_eval = None

with open(f"./datasets/cifar-10-batches-py/test_batch", "rb") as fo:
    dict = pickle.load(fo, encoding='bytes')
    x_eval = Tensor(np.reshape(np.asarray(dict[b"data"], dtype=np.float32), (10000, 3,32,32))).cast(dtypes.default_float)
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

class WellTunedCNN:
    def __init__(self):
        self.layers = [
        nn.Conv2d(3, 96, (5,5), padding=2),
        Tensor.max_pool2d,
        nn.BatchNorm2d(96),
        nn.Conv2d(96, 96, (5,5), padding=2),
        Tensor.max_pool2d,
        nn.BatchNorm2d(96),
        nn.Conv2d(96, 80, (5,5), padding=2),
        Tensor.relu,
        nn.Conv2d(80, 64, (5,5), padding=2),
        Tensor.relu,
        nn.Conv2d(64, 64, (5,5), padding=2),
        Tensor.relu,
        nn.Conv2d(64, 96, (5,5), padding=2),
        Tensor.relu,
        lambda x: x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])),
        nn.Linear(6144, 256),
        Tensor.relu,
        nn.Linear(256, 10),]
    def __call__(self, x:Tensor):
        return x.sequential(self.layers)


class Block:
    def __init__(self, cin, cout, stride):
        self.bn1 = nn.BatchNorm2d(cin)
        #relu1
        self.conv1 = nn.Conv2d(cin, cout, (3,3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        #relu2
        self.conv2 = nn.Conv2d(cout, cout, (3, 3), stride=1, padding=1, bias=False)

        self.projection = (stride != 1) or (cin != cout)
        if self.projection:
            self.conv3 = nn.Conv2d(cin,cout, (1, 1), stride=stride, bias=False)
    
    def __call__(self, x) -> Tensor:
        x = self.bn1(x)
        x = x.relu()
        x1 = x
        if self.projection:
            x1 = self.conv3(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv2(x)
        x = x+x1
        return x

class FastCifar:
    def __init__(self, channels):
        self.channels = [channels, channels*2, channels*4, channels*4]
        self.layers = [
            nn.Conv2d(3, self.channels[0], (3, 3), 1, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            Tensor.relu,

            Block(self.channels[0], self.channels[0], 1),
            Block(self.channels[0], self.channels[0], 1),

            Block(self.channels[0], self.channels[1], 2),
            Block(self.channels[1], self.channels[1], 1),

            Block(self.channels[1], self.channels[2], 2),
            Block(self.channels[2], self.channels[2], 1),

            Block(self.channels[2], self.channels[3], 2),
            Block(self.channels[3], self.channels[3], 1),

            lambda x: x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])),
            nn.Linear(4096, 10)
        ]
    
    def __call__(self, x):
        return x.sequential(self.layers)

net = FastCifar(64)
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
for i in (t:=trange(7000)):
    samples = Tensor.randint(128, high=50000)
    loss = step(samples)
    if i%1000 == 999:
        test_acc = eval().item()
        state_dict = get_state_dict(net)
        safe_save(state_dict, f"cifar-{i}-{test_acc:5.2f}.safetensors")
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")