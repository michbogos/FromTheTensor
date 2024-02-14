from tinygrad import nn, Tensor
from tinygrad.nn.optim import Adam, SGD, AdamW
from tinygrad.nn.state import get_parameters
from tinygrad import dtypes
from extra.models import transformer
import gensim.downloader as api
import random
from tqdm import trange
from tinygrad import TinyJit
from lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from extra.training import train, evaluate
from tinygrad.nn.state import get_state_dict, safe_save

data = open("shakespeare.txt", "r").read()


vocab = {k:v for v,k in enumerate(set(data))}
vocab_rev = {k:v for k,v in enumerate(set(data))}

def make_dataset():
  dataset = []
  for i in trange(len(data)-1000):
    dataset.append([vocab[data[i+j]] for j in range(257)])
  random.shuffle(dataset)
  ds = np.array(dataset).astype(np.float32)

  ds_X = ds[:, 0:256]

  ds_Y = np.copy(ds[:, 1:])
  ds_X_train, ds_X_test = ds_X[0:1000000], ds_X[1000000:]
  ds_Y_train, ds_Y_test = ds_Y[0:1000000], ds_Y[1000000:]
  return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test


model = transformer.Transformer(len(vocab), 256, 4, 256,16,1024)
optim = Adam(get_parameters(model), lr=0.001)

@TinyJit
def step(X:Tensor, Y:Tensor):
    print(model.forward(X).shape)
    loss = model.forward(X).reshape(128*256, -1).sparse_categorical_crossentropy(Y.reshape(128*256)).backward()
    return loss


def validate(X, Y):
    return ((model.forward(X).argmax(-1)==Y).mean()*100).realize().item()

xt,yt,xv,yv = make_dataset()

for i in range(50):
    train(model, xt, yt, optim=optim, steps=1000)
    optim.lr /=1.2
    print(evaluate(model, xv,yv,65))

state = get_state_dict(model)
safe_save(state, "char_transformer.safetensors")

# with Tensor.train(val=False):
#     for i in (t:=trange(10000)):
#         X, Y = fetch_batch()
#         loss=step(X,Y)
#         optim.step()
#         t.set_description_str(f"loss:{loss.item()} lr:{optim.lr.item()} validation:{val}")
#         if i%200==0:
#             with Tensor.train(val=True):
#                 val = 0
#                 for i in range(10):
#                     val += validate(*fetch_batch())
#                 val/=10.0