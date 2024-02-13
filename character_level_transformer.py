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

data = open("shakespeare.txt", "r").read()


vocab = {k:v for v,k in enumerate(set(data))}
vocab_rev = {k:v for k,v in enumerate(set(data))}

def fetch_batch(bs=128):
    x = []
    y = []
    for i in range(bs):
        start = random.randint(0, len(data)-257)
        x.append([vocab[token] for token in data[start:start+256]])
        y.append([vocab[data[start+i+1]] for i in range(256)])
    return (Tensor(x), Tensor(y))


model = transformer.Transformer(len(vocab), 256, 4, 256,8,1024)
optim = Adam(get_parameters(model), lr=0.001)

@TinyJit
def step(X:Tensor, Y:Tensor):
    print(model.forward(X).shape)
    loss = model.forward(X).reshape(128*256, -1).sparse_categorical_crossentropy(Y.reshape(128*256)).backward()
    return loss


def validate(X, Y):
    return ((model.forward(X).argmax(-1)==Y).mean()*100).realize().item()

val = 0

x, y = fetch_batch()
print(x.shape)

print(model.forward(x).shape)

with Tensor.train(val=False):
    for i in (t:=trange(10000)):
        X, Y = fetch_batch()
        loss=step(X,Y)
        optim.step()
        t.set_description_str(f"loss:{loss.item()} lr:{optim.lr.item()} validation:{val}")
        if i%200==0:
            with Tensor.train(val=True):
                val = 0
                for i in range(10):
                    val += validate(*fetch_batch())
                val/=10.0