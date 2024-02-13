from tinygrad import Tensor
from tinygrad.nn import Embedding
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad import TinyJit
import random
BS = 256
T = 256


def make_mask(t:int)->Tensor:
    tril = Tensor.triu(Tensor.ones(t,t)*float("-inf"), k=1)+Tensor.ones(t,t)
    return tril.softmax()

print(make_mask(50).numpy())



data = ""
with open("shakespeare.txt", "r") as f:
    data = f.read()

vocab = {k:v for v,k in enumerate(set(data))}
vocab_rev = {k:v for k,v in enumerate(set(data))}

val = data[:len(data)//10]
data = data[len(data)//10:]

# class AttentionHead:
#     def __init__(self, head_size):
#         self.key = Tensor.

class BigramModel:
    def __init__(self) -> None:
        self.embedding = Embedding(len(vocab.keys()), len(vocab.keys()))
        self.positional_embedding = Embedding(len())
    def __call__(self, x:Tensor)->Tensor:
        tok_embedding = self.embedding(x)
        pos_embedding = self.positional_embedding(x)
        return x.sequential([self.embedding])

def fetch_batch():
    x = []
    y = []
    for _ in range(BS):
        rand = random.randint(0, len(data)-(T+1))
        x.append([vocab[i] for i in data[rand:rand+T]])
        y.append([vocab[i] for i in data[rand+1:rand+T+1]])
    return (Tensor(x, dtype=dtypes.float), Tensor(y, dtype=dtypes.float))

model = BigramModel()
optimizer = Adam(get_parameters(model))

@TinyJit
def step(x, y)->Tensor:
    return model(x).flatten(0,1).sparse_categorical_crossentropy(y.flatten(0, 1)).backward().realize()


with Tensor.train():
    for i in range(1000):
        print(step(*fetch_batch()).item())
        optimizer.step()