from tinygrad import Tensor
from tinygrad.nn import Embedding, Linear
from tinygrad.dtype import dtypes
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad import TinyJit
import random
BS = 256
T = 256
head_size = 65


def make_mask(t:int)->Tensor:
    return Tensor.triu(Tensor.ones(t,t, requires_grad=False)*float("-inf"), k=1)


print(make_mask(50).numpy())



data = ""
with open("shakespeare.txt", "r") as f:
    data = f.read()

vocab = {k:v for v,k in enumerate(set(data))}
vocab_rev = {k:v for k,v in enumerate(set(data))}

val = data[:len(data)//10]
data = data[len(data)//10:]

class AttentionHead:
    def __init__(self):
        self.key = Tensor.scaled_uniform(len(vocab.keys()), head_size)
        self.query = Tensor.scaled_uniform(len(vocab.keys()), head_size)
        #self.value = Tensor.scaled_uniform(len(vocab.keys()), head_size)
    def __call__(self, x:Tensor) -> Tensor:
        k = x.linear(self.key)
        q = x.linear(self.query)
        wei = (q@k.transpose(-2, -1))*head_size**-0.5
        # wei = wei@(Tensor.triu(Tensor.ones(T,T, requires_grad=False)*float("-inf"), k=1)).softmax()
        return wei@x


class BigramModel:
    def __init__(self) -> None:
        self.embedding = Embedding(len(vocab.keys()), len(vocab.keys()))
        # self.positional_embedding = Embedding(T)
    def __call__(self, x:Tensor)->Tensor:
        tok_embedding = self.embedding(x)
        # pos_embedding = self.positional_embedding(x)
        return x.sequential([self.embedding])

class SingleHeadTransformer:
    def __init__(self) -> None:
        self.embedding = Embedding(len(vocab.keys()), len(vocab.keys()))
        self.positional_embedding = Embedding(1024, len(vocab.keys()))
        self.head = AttentionHead()
        self.linear = Linear(len(vocab.keys()), len(vocab.keys()))
    def __call__(self, x:Tensor)->Tensor:
        tok_embedding = self.embedding(x)
        pos_embedding = self.positional_embedding(Tensor.arange(T).repeat([x.shape[-1],1]).cast(dtype=dtypes.float))
        embedding = tok_embedding+pos_embedding
        x = self.head(embedding)
        x = self.linear(x)
        return x


def fetch_batch():
    x = []
    y = []
    for _ in range(BS):
        rand = random.randint(0, len(data)-(T+1))
        x.append([vocab[i] for i in data[rand:rand+T]])
        y.append([vocab[i] for i in data[rand+1:rand+T+1]])
    return (Tensor(x, dtype=dtypes.float), Tensor(y, dtype=dtypes.float))

model = SingleHeadTransformer()

# attention = AttentionHead(16)

# attention(model(fetch_batch()[0]))
optimizer = Adam(get_parameters(model))

@TinyJit
def step(x, y)->Tensor:
    return model(x).flatten(0,1).sparse_categorical_crossentropy(y.flatten(0, 1)).backward().realize()

print(fetch_batch()[0].shape, fetch_batch()[1].shape)

with Tensor.train():
    for i in range(1000):
        print(step(*fetch_batch()).item())
        optimizer.step()