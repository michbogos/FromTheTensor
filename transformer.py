from tinygrad import nn, Tensor
from extra.models import transformer
import gensim

data = open("datasets/shakespeare.txt", "r").read()

model1 = gensim.models.Word2Vec(data, min_count=1,
                                vector_size=100, window=5)


transformer.Transformer()

print()