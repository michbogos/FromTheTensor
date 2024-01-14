from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.nn.state import get_state_dict, safe_save
from extra.datasets import fetch_mnist
from tqdm import trange

class Model:
  def __init__(self):
    self.layers: List[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 48, (7, 7)),
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

      lambda x:x.reshape(x.shape[0], 3072),
      nn.Linear(3072, 10),
      nn.LayerNorm(10)]

  def __call__(self, x:Tensor):return x.sequential(self.layers)

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = fetch_mnist(tensors=True)

  model = Model()
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=0.001, b1=0.98, b2=0.999)

  @TinyJit
  def train_step() -> Tensor:
    with Tensor.train():
      opt.zero_grad()
      samples = Tensor.randint(128, high=X_train.shape[0])
      loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
      opt.step()
      return loss.realize()
  @TinyJit
  def get_test_acc(idx) -> Tensor:
    return ((model(X_test[idx]).argmax(axis=1) == Y_test[idx]).mean()*100).realize()

  test_acc = float('nan')
  for i in (t:=trange(500)):
    GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
    loss = train_step()
    if i%50 == 49: test_acc = get_test_acc(Tensor.randint(10000, high=10000)).item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
  
  state_dict = get_state_dict(model)
  safe_save(state_dict, "mnist.safetensors")