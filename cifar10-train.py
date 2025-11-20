import numpy as np
from tinygrad import Tensor, TinyJit
from tinygrad.nn import Conv2d, Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.datasets import cifar
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_save, get_parameters, get_state_dict
from tinygrad.helpers import trange, getenv

# Loading CIFAR-10 dataset (train and test splits)
X_train, Y_train, X_test, Y_test = cifar()

# Ensure that the data is in the correct format
# `nn.datasets.cifar()` returns Tensors (not numpy arrays), so use Tensor APIs
X_train = X_train.float() / 255.0
X_test  = X_test.float() / 255.0

# Ensure labels are integer dtype for loss/indexing
Y_train = Y_train.cast(dtypes.int64)
Y_test  = Y_test.cast(dtypes.int64)

# Print dataset info
print(f"Loaded CIFAR-10 dataset:")
try:
  print(f"  X_train shape: {X_train.shape}, dtype: {getattr(X_train, 'dtype', 'unknown')}")
  print(f"  Y_train shape: {Y_train.shape}, dtype: {getattr(Y_train, 'dtype', 'unknown')}")
  print(f"  X_test shape:  {X_test.shape}, dtype: {getattr(X_test, 'dtype', 'unknown')}")
  print(f"  Y_test shape:  {Y_test.shape}, dtype: {getattr(Y_test, 'dtype', 'unknown')}")
except Exception:
  print(f"  (shapes/dtypes not available for these Tensor objects)")

# Simple CNN Model (Convolution + Fully connected)
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2d(3, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.fc = Linear(64 * 8 * 8, 10)  # 10 CIFAR-10 classes

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu().max_pool2d(2)
        x = self.conv2(x).relu().max_pool2d(2)
        x = x.reshape(x.shape[0], -1)  # Flatten for fully connected layer
        x = self.fc(x)
        return x

# Instantiate the model and optimizer
model = SimpleCNN()
params = get_parameters(model)
optimizer = Adam(params, lr=0.001)

# Print model and optimizer info
try:
  total_params = int(sum(np.prod(p.shape) for p in params))
except Exception:
  total_params = None
lr = getattr(optimizer, 'lr', getattr(optimizer, 'learning_rate', 'unknown'))
print(f"Model instantiated: SimpleCNN, total parameters: {total_params}, optimizer: Adam, lr={lr}")

# Training step using JIT for performance
@TinyJit
@Tensor.train()
def train_step() -> Tensor:
  optimizer.zero_grad()
  samples = Tensor.randint(64, high=X_train.shape[0])
  imgs = X_train[samples]
  labels = Y_train[samples]
  logits = model(imgs)
  loss = logits.sparse_categorical_crossentropy(labels)
  loss.backward()
  return loss.realize(*optimizer.schedule_step())

# Evaluation step (sample for speed)
@TinyJit
def evaluate(sample_size=500) -> Tensor:
  samples = Tensor.randint(sample_size, high=X_test.shape[0])
  predictions = model(X_test[samples]).argmax(axis=1)
  labels = Y_test[samples]
  accuracy = (predictions == labels).mean() * 100
  return accuracy.realize()

# Training loop
EPOCHS = getenv('EPOCHS',10)
print(f"Starting training for {EPOCHS} epochs")
for epoch in range(EPOCHS):  # 2 epochs
  print(f"Starting epoch {epoch+1}")
  total_loss = 0
  num_batches = 10  # train on 10 batches per epoch
  for batch in range(num_batches):
    loss = train_step()
    total_loss += loss.item()
  avg_loss = total_loss / num_batches
  print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
  if (epoch + 1) % 1 == 0:
    acc = evaluate(sample_size=500)
    print(f"  Test Accuracy (500 samples): {acc.item():.4f}%")

safe_save(get_state_dict(model), "cifar_net.safetensors")
print("Model saved to cifar_net.safetensors")