from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.nn.datasets import cifar
import numpy as np

# -----------------------------
# Hyperparameters
# -----------------------------
LR = 1e-3
EPOCHS = 1
BATCH_SIZE = 64

# -----------------------------
# Conv + Deconv Layers
# -----------------------------
class ConvLayer:
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        self.W = Tensor.randn(out_ch, in_ch, kernel, kernel, requires_grad=True) * 0.1
        self.b = Tensor.zeros(out_ch, requires_grad=True)
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        return x.conv2d(self.W, self.b, stride=self.stride, padding=self.padding)

    def parameters(self):
        return [self.W, self.b]


class DeconvLayer:
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, output_padding=0):
        self.W = Tensor.randn(in_ch, out_ch, kernel, kernel, requires_grad=True) * 0.1
        self.b = Tensor.zeros(out_ch, requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def __call__(self, x):
        return x.conv_transpose2d(self.W, self.b, stride=self.stride, padding=self.padding, output_padding=self.output_padding)

    def parameters(self):
        return [self.W, self.b]


# -----------------------------
# Autoencoder
# -----------------------------
class ConvAutoEncoder:
    def __init__(self):
        # Encoder: downsample 32 -> 16 -> 8
        self.e1 = ConvLayer(3, 32, stride=2, padding=1)
        self.e2 = ConvLayer(32, 64, stride=2, padding=1)
        self.e3 = ConvLayer(64, 128, stride=2, padding=1)

        # Decoder: upsample 8 -> 16 -> 32
        self.d1 = DeconvLayer(128, 64, stride=2, padding=1, output_padding=1)
        self.d2 = DeconvLayer(64, 32, stride=2, padding=1, output_padding=1)
        self.d3 = DeconvLayer(32, 3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        x = self.e1(x).relu()
        x = self.e2(x).relu()
        x = self.e3(x).relu()
        return x

    def decode(self, z):
        z = self.d1(z).relu()
        z = self.d2(z).relu()
        z = self.d3(z).sigmoid()
        return z

    def __call__(self, x):
        return self.decode(self.encode(x))

    def parameters(self):
        return (
            self.e1.parameters() +
            self.e2.parameters() +
            self.e3.parameters() +
            self.d1.parameters() +
            self.d2.parameters() +
            self.d3.parameters()
        )


# -----------------------------
# Training Loop
# -----------------------------
# Load CIFAR-10 dataset (once, outside training loop)
print("Loading CIFAR-10 dataset...")
X_train, Y_train, X_test, Y_test = cifar()
print(f"✓ CIFAR-10 loaded: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Ensure that the data is in the correct format
# `nn.datasets.cifar()` returns Tensors (not numpy arrays), so use Tensor APIs
print("Normalizing data (divide by 255.0)...")
X_train = X_train.float() / 255.0
X_test  = X_test.float() / 255.0
print(f"✓ Data normalized")

print("\nInstantiating model...")
model = ConvAutoEncoder()
params = get_parameters(model)
total_params = sum(p.numel() for p in params)
print(f"✓ Model created with {total_params:,} parameters")

print(f"Creating optimizer (Adam, lr={LR})...")
opt = Adam(params, lr=LR)
print(f"✓ Optimizer initialized")

@Tensor.train()
def train_step(batch):
  opt.zero_grad()
  recon = model(batch)
  loss = (recon - batch).pow(2).mean()
  loss.backward()
  opt.step()
  return loss

if __name__ == "__main__":
  print(f"\n{'='*60}")
  print(f"Starting training: {EPOCHS} epochs, batch_size={BATCH_SIZE}")
  print(f"Total batches per epoch: {(len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE}")
  print(f"{'='*60}\n")
  
  for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    losses = []
    num_batches = 0
    
    for i in range(0, len(X_train), BATCH_SIZE):
      batch_idx = num_batches + 1
      batch = X_train[i:i+BATCH_SIZE]
      batch_size = batch.shape[0]
      
      if num_batches < 5 or num_batches % 10 == 0:  # Print first 5 and every 10th
        print(f"  Batch {batch_idx}: processing {batch_size} images...", end=" ")
        loss = train_step(batch)
        loss_val = loss.item()
        losses.append(loss_val)
        print(f"loss: {loss_val:.6f}")
      else:
        loss = train_step(batch)
        losses.append(loss.item())
      
      num_batches += 1

    avg_loss = sum(losses) / len(losses) if losses else 0
    min_loss = min(losses) if losses else 0
    max_loss = max(losses) if losses else 0
    print(f"\n  Epoch {epoch+1} Summary:")
    print(f"    Batches processed: {num_batches}")
    print(f"    Avg loss: {avg_loss:.6f}")
    print(f"    Min loss: {min_loss:.6f}")
    print(f"    Max loss: {max_loss:.6f}")
    print(f"{'='*60}\n")
