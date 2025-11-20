"""
CONVOLUTIONAL AUTOENCODER FOR IMAGE RECONSTRUCTION

This autoencoder learns to:
1. ENCODE: Compress 32x32 RGB images to a latent representation (128 channels, 4x4 spatial)
2. DECODE: Reconstruct the original image from the latent representation

Architecture:
  Encoder:  32x32x3 -> 16x16x32 -> 8x8x64 -> 4x4x128 (compressed)
  Decoder:  4x4x128 -> 8x8x64 -> 16x16x32 -> 32x32x3 (reconstructed)
"""

import matplotlib.pyplot as plt
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor
from tinygrad.nn.datasets import cifar
import numpy as np

# ============================================================
# HYPERPARAMETERS
# ============================================================
LR = 1e-3        # Learning rate: how fast the model learns
EPOCHS = 1       # Number of full passes through the dataset
BATCH_SIZE = 64  # Images per batch for gradient computation

# ============================================================
# CONVOLUTIONAL LAYER (Encoder)
# ============================================================
class ConvLayer:
    """
    Applies 2D convolution to compress spatial dimensions and extract features.
    
    Forward pass: x -> conv2d(x, W) + b
    Input shape:  (batch, height, width, in_channels)
    Output shape: (batch, height', width', out_channels)
    
    With stride=2: height' = height//2, width' = width//2
    With padding=1: preserves boundary information
    """
    
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        """
        Initialize convolutional layer.
        
        Args:
          in_ch: number of input channels (e.g., 3 for RGB)
          out_ch: number of output channels (filters)
          kernel: kernel size (e.g., 3x3)
          stride: step size for sliding window (1=no downsampling, 2=half size)
          padding: zeros added around borders (1=preserves edges)
        """
        # Weight tensor: shape (out_ch, in_ch, kernel, kernel)
        # Each of out_ch filters processes in_ch input channels
        self.W = Tensor.randn(out_ch, in_ch, kernel, kernel, requires_grad=True) * 0.1
        
        # Bias: one per output channel (shape: out_ch)
        self.b = Tensor.zeros(out_ch, requires_grad=True)
        
        # Store stride and padding for forward pass
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        """
        Forward pass: apply convolution to input tensor x.
        
        Args:
          x: input tensor of shape (batch, height, width, in_channels)
        
        Returns:
          output of shape (batch, height', width', out_channels)
        """
        # Conv2d applies: y = sum_over_channels(W * x) + b
        return x.conv2d(self.W, self.b, stride=self.stride, padding=self.padding)

    def parameters(self):
        """Return all trainable parameters (weights and biases)."""
        return [self.W, self.b]


# ============================================================
# DECONVOLUTIONAL LAYER (Decoder)
# ============================================================
class DeconvLayer:
    """
    Applies 2D transposed convolution (deconvolution) to upsample spatial dimensions.
    
    Forward pass: x -> conv_transpose2d(x, W) + b
    Input shape:  (batch, height, width, in_channels)
    Output shape: (batch, height'*stride, width'*stride, out_channels)
    
    With stride=2: doubles spatial dimensions (4x4 -> 8x8)
    output_padding: adds extra padding to match exact output size
    """
    
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, output_padding=0):
        """
        Initialize deconvolutional layer.
        
        Args:
          in_ch: number of input channels
          out_ch: number of output channels
          kernel: kernel size (e.g., 3x3)
          stride: upsampling factor (2=double spatial size)
          padding: input padding
          output_padding: extra output padding for exact size matching
        """
        # Weight tensor for transpose conv: shape (in_ch, out_ch, kernel, kernel)
        self.W = Tensor.randn(in_ch, out_ch, kernel, kernel, requires_grad=True) * 0.1
        
        # Bias: one per output channel
        self.b = Tensor.zeros(out_ch, requires_grad=True)
        
        # Store parameters for forward pass
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def __call__(self, x):
        """
        Forward pass: apply transposed convolution to upsample x.
        
        Args:
          x: input tensor of shape (batch, height, width, in_channels)
        
        Returns:
          upsampled output of shape (batch, height'*stride, width'*stride, out_channels)
        """
        # Transposed conv: inverse of regular conv, used for upsampling
        return x.conv_transpose2d(
            self.W, self.b,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )

    def parameters(self):
        """Return all trainable parameters."""
        return [self.W, self.b]


# ============================================================
# CONVOLUTIONAL AUTOENCODER
# ============================================================
class ConvAutoEncoder:
    """
    Complete autoencoder architecture:
    
    ENCODER (compresses):
      Input: 32x32x3 (RGB image)
      -> Conv(3->32, stride=2) + ReLU -> 16x16x32
      -> Conv(32->64, stride=2) + ReLU -> 8x8x64
      -> Conv(64->128, stride=2) + ReLU -> 4x4x128 (LATENT)
    
    DECODER (reconstructs):
      Input: 4x4x128 (latent representation)
      -> Deconv(128->64, stride=2) + ReLU -> 8x8x64
      -> Deconv(64->32, stride=2) + ReLU -> 16x16x32
      -> Deconv(32->3, stride=2) + Sigmoid -> 32x32x3 (reconstructed image)
    """
    
    def __init__(self):
        """Initialize encoder and decoder layers."""
        
        # ========== ENCODER ==========
        # Progressively compress 32x32 -> 16x16 -> 8x8 -> 4x4
        # while increasing channels: 3 -> 32 -> 64 -> 128
        self.e1 = ConvLayer(3, 32, stride=2, padding=1)    # 32x32x3 -> 16x16x32
        self.e2 = ConvLayer(32, 64, stride=2, padding=1)   # 16x16x32 -> 8x8x64
        self.e3 = ConvLayer(64, 128, stride=2, padding=1)  # 8x8x64 -> 4x4x128 (latent)

        # ========== DECODER ==========
        # Progressively upsample 4x4 -> 8x8 -> 16x16 -> 32x32
        # while decreasing channels: 128 -> 64 -> 32 -> 3
        self.d1 = DeconvLayer(128, 64, stride=2, padding=1, output_padding=1)   # 4x4x128 -> 8x8x64
        self.d2 = DeconvLayer(64, 32, stride=2, padding=1, output_padding=1)    # 8x8x64 -> 16x16x32
        self.d3 = DeconvLayer(32, 3, stride=2, padding=1, output_padding=1)     # 16x16x32 -> 32x32x3

    def encode(self, x):
        """
        Compress image to latent representation.
        
        Args:
          x: image tensor of shape (batch, 3, 32, 32)
        
        Returns:
          latent tensor of shape (batch, 128, 4, 4)
        """
        # Pass through encoder with ReLU activations between layers
        x = self.e1(x).relu()  # Add non-linearity to learn complex patterns
        x = self.e2(x).relu()
        x = self.e3(x).relu()  # Final latent representation
        return x

    def decode(self, z):
        """
        Reconstruct image from latent representation.
        
        Args:
          z: latent tensor of shape (batch, 128, 4, 4)
        
        Returns:
          reconstructed image of shape (batch, 3, 32, 32) with values in [0,1]
        """
        # Pass through decoder with ReLU for hidden layers
        z = self.d1(z).relu()  # Add non-linearity
        z = self.d2(z).relu()
        z = self.d3(z).sigmoid()  # Output in [0,1] range (pixel values)
        return z

    def __call__(self, x):
        """Full forward pass: encode then decode."""
        return self.decode(self.encode(x))

    def parameters(self):
        """Return all trainable parameters from encoder and decoder."""
        return (
            self.e1.parameters() +
            self.e2.parameters() +
            self.e3.parameters() +
            self.d1.parameters() +
            self.d2.parameters() +
            self.d3.parameters()
        )


# ============================================================
# LOAD DATASET
# ============================================================
print("Loading CIFAR-10 dataset...")
# Download and load CIFAR-10: 50,000 train images, 10,000 test images
X_train, Y_train, X_test, Y_test = cifar()
print(f"✓ CIFAR-10 loaded: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Use only 1/2 of the dataset for faster training
X_train = X_train[:len(X_train)//2]
X_test = X_test[:len(X_test)//2]
print(f"✓ Using 1/2 of dataset: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Normalize images from [0, 255] to [0, 1]
# This helps the neural network learn better
print("Normalizing data (divide by 255.0)...")
X_train = X_train.float() / 255.0
X_test = X_test.float() / 255.0
print(f"✓ Data normalized")

# ============================================================
# INITIALIZE MODEL AND OPTIMIZER
# ============================================================
print("\nInstantiating model...")
# Create the autoencoder with random weights
model = ConvAutoEncoder()
params = get_parameters(model)  # Extract all trainable parameters
total_params = sum(p.numel() for p in params)  # Count total number of parameters
print(f"✓ Model created with {total_params:,} parameters")

# Create optimizer: Adam adapts learning rate per parameter
print(f"Creating optimizer (Adam, lr={LR})...")
opt = Adam(params, lr=LR)
print(f"✓ Optimizer initialized")


# ============================================================
# TRAINING STEP (No JIT - variable batch handling)
# ============================================================
@Tensor.train()  # Enable training mode
def train_step(batch):
    """
    Single training iteration.
    
    Args:
      batch: tensor of shape (batch_size, 32, 32, 3)
    
    Returns:
      scalar loss value
    """
    opt.zero_grad()
    recon = model(batch)
    loss = (recon - batch).pow(2).mean()
    loss.backward()
    opt.step()
    return loss


# ============================================================
# TRAINING LOOP
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Starting training: {EPOCHS} epochs, batch_size={BATCH_SIZE}")
    total_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Total batches per epoch: {total_batches}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        losses = []

        # Iterate through dataset in batches
        for batch_idx in range(total_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(X_train))
            batch = X_train[start_idx:end_idx]
            batch_size = batch.shape[0]

            # Only process full batches
            if batch_size != BATCH_SIZE:
                break

            loss = train_step(batch)
            loss_val = loss.item()
            losses.append(loss_val)
            
            print(f"  Batch {batch_idx + 1}: processing {batch_size} images... loss: {loss_val:.6f}")

        # Print epoch summary statistics
        if losses:
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            print(f"\n  Epoch {epoch+1} Summary:")
            print(f"    Batches processed: {len(losses)}")
            print(f"    Avg loss: {avg_loss:.6f}")
            print(f"    Min loss: {min_loss:.6f}")
            print(f"    Max loss: {max_loss:.6f}")
        print(f"{'='*60}\n")

    # Training complete, now visualize results
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    Tensor.training = False
    
    # ========== 1. ORIGINAL VS RECONSTRUCTED ==========
    print("\n1. Showing original vs reconstructed images...")
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Original vs Reconstructed', fontsize=14)
    
    for idx in range(4):
        rand_idx = int(np.random.randint(0, len(X_test)))
        original = X_test[rand_idx:rand_idx+1]
        reconstructed = model(original)
        
        orig_np = original.numpy()[0].transpose(1, 2, 0)
        recon_np = reconstructed.numpy()[0].transpose(1, 2, 0)
        
        # Original
        axes[idx, 0].imshow(orig_np)
        axes[idx, 0].set_title(f'Orig {idx+1}')
        axes[idx, 0].axis('off')
        
        # Reconstructed
        axes[idx, 1].imshow(recon_np)
        axes[idx, 1].set_title(f'Recon {idx+1}')
        axes[idx, 1].axis('off')
        
        # Difference
        diff = np.abs(orig_np - recon_np)
        axes[idx, 2].imshow(diff)
        axes[idx, 2].set_title(f'Diff {idx+1}')
        axes[idx, 2].axis('off')
        
        # MSE
        mse = np.mean((orig_np - recon_np) ** 2)
        axes[idx, 3].text(0.5, 0.5, f'{mse:.4f}', ha='center', va='center', fontsize=12)
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('autoenc_reconstruction.png', dpi=100)
    print("✓ Saved: autoenc_reconstruction.png")
    plt.close()
    
    # ========== 2. INTERPOLATION ==========
    print("\n2. Showing interpolation between two images...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Latent Space Interpolation (t: 0 -> 1)', fontsize=14)
    
    idx1 = int(np.random.randint(0, len(X_test)))
    idx2 = int(np.random.randint(0, len(X_test)))
    img1 = X_test[idx1:idx1+1]
    img2 = X_test[idx2:idx2+1]
    
    z1 = model.encode(img1)
    z2 = model.encode(img2)
    
    for step_idx, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        z_interp = (1 - t) * z1 + t * z2
        img_interp = model.decode(z_interp)
        img_np = img_interp.numpy()[0].transpose(1, 2, 0)
        axes[0, step_idx].imshow(img_np)
        axes[0, step_idx].set_title(f't={t:.2f}')
        axes[0, step_idx].axis('off')
    
    axes[1, 0].imshow(img1.numpy()[0].transpose(1, 2, 0))
    axes[1, 0].set_title('Image 1')
    axes[1, 0].axis('off')
    
    axes[1, 4].imshow(img2.numpy()[0].transpose(1, 2, 0))
    axes[1, 4].set_title('Image 2')
    axes[1, 4].axis('off')
    
    for col in range(1, 4):
        axes[1, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('autoenc_interpolation.png', dpi=100)
    print("✓ Saved: autoenc_interpolation.png")
    plt.close()
    
    # ========== 3. RANDOM SAMPLES ==========
    print("\n3. Showing random samples from latent space...")
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Random Samples from Latent Space', fontsize=14)
    
    for row in range(3):
        for col in range(5):
            # Sample random latent vector - shape: (batch, channels, height, width)
            z_random = Tensor.randn(1, 128, 4, 4)
            fake_image = model.decode(z_random)
            img_np = fake_image.numpy()[0].transpose(1, 2, 0)
            axes[row, col].imshow(img_np)
            axes[row, col].set_title(f'#{row*5+col+1}')
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('autoenc_random_samples.png', dpi=100)
    print("✓ Saved: autoenc_random_samples.png")
    plt.close()
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60 + "\n")
