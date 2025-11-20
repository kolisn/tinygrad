# fourier_animation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------------
# Fourier feature generator
# ----------------------------
def build_fourier_coords(H, W, B=4, scale=5.0, seed=0):
    np.random.seed(seed)
    ys = np.linspace(-1, 1, H)
    xs = np.linspace(-1, 1, W)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)  # (H, W, 2)
    coords = grid.reshape(-1, 2)
    Bmat = np.random.normal(scale=scale, size=(2, B)).astype(np.float32)
    proj = (coords @ Bmat) * (2.0 * np.pi)
    feats = np.concatenate([np.sin(proj), np.cos(proj)], axis=-1)
    out = np.concatenate([coords.astype(np.float32), feats], axis=-1)
    return out  # shape: (H*W, 2 + 2*B)

# ----------------------------
# Setup
# ----------------------------
H, W = 8, 8  # small grid for clarity
B_values = [1, 2, 3, 4, 6, 8, 12, 16]  # increasing B
scale = 0.1
seed = 42

fig, axes = plt.subplots(2, 4, figsize=(10,5))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# Initialize heatmaps
ims = []
for ax_row in axes:
    for ax in ax_row:
        im = ax.imshow(np.zeros((H,W)), cmap='coolwarm', vmin=-1, vmax=1)
        ax.axis('off')
        ims.append(im)

# ----------------------------
# Update function for animation
# ----------------------------
def update(frame):
    B = B_values[frame]
    features = build_fourier_coords(H, W, B=B, scale=scale, seed=seed)
    
    # Extract sine and cosine features
    sine_feats = features[:, 2:2+B] if B>1 else features[:, 2:3]
    cos_feats  = features[:, 2+B:2+2*B] if B>1 else features[:, 3:4]
    
    # Fill heatmaps
    for i in range(4):
        if i < B:
            ims[i].set_data(sine_feats[:, i].reshape(H, W))
            ims[i+4].set_data(cos_feats[:, i].reshape(H, W))
        else:
            ims[i].set_data(np.zeros((H,W)))
            ims[i+4].set_data(np.zeros((H,W)))
    
    fig.suptitle(f"Fourier Features with B={B}", fontsize=16)

# ----------------------------
# Run animation
# ----------------------------
ani = FuncAnimation(fig, update, frames=len(B_values), interval=1000, repeat=True)
plt.show()
