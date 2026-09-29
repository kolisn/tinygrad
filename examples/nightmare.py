import os
import numpy as np
import random
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# 1. Dataset Loading (Flickr8k)
# -----------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
print("Dataset directory:", THIS_DIR)
IMAGES_DIR = os.path.join(THIS_DIR, "Flickr8k_Dataset")
CAPTION_FILE = os.path.join(THIS_DIR, "Flickr8k.token.txt")

captions_dict = {}  # image_filename -> list of captions
with open(CAPTION_FILE, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue
        img_id, cap = parts
        # img_id is like "1007129816_e794419615.jpg#0"
        fname, _ = img_id.split("#", 1)
        fname = fname.strip()
        if not fname.lower().endswith(".jpg"):
            continue  # skip bad lines
        captions_dict.setdefault(fname, []).append(cap.lower().strip())

all_caps = [c for caps in captions_dict.values() for c in caps]
words = set()
for c in all_caps:
    for w in c.split():
        words.add(w)
words = sorted(list(words))

vocab = ["<pad>", "<start>", "<end>"] + words
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
print("Vocab size:", vocab_size)

def caption_to_indices(cap):
    tokens = ["<start>"] + cap.split() + ["<end>"]
    return [word2idx.get(t, 0) for t in tokens]

# -----------------------------
# 2. Model Definition (tinygrad)
# -----------------------------

class SimpleCNN:
    def __init__(self):
        self.W1 = Tensor.randn(16, 3, 3, 3, dtype=dtypes.float32) * 0.1
        self.b1 = Tensor.zeros(16, dtype=dtypes.float32)
        self.W2 = Tensor.randn(32, 16, 3, 3, dtype=dtypes.float32) * 0.1
        self.b2 = Tensor.zeros(32, dtype=dtypes.float32)

    def __call__(self, x):
        y = x.conv2d(self.W1, self.b1).relu()
        y = y.conv2d(self.W2, self.b2).relu()
        N, C, H, W = y.shape
        return y.reshape(N, C * H * W)

class SimpleRNN:
    def __init__(self, fdim, hidden_dim, out_dim):
        self.Wx = Tensor.randn(fdim + vocab_size, hidden_dim, dtype=dtypes.float32) * 0.1
        self.Wh = Tensor.randn(hidden_dim, hidden_dim, dtype=dtypes.float32) * 0.1
        self.Wo = Tensor.randn(hidden_dim, out_dim, dtype=dtypes.float32) * 0.1
        self.bh = Tensor.zeros(hidden_dim, dtype=dtypes.float32)
        self.bo = Tensor.zeros(out_dim, dtype=dtypes.float32)
        self.hidden_dim = hidden_dim

    def step(self, x, h_prev):
        h = (x.dot(self.Wx) + h_prev.dot(self.Wh) + self.bh).tanh()
        out = h.dot(self.Wo) + self.bo
        return out, h

    def init_hidden(self, batch):
        return Tensor.zeros(batch, self.hidden_dim, dtype=dtypes.float32)

# -----------------------------
# 3. Training Loop
# -----------------------------

lr = 1e-3
num_epochs = 1
batch_size = 4

cnn = SimpleCNN()
dummy = cnn(Tensor(np.zeros((1, 3, 32, 32), dtype=np.float32)))
_, feat_dim = dummy.shape
print("Detected feature_dim:", feat_dim)

rnn = SimpleRNN(feat_dim, hidden_dim=128, out_dim=vocab_size)

params = [cnn.W1, cnn.W2, cnn.b1, cnn.b2,
          rnn.Wx, rnn.Wh, rnn.Wo, rnn.bh, rnn.bo]

def load_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32))
    arr = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    return Tensor(arr)

train_items = list(captions_dict.items())
random.shuffle(train_items)

for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(0, len(train_items), batch_size):
        print(f"Epoch {epoch}, processing batch {i//batch_size + 1}/{(len(train_items)+batch_size-1)//batch_size}")
        print("Percent complete:", (i / len(train_items)) * 100.0)
        batch = train_items[i:i+batch_size]
        img_tensors = []
        caption_idxs = []
        for img_filename, caps in batch:
            img = load_image(os.path.join(IMAGES_DIR, img_filename))
            img_tensors.append(img)
            cap = random.choice(caps)
            caption_idxs.append(caption_to_indices(cap))
        x = img_tensors[0].stack(*img_tensors[1:], dim=0).float()
        features = cnn(x).float()
        h = rnn.init_hidden(len(batch))
        losses = []
        max_len = max(len(ci) for ci in caption_idxs)

        for t in range(max_len - 1):
            batch_input = np.zeros((len(batch), vocab_size), dtype=np.float32)
            targets = np.zeros((len(batch),), dtype=np.int32)
            for bi in range(len(batch)):
                widx = caption_idxs[bi][t] if t < len(caption_idxs[bi]) else word2idx["<pad>"]
                batch_input[bi, widx] = 1.0
                targets[bi] = caption_idxs[bi][t + 1] if t + 1 < len(caption_idxs[bi]) else word2idx["<pad>"]

            batch_input_tensor = Tensor(batch_input, dtype=dtypes.float32)
            combined = Tensor(
                np.concatenate([batch_input, features.numpy().astype(np.float32)], axis=1),
                dtype=dtypes.float32
            )
            out, h = rnn.step(combined, h)
            loss = out.sparse_categorical_crossentropy(Tensor(targets, dtype=dtypes.int32))
            losses.append(loss)

        avg_loss = Tensor.stack(*losses, dim=0).mean()
        total_loss += float(avg_loss.numpy().item())
        avg_loss.backward()
        for p in params:
            if p.grad is not None:
                p -= lr * p.grad
                p.grad = None

    print(f"Epoch {epoch}: loss {total_loss / (len(train_items)//batch_size)}")

# -----------------------------
# 4. Inference + Visualization
# -----------------------------

def generate_caption(image_tensor, max_len=10):
    features = cnn(image_tensor.reshape(1, *image_tensor.shape)).float()
    h = rnn.init_hidden(1)
    token = "<start>"
    caption = []
    for _ in range(max_len):
        onehot = np.zeros((1, vocab_size), dtype=np.float32)
        onehot[0, word2idx[token]] = 1.0
        x_input = Tensor(np.concatenate([onehot, features.numpy().astype(np.float32)], axis=1),
                         dtype=dtypes.float32)
        out, h = rnn.step(x_input, h)
        idx = int(np.argmax(out.softmax().numpy().squeeze()))
        word = idx2word[idx]
        caption.append(word)
        if word == "<end>":
            break
        token = word
    return " ".join(caption)

def show_image_with_caption(img_path):
    img_t, pil = load_image(img_path), Image.open(img_path).convert("RGB")
    caption = generate_caption(img_t)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = None
    draw.text((10,10), caption, fill=(255,0,0), font=font)
    pil.show()
    print("Caption:", caption)

# Test on a random image
test_filename = random.choice(list(captions_dict.keys()))
test_path = os.path.join(IMAGES_DIR, test_filename)
show_image_with_caption(test_path)
