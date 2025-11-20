#!/usr/bin/env python3
"""Load a checkpoint produced by `examples/hlb_cifar10.py` and run evaluation on CIFAR-10 test set.

Usage:
  PYTHONPATH=. CHECKPOINT_PATH=examples/hlb_cifar10_ckpt.npz python3 examples/hlb_cifar10_infer.py
"""
import os, sys
import random
import numpy as np

# ensure local package imports work
sys.path.insert(0, os.getcwd())

from tinygrad import nn, dtypes, Tensor, Device
from tinygrad.nn.state import get_state_dict, safe_load, load_state_dict

# import the model class from the training module
from examples.hlb_cifar10 import SpeedyResNet, cifar_mean, cifar_std

# recreate the same preprocessing transform used in training
def make_transform():
  return [
    lambda x: x.float() / 255.0,
    lambda x: x.reshape((-1,3,32,32)) - Tensor(cifar_mean, device=x.device, dtype=x.dtype).reshape((1,3,1,1)),
    lambda x: x / Tensor(cifar_std, device=x.device, dtype=x.dtype).reshape((1,3,1,1)),
  ]

def load_checkpoint(path):
  ckpt = np.load(path)
  return ckpt

def main():
  ckpt_path = os.environ.get('CHECKPOINT_PATH', 'examples/hlb_cifar10_ckpt.npz')
  if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

  # load safetensors-like checkpoint and apply to model
  ckpt = safe_load(ckpt_path)
  if 'whitening' not in ckpt:
    raise KeyError('whitening not found in checkpoint')
  # ensure whitening tensor is moved off disk to the default device
  W = ckpt['whitening'].cast(dtypes.default_float).to(Device.DEFAULT)

  model = SpeedyResNet(W)
  # Prefer EMA weights if present in the checkpoint
  ema_keys = [k for k in ckpt.keys() if k.startswith('ema.')]
  if len(ema_keys) > 0:
    print(f"Found {len(ema_keys)} EMA parameters in checkpoint, loading EMA weights for inference.")
    ema_state = {k[len('ema.'):]: ckpt[k] for k in ema_keys}
    load_state_dict(model, ema_state, strict=False, verbose=False)
  else:
    # load into model (allow missing/extra keys)
    load_state_dict(model, ckpt, strict=False, verbose=False)

  # prepare test data (same as training script preprocessing)
  _, _, X_test, Y_test = nn.datasets.cifar()
  Y_test = Y_test.one_hot(10)
  transform = make_transform()
  X_test = X_test.sequential(transform)
  X_test = X_test.cast(dtypes.default_float)

  BS = int(os.environ.get('EVAL_BS', 64))
  corrects = []
  total = 0
  # CIFAR-10 class names
  class_names = [
    'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
  ]
  SHOW_SAMPLES = int(os.environ.get('SHOW_SAMPLES', 10))
  printed = 0
  # pick random sample indices from the evaluation set (only consider full batches)
  total_samples = (X_test.shape[0] // BS) * BS
  if total_samples <= 0:
    sample_indices = set()
  else:
    num_to_pick = min(SHOW_SAMPLES, total_samples)
    sample_indices = set(random.sample(range(total_samples), num_to_pick))
  for i in range(0, (X_test.shape[0] // BS) * BS, BS):
    Xt = X_test[i:i+BS]
    Yt = Y_test[i:i+BS]
    out = model(Xt, training=False)
    preds = out.argmax(axis=1).numpy().tolist()
    trues = Yt.argmax(axis=1).numpy().tolist()
    correct = [int(p==t) for p,t in zip(preds, trues)]
    corrects.extend(correct)
    total += len(correct)
    # print random selected predicted vs actual labels
    if sample_indices:
      # check which indices in this batch are selected
      for j, (p, t) in enumerate(zip(preds, trues)):
        global_idx = i + j
        if global_idx in sample_indices:
          ok = 'OK' if p==t else 'ERR'
          print(f"idx {global_idx}: pred={p} ({class_names[p]}), actual={t} ({class_names[t]}) -> {ok}")
          printed += 1
          # optional early exit when printed enough
          if printed >= len(sample_indices):
            # continue evaluating remaining batches but stop printing
            sample_indices = set()
            break

  acc = sum(corrects) / total * 100.0 if total>0 else 0.0
  print(f"Inference accuracy: {sum(corrects)}/{total} {acc:.2f}%")

if __name__ == '__main__':
  main()
