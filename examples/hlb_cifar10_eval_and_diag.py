#!/usr/bin/env python3
"""Evaluate a checkpoint and run diagnostics for hlb-CIFAR10.

Saves an EMA-only checkpoint if EMA weights are present, prints a confusion matrix,
per-class logits stats, feature activation stats, and final-linear weight stats.

Usage:
  PYTHONPATH=. CHECKPOINT_PATH=examples/hlb_cifar10_ckpt.safetensors python3 examples/hlb_cifar10_eval_and_diag.py
"""
import os, sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.getcwd())

from tinygrad import nn, dtypes, Tensor, Device
from tinygrad.nn.state import safe_load, safe_save, load_state_dict, get_state_dict

from examples.hlb_cifar10 import SpeedyResNet, cifar_mean, cifar_std


def make_transform():
  return [
    lambda x: x.float() / 255.0,
    lambda x: x.reshape((-1,3,32,32)) - Tensor(cifar_mean, device=x.device, dtype=x.dtype).reshape((1,3,1,1)),
    lambda x: x / Tensor(cifar_std, device=x.device, dtype=x.dtype).reshape((1,3,1,1)),
  ]


def find_final_linear(state):
  # find a weight with first dim == 10 (classifier)
  for k, v in state.items():
    if hasattr(v, 'shape') and len(v.shape) == 2 and int(v.shape[0]) == 10:
      return k, v
  return None, None


def main():
  ckpt_path = os.environ.get('CHECKPOINT_PATH', 'examples/hlb_cifar10_ckpt.safetensors')
  if not os.path.exists(ckpt_path):
    raise FileNotFoundError(ckpt_path)

  print(f"Loading checkpoint: {ckpt_path}")
  ckpt = safe_load(ckpt_path)

  if 'whitening' not in ckpt:
    raise KeyError('whitening missing in checkpoint')
  W = ckpt['whitening'].cast(dtypes.default_float).to(Device.DEFAULT)

  # build model and prefer EMA if present
  ema_keys = [k for k in ckpt.keys() if k.startswith('ema.')]
  model = SpeedyResNet(W)
  if ema_keys:
    print(f"Found {len(ema_keys)} EMA params; loading EMA into model and saving EMA-only checkpoint.")
    ema_state = {k[len('ema.'):]: ckpt[k] for k in ema_keys}
    load_state_dict(model, ema_state, strict=False, verbose=False)
    # save EMA-only checkpoint for convenience
    ema_path = ckpt_path + '.ema.safetensors'
    try:
      safe_save(ema_state, ema_path, metadata={'src': ckpt_path})
      print(f"Saved EMA-only checkpoint: {ema_path}")
    except Exception as e:
      print("Warning: failed to save EMA checkpoint:", e)
  else:
    print("No EMA params found; loading standard checkpoint into model.")
    load_state_dict(model, ckpt, strict=False, verbose=False)

  # prepare test data
  _, _, X_test, Y_test = nn.datasets.cifar()
  Y_test = Y_test.one_hot(10)
  transform = make_transform()
  X_test = X_test.sequential(transform)
  X_test = X_test.cast(dtypes.default_float)

  BS = int(os.environ.get('EVAL_BS', 64))

  total = 0
  correct = 0
  num_classes = 10
  conf = np.zeros((num_classes, num_classes), dtype=int)

  # accumulators for diagnostics
  logits_sum = np.zeros((num_classes,), dtype=float)
  logits_sq = np.zeros((num_classes,), dtype=float)
  logits_count = 0

  feat_stats = defaultdict(list)

  # find final linear weight key
  state = get_state_dict(model)
  final_k, final_w = find_final_linear(state)
  if final_k is None:
    print("Warning: final linear not found in state dict")
  else:
    print(f"Final linear weight key: {final_k}, shape={final_w.shape}")

  # helper to get features before final linear
  def get_features(x):
    # replicate SpeedyResNet forward up to before the classifier
    forward = lambda x: x.conv2d(W).pad((1,0,0,1)).sequential(model.net[:-2])
    return forward(x)

  for i in range(0, (X_test.shape[0] // BS) * BS, BS):
    Xt = X_test[i:i+BS]
    Yt = Y_test[i:i+BS]
    out = model(Xt, training=False)
    preds = out.argmax(axis=1).numpy()
    trues = Yt.argmax(axis=1).numpy()
    for p, t in zip(preds, trues):
      conf[int(t), int(p)] += 1
    total += len(preds)
    correct += int((preds == trues).sum())

    # logits stats
    logits = out.numpy()
    logits_sum += logits.sum(axis=0)
    logits_sq += (logits**2).sum(axis=0)
    logits_count += logits.shape[0]

    # feature stats on first batch only (representative)
    if i == 0:
      feats = get_features(Xt).numpy()
      feat_stats['mean'] = float(feats.mean())
      feat_stats['std'] = float(feats.std())
      feat_stats['min'] = float(feats.min())
      feat_stats['max'] = float(feats.max())

  acc = correct/total*100.0 if total>0 else 0.0
  print(f"Evaluation: {correct}/{total} {acc:.2f}%")

  # confusion matrix
  print("Confusion matrix (rows=true, cols=pred):")
  print(conf)

  # per-class logits mean/std
  if logits_count>0:
    logits_mean = logits_sum / logits_count
    logits_std = (logits_sq / logits_count - logits_mean**2)**0.5
    print("Per-class logits mean:", logits_mean.tolist())
    print("Per-class logits std :", logits_std.tolist())

  # feature stats
  if feat_stats:
    print("Feature stats (first batch):", feat_stats)

  # final linear weight stats
  if final_w is not None:
    w = final_w.numpy()
    norms = np.linalg.norm(w, axis=1)
    print(f"Final linear weight norms (per-class): {norms.tolist()}")
    print(f"Final linear weight mean: {w.mean():.6f}, std: {w.std():.6f}")

if __name__ == '__main__':
  main()
