import json
import math
import os
import random

import numpy as np
import torch

TARGETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'color_aug_targets.json')
B_RANGE, S_RANGE = (0.6, 1.8), (0.6, 1.8)
C_RANGE, H_RANGE, G_RANGE = (0.75, 1.25), (-30.0, 30.0), (0.8, 1.25)
_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ_INV = torch.linalg.inv(_YIQ)
_targets = None


def fg_stats(rgb: torch.Tensor, mask: torch.Tensor):
    m = mask[0] > 0.5
    if int(m.sum()) < 50:
        return None
    mx, mn = rgb.max(0).values, rgb.min(0).values
    s = torch.where(mx > 1e-6, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))
    return float(mx[m].mean()), float(s[m].mean())


def _load_targets():
    global _targets
    if _targets is None:
        with open(TARGETS_PATH) as fh:
            _targets = json.load(fh)['targets']
        assert len(_targets) >= 500, f'{TARGETS_PATH} has only {len(_targets)} targets'
    return _targets


def color_aug(rgb: torch.Tensor, mask: torch.Tensor, rng: random.Random = random):
    squeeze = rgb.dim() == 4
    x0, m = (rgb[0], mask[0]) if squeeze else (rgb, mask)
    st = fg_stats(x0, m)
    if st is None:
        return rgb, None
    v0, s0 = st
    vt, stt = rng.choice(_load_targets())
    p = {'b': min(max(vt / max(v0, 1e-3), B_RANGE[0]), B_RANGE[1]),
         's': min(max(stt / s0, S_RANGE[0]), S_RANGE[1]) if s0 > 0.02 else 1.0,
         'c': rng.uniform(*C_RANGE), 'h': rng.uniform(*H_RANGE), 'g': rng.uniform(*G_RANGE)}
    w = m.clamp(0, 1)
    x = x0 * p['b']
    mu = (x * w).sum((1, 2), keepdim=True) / w.sum().clamp_min(1.0)
    x = (x - mu) * p['c'] + mu
    gray = (x * torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)).sum(0, keepdim=True)
    x = gray + (x - gray) * p['s']
    t = math.radians(p['h'])
    Rq = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]])
    M = _YIQ_INV @ Rq @ _YIQ
    x = torch.einsum('ij,jhw->ihw', M, x).clamp(0, 1) ** p['g']
    out = x0 * (1 - w) + x * w
    return (out[None] if squeeze else out), p


def rot90_z(coords: np.ndarray, k: int, grid: int = 64) -> np.ndarray:
    c = np.asarray(coords).copy()
    for _ in range(int(k) % 4):
        c = np.stack([c[:, 1], grid - 1 - c[:, 0], c[:, 2]], 1)
    return c


