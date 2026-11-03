from typing import Tuple

import numpy as np


ROOT_LEVEL = 2
NUM_CLASSES = 256
MASK_INDEX = NUM_CLASSES


def native_level_of(num_levels: int) -> int:
    assert num_levels in (2, 3), f'num_levels must be 2 (64^3) or 3 (256^3), got {num_levels}'
    return ROOT_LEVEL + 2 * num_levels


def _read_vxz_coords(path: str) -> np.ndarray:
    import o_voxel
    v = o_voxel.io.read_vxz(path)
    if isinstance(v, (tuple, list)):
        c = v[0]
    elif isinstance(v, dict):
        c = v['coords'] if 'coords' in v else list(v.values())[0]
    else:
        c = v
    c = c.cpu().numpy() if hasattr(c, 'cpu') else np.asarray(c)
    return c[:, -3:].astype(np.int64)


def _down(coords: np.ndarray, shift: int) -> np.ndarray:
    return np.unique(coords >> shift, axis=0) if shift else np.unique(coords, axis=0)


def _key(coords: np.ndarray, level: int) -> np.ndarray:
    return (coords[:, 0] << (2 * level)) | (coords[:, 1] << level) | coords[:, 2]


def _unkey(keys: np.ndarray, level: int) -> np.ndarray:
    m = (1 << level) - 1
    return np.stack([(keys >> (2 * level)) & m, (keys >> level) & m, keys & m], axis=1)


def child_pattern(parent_level: int, child_coords: np.ndarray,
                  all_cells: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    pk = _key(child_coords >> 1, parent_level)
    bit = ((child_coords[:, 0] & 1) << 2) | ((child_coords[:, 1] & 1) << 1) | (child_coords[:, 2] & 1)

    if all_cells:
        n_cells = 8 ** parent_level
        mask = np.zeros(n_cells, dtype=np.int64)
        np.bitwise_or.at(mask, pk, (1 << bit).astype(np.int64))
        keys = np.arange(n_cells, dtype=np.int64)
        assert mask.min() >= 0 and mask.max() <= 255
        return keys, mask

    uniq, inv = np.unique(pk, return_inverse=True)
    mask = np.zeros(len(uniq), dtype=np.int64)
    np.bitwise_or.at(mask, inv, (1 << bit).astype(np.int64))
    assert mask.min() >= 1 and mask.max() <= 255, f'illegal child pattern range [{mask.min()},{mask.max()}]'
    return uniq, mask


def expand_children(parent_keys: np.ndarray, labels: np.ndarray, parent_level: int) -> np.ndarray:
    xyz = _unkey(parent_keys, parent_level)
    patt = labels
    out = []
    for bit in range(8):
        sel = (patt >> bit) & 1
        idx = np.nonzero(sel)[0]
        if len(idx) == 0:
            continue
        ch = (xyz[idx] << 1) + np.array([(bit >> 2) & 1, (bit >> 1) & 1, bit & 1], dtype=np.int64)
        out.append(ch)
    return np.unique(np.concatenate(out, 0), axis=0) if out else np.zeros((0, 3), dtype=np.int64)


SLOT_OCTANT = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)],
                       dtype=np.int64)


def expand_two_level(parent_keys: np.ndarray, slots: np.ndarray, p: int) -> np.ndarray:
    if len(parent_keys) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    pxyz = _unkey(parent_keys, p)
    cxyz = ((pxyz[:, None, :] << 1) + SLOT_OCTANT[None]).reshape(-1, 3)
    vals = slots.reshape(-1)
    keep = vals != 0
    if not keep.any():
        return np.zeros((0, 3), dtype=np.int64)
    return expand_children(_key(cxyz[keep], p + 1), vals[keep], p + 1)


