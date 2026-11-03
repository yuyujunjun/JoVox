# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from OpenLRM (https://github.com/3DTopia/OpenLRM).

import functools
import json
import math
import os
import random
from typing import Iterator, List

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DistributedSampler

from voxelgen.network.hierarchy import ROOT_LEVEL, _down, _key, child_pattern, _read_vxz_coords, native_level_of
from .augment import color_aug, rot90_z, _load_targets, TARGETS_PATH


def build_voxel_pyramid(coords, where: str, native_level: int):
    nl = native_level
    c64 = np.unique(np.asarray(coords).astype(np.int64), axis=0)
    assert c64.ndim == 2 and c64.shape[1] == 3, \
        f'occupancy coords must be [M,3], got {c64.shape} ({where})'
    assert int(c64.max()) < 2 ** nl, \
        (f'max occupancy coord {int(c64.max())} exceeds {2 ** nl}^3 '
         f'({where})')
    per_level = {nl: c64}
    for lv in range(nl - 1, ROOT_LEVEL - 1, -1):
        per_level[lv] = _down(per_level[lv + 1], 1)
    stages = []
    for si, (p, c) in enumerate([(l, l + 1)
                                 for l in range(ROOT_LEVEL, nl)]):
        root_stage = (si == 0)
        pk, lab = child_pattern(p, per_level[c], all_cells=root_stage)
        ref = np.sort(_key(per_level[p], p))
        if root_stage:
            assert np.array_equal(pk[lab > 0], ref), \
                f'root level {p}->{c} max-pool mismatch ({where})'
        else:
            assert np.array_equal(pk, ref), \
                f'level {si} {p}->{c} max-pool mismatch ({where})'
            assert lab.min() >= 1, \
                f'empty pattern at non-root level {p}->{c} ({where})'
        stages.append((pk, lab))
    return tuple(stages), (c64.astype(np.int16),)


def load_rgba_image(file_path):
    with open(file_path, 'rb') as f:
        rgba = np.array(Image.open(f))
    rgba = torch.from_numpy(rgba).float() / 255.0
    rgba = rgba.permute(2, 0, 1).unsqueeze(0)
    return rgba[:, :3, :, :], rgba[:, 3:4, :, :]


class VoxelDataset(Dataset):

    COND_ELEV_LO = -10.0
    COND_ELEV_HI = 30.0

    def __init__(self, data_root: str, meta_path: str, image_root: str, train: bool, native_level: int):
        self.data_root = data_root
        self.image_root = image_root
        self.train = train
        self.native_level = int(native_level)
        assert self.native_level in (6, 8), f"[dataset] native_level must be 6 (64^3) or 8 (256^3), got {self.native_level}"
        print(f"[dataset] ground truth {2 ** self.native_level}^3, read from "
              f"{'dual_grid_512 (512^3, max-pooled by 2)' if self.native_level == 8 else 'voxels/'}", flush=True)

        df = pd.read_csv(meta_path)
        missing = {'subset', 'uid'} - set(df.columns)
        assert not missing, (
            f"[dataset] split file {meta_path} is missing columns {sorted(missing)}")
        self.rows = list(zip(df['subset'].tolist(), df['uid'].tolist()))
        self.uids = [u for _, u in self.rows]

        if train:
            n_t = len(_load_targets())
        print(f"[dataset] {'train' if train else 'val'} augmentation: condition image color "
              f"{f'match ({n_t} targets, {TARGETS_PATH})' if train else 'off'}; "
              f"random 90-degree voxel rotation about z {'on' if train else 'off'}", flush=True)

        sub0, uid0 = self.rows[0]
        v0 = self._vxz_path(sub0, uid0)
        assert os.path.isfile(v0), f"[dataset] occupancy file not found: {v0}"
        n0 = len(self._image_files(sub0, uid0))
        assert n0 > 0, f"[dataset] condition images not found: {self._image_dir(sub0, uid0)}"
        print(f"[dataset] {'train' if train else 'val'} {len(self.rows)} assets | "
              f"occupancy root {self.data_root} | image root {self.image_root} | "
              f"first row {sub0}/{uid0[:12]}... {n0} condition images", flush=True)

        probe = self.rows[::max(1, len(self.rows) // 200)][:200]
        hit, n_in = 0, 0
        for sub, u in probe:
            c, ok = self._cond_candidates(sub, u)
            hit += int(ok)
            n_in += len(c) if ok else 0
        print(f"[dataset] condition image elevation window [{self.COND_ELEV_LO:.0f}, "
              f"{self.COND_ELEV_HI:.0f}] deg: of {len(probe)} probed assets, "
              f"{hit} have images in the window (mean {n_in / max(1, hit):.1f}), "
              f"{len(probe) - hit} fall back to all images", flush=True)

    def _vxz_path(self, subset: str, uid: str) -> str:
        sub = 'dual_grid_512' if self.native_level == 8 else 'voxels'
        return os.path.join(self.data_root, subset, sub, uid + '.vxz')

    def _image_dir(self, subset: str, uid: str) -> str:
        return os.path.join(self.image_root, subset, 'renders3', uid)

    def _image_files(self, subset: str, uid: str):
        d = self._image_dir(subset, uid)
        return [f for f in os.listdir(d)
                if f.endswith('.png') and not f.endswith('_depth.png')]

    def _cond_candidates(self, subset: str, uid: str):
        files = self._image_files(subset, uid)
        tj = os.path.join(self._image_dir(subset, uid), 'transforms.json')
        try:
            with open(tj) as fh:
                frames = json.load(fh)['frames']
        except (OSError, ValueError, KeyError):
            return files, False
        have = set(files)
        win = []
        for f in frames:
            name = os.path.basename(f['file_path'])
            if name not in have:
                continue
            m = f['transform_matrix']
            x, y, z = m[0][3], m[1][3], m[2][3]
            r = math.sqrt(x * x + y * y + z * z)
            if r <= 0.0:
                continue
            el = math.degrees(math.asin(max(-1.0, min(1.0, z / r))))
            if self.COND_ELEV_LO <= el <= self.COND_ELEV_HI:
                win.append(name)
        return (win, True) if win else (files, False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            n = len(self.uids)
            uid = self.uids[idx % n] if n else "<empty>"
            print(f"[dataset] sample {idx} (uid {uid}) failed to load: "
                  f"{type(e).__name__}: {e}; using the next one", flush=True)
            return self.__getitem__((idx + 1) % n)

    def inner_get_item(self, idx):
        subset, uid = self.rows[idx % len(self.rows)]

        coords64 = _read_vxz_coords(self._vxz_path(subset, uid))
        if self.native_level == 8:
            coords64 = np.unique(np.asarray(coords64, np.int64) >> 1, axis=0)
        if self.train:
            coords64 = rot90_z(coords64, random.randrange(4), grid=2 ** self.native_level)
        voxel_pyramid, voxel_occ_gt = build_voxel_pyramid(coords64, f'{subset}/{uid}', native_level=self.native_level)

        files, _in_window = self._cond_candidates(subset, uid)
        assert files, f"[dataset] no condition images in {self._image_dir(subset, uid)}"
        images, masks = load_rgba_image(os.path.join(self._image_dir(subset, uid), random.choice(files)))
        if self.train:
            images, _ = color_aug(images, masks)

        return {
            "images": images,
            "masks": masks,
            "n_occ": len(coords64),
            "voxel_pyramid": voxel_pyramid,
            "voxel_occ_gt": voxel_occ_gt,
        }


def load_balanced_group_indices(load: List[int], num_groups: int, equal_size: bool = False) -> List[List[int]]:
    if equal_size:
        group_size = len(load) // num_groups
    indices = np.argsort(load)[::-1]
    groups = [[] for _ in range(num_groups)]
    group_load = np.zeros(num_groups)
    for idx in indices:
        min_group_idx = np.argmin(group_load)
        groups[min_group_idx].append(idx)
        if equal_size and len(groups[min_group_idx]) == group_size:
            group_load[min_group_idx] = float('inf')
        else:
            group_load[min_group_idx] += load[idx]
    return groups


def collate_fn(batch, sort_by_size: bool):
    order = load_balanced_group_indices([b['n_occ'] for b in batch], 1)[0] if sort_by_size \
        else list(range(len(batch)))
    sub = [batch[i] for i in order]
    return [{
        "images": torch.stack([b["images"] for b in sub]),
        "masks": torch.stack([b["masks"] for b in sub]),
        "voxel_pyramid": [b["voxel_pyramid"] for b in sub],
        "voxel_occ_gt": [b["voxel_occ_gt"] for b in sub],
    }]


class VirtualLenWrapper:
    def __init__(self, ds, vlen, loads):
        self._ds = ds
        self._len = vlen
        self.loads = loads

    def __getitem__(self, idx):
        return self._ds[idx]

    def __len__(self):
        return self._len


class BalancedResumableSampler(DistributedSampler):

    def __init__(self, dataset, batch_size: int, weights, seed: int = 0):
        assert hasattr(dataset, 'loads'), 'Dataset must have "loads" attribute to use BalancedResumableSampler'
        self.dataset = dataset
        self.epoch = 0
        self.idx = 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_samples = math.ceil(len(self.dataset) / self.world_size)
        self.total_size = self.num_samples * self.world_size
        self.shuffle = True
        self.seed = seed
        self.drop_last = False
        self.generator = torch.Generator()
        g_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator.manual_seed(g_seed)
        self.batch_size = batch_size
        self.loads = dataset.loads
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        self._is_distributed = True

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, num_samples=self.total_size, replacement=True, generator=g).tolist()
        assert len(indices) == self.total_size
        num_batches = len(indices) // (self.batch_size * self.world_size)
        balanced_indices = []
        for i in range(num_batches):
            start_idx = i * self.batch_size * self.world_size
            end_idx = (i + 1) * self.batch_size * self.world_size
            batch_indices = indices[start_idx:end_idx]
            batch_loads = [self.loads[idx] for idx in batch_indices]
            groups = load_balanced_group_indices(batch_loads, self.world_size, equal_size=True)
            balanced_indices.extend([batch_indices[j] for j in groups[self.rank]])
        assert self.idx == 0
        return iter(balanced_indices[self.idx:])

    def state_dict(self) -> dict:
        return {'epoch': self.epoch, 'idx': self.idx}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.idx = state_dict['idx']


def build_dataloader(cfg, split: str, balanced: bool):
    train = split == 'train'
    ds = VoxelDataset(cfg.dataset.root, cfg.dataset.splits[split], cfg.dataset.image_root, train=train,
                      native_level=native_level_of(cfg.model.num_levels))
    wrapped = VirtualLenWrapper(ds, len(ds), [1] * len(ds))
    batch_size = cfg.train.batch_size if train else cfg.val.batch_size
    sampler = (BalancedResumableSampler(wrapped, batch_size=batch_size, weights=torch.DoubleTensor([1.0] * len(ds)))
               if balanced else None)
    return torch.utils.data.DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False if sampler is not None else True,
        drop_last=False,
        num_workers=cfg.dataset.num_train_workers if train else cfg.dataset.num_val_workers,
        pin_memory=cfg.dataset.pin_mem,
        persistent_workers=True,
        sampler=sampler,
        collate_fn=functools.partial(collate_fn, sort_by_size=train),
    )
