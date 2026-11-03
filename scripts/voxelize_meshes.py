import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import torch
import trimesh
from easydict import EasyDict as edict
from tqdm import tqdm

import o_voxel
from o_voxel.convert import mesh_to_flexible_dual_grid

SUBDIR = {512: 'dual_grid_512', 64: 'voxels'}
AABB   = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
BASE   = 512
FIT    = 0.99999


def _sort_morton(coord):
    return coord[torch.argsort(o_voxel.serialize.encode_seq(coord))]


def _mesh_to_coord(file):
    mesh = trimesh.load(file, process=False, force='mesh', skip_materials=True)
    assert mesh.faces is not None and len(mesh.faces) > 0, 'mesh has no faces'
    v = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device='cuda')
    f = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int32, device='cuda')
    vmin, vmax = v.min(dim=0)[0], v.max(dim=0)[0]
    extent = (vmax - vmin).max()
    assert extent > 1e-8, f'degenerate mesh, longest extent {float(extent)}'
    v = (v - (vmin + vmax) / 2) * (FIT / extent)
    assert v.min() >= -0.5 and v.max() <= 0.5, 'out of bounds after normalization'
    coord, _, _ = mesh_to_flexible_dual_grid(
        v, f, grid_size=BASE, aabb=AABB, return_voxel_only=True)
    assert coord.numel() > 0, 'empty occupancy'
    assert coord.min() >= 0 and coord.max() < BASE, \
        f'coordinates out of range [{int(coord.min())},{int(coord.max())}]'
    return _sort_morton(coord)


def _write_checked(path, coord):
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    o_voxel.io.write_vxz(
        tmp, coord, {'o': torch.zeros(len(coord), 1, dtype=torch.uint8)},
        compression='zstd', num_threads=1)
    back = o_voxel.io.read_vxz(tmp)[0].int().cpu()
    assert back.shape == coord.shape, \
        f'voxel count differs after read-back: {tuple(back.shape)} vs {tuple(coord.shape)}'
    a = torch.sort(o_voxel.serialize.encode_seq(back)).values
    b = torch.sort(o_voxel.serialize.encode_seq(coord.int())).values
    assert torch.equal(a, b), 'voxel set differs after read-back'
    os.rename(tmp, path)


def _voxelize(file, metadatum, root, resolutions, pbar=None):
    sha256 = metadatum['sha256']
    try:
        pack = {'sha256': sha256}
        coord_base = None
        for res in resolutions:
            out = os.path.join(root, SUBDIR[res], f'{sha256}.vxz')
            need = False
            if os.path.exists(out):
                try:
                    info = o_voxel.io.read_vxz_info(out)
                    pack[f'voxelized_{res}'] = True
                    pack[f'num_voxels_{res}'] = info['num_voxel']
                except Exception as e:
                    print(f'[redo] {sha256} {res}: cannot read existing file ({e})', flush=True)
                    need = True
            else:
                need = True

            if need:
                if coord_base is None:
                    coord_base = _mesh_to_coord(file)
                if res == BASE:
                    coord = coord_base
                else:
                    assert BASE % res == 0, f'{BASE} is not divisible by {res}'
                    coord = _sort_morton(torch.unique(coord_base // (BASE // res), dim=0))
                coord = coord.cpu()
                _write_checked(out, coord)
                pack[f'voxelized_{res}'] = True
                pack[f'num_voxels_{res}'] = len(coord)
        return pack
    except Exception as e:
        print(f'Error voxelizing {sha256}: {e}', flush=True)
        return {'sha256': sha256, 'error': str(e)}


def foreach_instance(metadata, func, max_workers):
    rows = metadata.to_dict('records')
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(rows), desc='Voxelizing') as pbar:
        def work(m):
            r = func(m['local_path'], m)
            pbar.update()
            return r
        records = list(executor.map(work, rows))
    return pd.DataFrame.from_records(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meshes listed in <root>/metadata.csv (columns sha256, local_path) '
                                                 '-> 512^3 and 64^3 occupancy (.vxz)')
    parser.add_argument('--root', type=str, required=True,
                        help='subset root directory containing metadata.csv')
    parser.add_argument('--instances', type=str, default=None,
                        help='only these sha256 ids (a file, or comma-separated)')
    parser.add_argument('--resolution', type=str, default='512,64')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.resolution = sorted([int(x) for x in opt.resolution.split(',')], reverse=True)

    assert torch.cuda.is_available(), 'voxelization requires a GPU'
    assert opt.resolution[0] == BASE, f'the highest resolution must be {BASE}, got {opt.resolution}'
    print(f'[cfg] root={opt.root}', flush=True)
    print(f'[cfg] resolution={opt.resolution} -> {[SUBDIR[r] for r in opt.resolution]}', flush=True)
    print(f'[cfg] rank={opt.rank}/{opt.world_size}  max_workers={opt.max_workers}', flush=True)
    print(f'[cfg] device={torch.cuda.get_device_name(0)}  '
          f'cpus={len(os.sched_getaffinity(0))}', flush=True)
    for res in opt.resolution:
        os.makedirs(os.path.join(opt.root, SUBDIR[res]), exist_ok=True)
    os.makedirs(os.path.join(opt.root, 'new_records'), exist_ok=True)

    meta_path = os.path.join(opt.root, 'metadata.csv')
    assert os.path.exists(meta_path), f'not found: {meta_path}'
    metadata = pd.read_csv(meta_path)
    if opt.instances is not None:
        if os.path.exists(opt.instances):
            with open(opt.instances) as f:
                inst = f.read().splitlines()
        else:
            inst = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(inst)]

    probe = metadata.iloc[0]['local_path']
    assert os.path.exists(probe), (
        f'source mesh not found: {probe}')

    start = len(metadata) * opt.rank // opt.world_size
    end   = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    print(f'[cfg] this rank: {len(metadata)} instances ({start}..{end})', flush=True)

    func = partial(_voxelize, root=opt.root, resolutions=opt.resolution)
    done = foreach_instance(metadata, func, opt.max_workers)

    if 'error' in done.columns:
        errs = done[done['error'].notna()]
        if len(errs) > 0:
            with open(os.path.join(opt.root, 'new_records',
                                   f'errors_{opt.rank}.txt'), 'w') as f:
                f.write('\n'.join(f"{r['sha256']}\t{r['error']}"
                                  for _, r in errs.iterrows()))
            print(f'[done] {len(errs)} failed, listed in new_records/errors_{opt.rank}.txt', flush=True)
        done = done[done['error'].isna()]
    cols = ['sha256'] + [c for c in done.columns if c.startswith(('voxelized_', 'num_voxels_'))]
    if len(done) > 0:
        done[cols].to_csv(os.path.join(opt.root, 'new_records',
                                       f'part_{opt.rank}.csv'), index=False)
    print(f'[done] rank {opt.rank}: {len(done)} succeeded', flush=True)
