import os

import numpy as np
import torch

import o_voxel

SUBDIR_GRID = {"dual_grid_512": 512, "voxels": 64}


def rotate_coords_x90(coord: np.ndarray, grid_size: int) -> np.ndarray:
    c = np.asarray(coord)
    assert c.ndim == 2 and c.shape[1] == 3, f"coordinates must be [M,3], got {c.shape}"
    assert c.size > 0, "empty occupancy"
    lo, hi = int(c.min()), int(c.max())
    assert lo >= 0 and hi < grid_size, \
        f"indices [{lo},{hi}] not in [0,{grid_size}); wrong grid_size?"
    out = np.empty_like(c)
    out[:, 0] = c[:, 0]
    out[:, 1] = grid_size - 1 - c[:, 2]
    out[:, 2] = c[:, 1]
    return out


def convert_file(src: str, dst: str, grid_size: int, overwrite: bool = False) -> int:
    if os.path.exists(dst) and not overwrite:
        return -1

    src_coord = o_voxel.io.read_vxz(src)[0]
    n_in = int(src_coord.shape[0])
    rot = rotate_coords_x90(src_coord.int().cpu().numpy(), grid_size)

    assert len(np.unique(rot, axis=0)) == n_in, \
        f"occupancy count changed after rotation: {n_in} -> {len(np.unique(rot, axis=0))}"

    c = torch.as_tensor(np.ascontiguousarray(rot), dtype=torch.int32)
    c = c[torch.argsort(o_voxel.serialize.encode_seq(c))]
    tmp = dst + ".tmp"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    o_voxel.io.write_vxz(tmp, c, {"o": torch.zeros(len(c), 1, dtype=torch.uint8)},
                         compression="zstd", num_threads=1)

    back = o_voxel.io.read_vxz(tmp)[0].int()
    assert back.shape == c.shape, \
        f"voxel count differs after read-back {tuple(back.shape)} vs {tuple(c.shape)}"
    a = torch.sort(o_voxel.serialize.encode_seq(back)).values
    b = torch.sort(o_voxel.serialize.encode_seq(c)).values
    assert torch.equal(a, b), "voxel set differs after read-back"
    os.rename(tmp, dst)
    return n_in


def convert_tree(src_root: str, dst_root: str, max_workers: int = 8,
                 rank: int = 0, world_size: int = 1) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    stats = {}
    for sub, g in SUBDIR_GRID.items():
        d = os.path.join(src_root, sub)
        if not os.path.isdir(d):
            print(f"[skip] not found: {d}", flush=True)
            continue
        names = sorted(x for x in os.listdir(d) if x.endswith(".vxz"))
        s, e = len(names) * rank // world_size, len(names) * (rank + 1) // world_size
        names = names[s:e]
        print(f"[{sub}] grid={g} this rank: {len(names)} files ({s}..{e})", flush=True)

        done = skipped = 0

        def work(n):
            return convert_file(os.path.join(d, n),
                                os.path.join(dst_root, sub, n), g)

        with ThreadPoolExecutor(max_workers) as ex:
            for i, r in enumerate(ex.map(work, names)):
                if r < 0:
                    skipped += 1
                else:
                    done += 1
                if (i + 1) % 2000 == 0:
                    print(f"  {sub} {i+1}/{len(names)}", flush=True)
        stats[sub] = {"converted": done, "skipped": skipped, "total": len(names)}
        print(f"[{sub}] converted {done}, skipped (already exist) {skipped}", flush=True)
    return stats


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="convert .vxz occupancy from the glTF frame (Y-up) to the Blender frame (Z-up)")
    p.add_argument("--src_root", required=True, help="subset directory containing dual_grid_512/ and voxels/")
    p.add_argument("--dst_root", required=True, help="output directory (must differ from --src_root)")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--max_workers", type=int, default=8)
    a = p.parse_args()
    assert os.path.abspath(a.src_root) != os.path.abspath(a.dst_root), \
        "--dst_root must differ from --src_root: converting in place twice would rotate by 180 degrees"
    print(f"[cfg] {a.src_root} -> {a.dst_root}", flush=True)
    print(f"[cfg] rank={a.rank}/{a.world_size} max_workers={a.max_workers}", flush=True)
    print("[cfg] rotation Rx(+90): (i,j,k) -> (i, G-1-k, j)", flush=True)
    st = convert_tree(a.src_root, a.dst_root, a.max_workers, a.rank, a.world_size)
    print("[done]", st, flush=True)
