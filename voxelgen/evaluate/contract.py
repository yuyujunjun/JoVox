import csv
import glob
import os

import numpy as np

EVAL_GRID = 64


def voxel_dir(root):
    return os.path.join(root, "voxels")


def _io():
    import torch
    import o_voxel
    return torch, o_voxel


_AX = {"X": 0, "Y": 1, "Z": 2}
IDENTITY = "+X+Y+Z"


def _mat(name):
    assert isinstance(name, str) and len(name) == 6, f"rotation name must have 6 characters, got {name!r}"
    M = np.zeros((3, 3), int)
    for i in range(3):
        s, a = name[2 * i], name[2 * i + 1]
        assert s in "+-" and a in _AX, f"{name!r}: invalid item {i}"
        M[i, _AX[a]] = 1 if s == "+" else -1
    assert sorted(np.abs(M).argmax(1)) == [0, 1, 2], f"{name!r} is not an axis permutation"
    assert round(float(np.linalg.det(M))) == 1, f"{name!r} is a reflection, not a rotation"
    return M


def _all_names():
    import itertools
    out = []
    for p in itertools.permutations("XYZ"):
        for s in itertools.product("+-", repeat=3):
            n = "".join(a + b for a, b in zip(s, p))
            try:
                _mat(n); out.append(n)
            except AssertionError:
                pass
    return out


def rotate(coord, rot, grid):
    M = _mat(rot)
    out = np.empty_like(coord)
    for i in range(3):
        a = int(np.argmax(np.abs(M[i])))
        out[:, i] = coord[:, a] if M[i, a] > 0 else (grid - 1 - coord[:, a])
    return out


def normalize_tight(coord, grid):
    lo, hi = coord.min(0), coord.max(0)
    e = int((hi - lo).max())
    assert e >= 1, "all voxels are at one point, cannot normalize"
    k = grid / (e + 1)
    assert k >= 1, f"k={k:.4f} < 1: source larger than target"
    c_s = (lo + hi + 1) / 2.0
    t = np.arange(grid)
    out = coord.astype(np.int64)
    for ax in range(3):
        lut = np.floor((t + 0.5 - grid / 2.0) / k + c_s[ax]).astype(np.int64)
        vals, starts, counts = np.unique(lut, return_index=True, return_counts=True)
        start_of = np.full(grid + 2, -1, np.int64)
        count_of = np.zeros(grid + 2, np.int64)
        ok = (vals >= 0) & (vals < grid)
        start_of[vals[ok]] = starts[ok]
        count_of[vals[ok]] = counts[ok]
        v = out[:, ax]
        n = count_of[v]
        assert (n >= 1).all(), "a source cell has no target cell"
        rep = np.repeat(out, n, axis=0)
        offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        rep[:, ax] = np.repeat(start_of[v], n) + offs
        out = rep
    out = np.unique(out, axis=0)
    span = out.max(0) - out.min(0)
    assert span.max() == grid - 1, (
        f"longest axis does not span the grid after normalization: span={span.max()}, expected {grid - 1}")
    return out, k


def maxpool(coord, src_grid, dst_grid):
    assert src_grid % dst_grid == 0, f"{src_grid} is not a multiple of {dst_grid}"
    k = src_grid // dst_grid
    return np.unique(coord if k == 1 else coord // k, axis=0)


def to_eval(coord, src_grid, rot, eval_grid=EVAL_GRID):
    c = np.asarray(coord)
    if c.ndim == 3:
        assert c.shape == (src_grid,) * 3, f"dense array {c.shape} does not match src_grid={src_grid}"
        c = np.argwhere(c > 0)
    assert c.ndim == 2 and c.shape[1] == 3, f"unsupported coord shape: {c.shape}"
    c = c.astype(np.int64)
    assert len(c) > 0, "empty occupancy"
    assert c.min() >= 0 and c.max() < src_grid, \
        f"indices [{c.min()},{c.max()}] not in [0,{src_grid})"
    r = rotate(c, rot, src_grid)
    bb_min, bb_max = r.min(0), r.max(0)
    n, k = normalize_tight(r, src_grid)
    return {"voxels": maxpool(n, src_grid, eval_grid), "native": n,
            "src_bbox_min": bb_min, "src_bbox_max": bb_max, "k": k}


def _write_vxz(path, coord):
    torch, o_voxel = _io()
    assert not os.path.exists(path), (
        f"{path} already exists")
    t = torch.as_tensor(np.ascontiguousarray(coord), dtype=torch.int32)
    t = t[torch.argsort(o_voxel.serialize.encode_seq(t))]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    o_voxel.io.write_vxz(tmp, t, {"o": torch.zeros(len(t), 1, dtype=torch.uint8)},
                         compression="zstd", num_threads=1)
    back = o_voxel.io.read_vxz(tmp)[0].int()
    assert back.shape == t.shape, f"read-back voxel count differs: {tuple(back.shape)} vs {tuple(t.shape)}"
    assert torch.equal(torch.sort(o_voxel.serialize.encode_seq(back)).values,
                       torch.sort(o_voxel.serialize.encode_seq(t)).values), "read-back voxels differ"
    os.rename(tmp, path)
    return len(t)


_ANNOUNCED = set()
_MANIFEST_HEADER = "sample_id,n_occ_64,n_occ_512,src_grid,rot,k,src_bbox_min,src_bbox_max,seed\n"


def save_sample(out_root, sample_id, coord, src_grid, seed=None):
    rot = IDENTITY
    key = (out_root, src_grid, rot)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        print(f"[contract] {out_root}  {src_grid}^3 --rot {rot}--> normalize --maxpool//"
              f"{src_grid // EVAL_GRID}--> {EVAL_GRID}^3", flush=True)
    r = to_eval(coord, src_grid, rot)
    n64 = _write_vxz(os.path.join(out_root, "voxels", f"{sample_id}.vxz"), r["voxels"])
    n512 = ""
    if src_grid >= 512:
        n512 = _write_vxz(os.path.join(out_root, "dual_grid_512", f"{sample_id}.vxz"),
                          maxpool(r["native"], src_grid, 512))
    mpath = os.path.join(out_root, "manifest.csv")
    os.makedirs(out_root, exist_ok=True)
    fd = os.open(mpath, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        if os.fstat(fd).st_size == 0:
            os.write(fd, _MANIFEST_HEADER.encode())
        bmin = " ".join(map(str, r["src_bbox_min"])); bmax = " ".join(map(str, r["src_bbox_max"]))
        os.write(fd, f"{sample_id},{n64},{n512},{src_grid},{rot},{r['k']:.5f},"
                     f"{bmin},{bmax},{'' if seed is None else seed}\n".encode())
    finally:
        os.close(fd)
    return n64


def already_done(out_root, sample_id):
    p = os.path.join(out_root, "voxels", f"{sample_id}.vxz")
    if not os.path.exists(p):
        return False
    try:
        _, o_voxel = _io()
        o_voxel.io.read_vxz_info(p)
    except Exception:
        return False
    mpath = os.path.join(out_root, "manifest.csv")
    if not os.path.exists(mpath):
        return False
    with open(mpath) as f:
        return any(line.split(",", 1)[0] == sample_id for line in f)



UP_AXIS = "z"
DATASETS = ("toys4k",)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_DATA = os.path.join(REPO, "data")
GEN = os.path.join(_DATA, "gen")
COND = os.path.join(_DATA, "splits", "eval_cond.csv")
SPLIT_OF = {"toys4k": "test"}


def cond_rows(ds):
    rows = [r for r in csv.DictReader(open(COND)) if r["split"] == SPLIT_OF[ds]]
    assert rows, f"{COND} has no rows with split {SPLIT_OF[ds]}"
    assert os.path.exists(rows[0]["image_path"]), f"condition image not found: {rows[0]['image_path']}"
    return rows


def leaf_of(method, step=None, gen_seed=None, n=None):
    if method == "gt":
        assert step is None and gen_seed is None and n is None, "gt takes no --step / --gen_seed / --n"
        return ""
    if step is None:
        assert gen_seed is None, f"{method}: --gen_seed requires --step"
        return "released" + (f"_n{n}" if n else "")
    return f"s{int(step):06d}" + (f"_seed{gen_seed}" if gen_seed is not None else "") + (f"_n{n}" if n else "")


def root_of(dataset, method, step=None, gen_seed=None, n=None):
    assert dataset in DATASETS, f"dataset must be one of {sorted(DATASETS)}, got {dataset!r}"
    if n is not None:
        size = len(cond_rows(dataset))
        assert 0 < n < size, f"--n {n} must be in (0, {size})"
    leaf = leaf_of(method, step, gen_seed, n)
    if method == "gt":
        return os.path.join(GEN, dataset, "gt")
    return os.path.join(GEN, dataset, method, leaf)


def gt_of(dataset):
    return root_of(dataset, "gt")


def eval_dir(root, ref_dataset):
    assert ref_dataset in DATASETS, f"reference dataset must be one of {sorted(DATASETS)}, got {ref_dataset!r}"
    return os.path.join(root, "eval", ref_dataset)


def add_identity_args(ap, ref=True):
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS), help="evaluation dataset")
    ap.add_argument("--method", required=True,
                    help="gt, or a model spec name")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (model specs only)")
    ap.add_argument("--gen_seed", type=int, default=None, help="generation seed (model specs only)")
    ap.add_argument("--n", type=int, default=None, help="use the first N entries of the dataset")
    if ref:
        ap.add_argument("--ref", default=None, choices=sorted(DATASETS), help="reference dataset (default: --dataset)")


def resolve(a, must_exist=True):
    root = root_of(a.dataset, a.method, a.step, a.gen_seed, a.n)
    ref = getattr(a, "ref", None) or a.dataset
    print(f"[data] {GEN}", flush=True)
    if must_exist and not os.path.isdir(root):
        d = os.path.join(GEN, a.dataset)
        ms = sorted(m for m in (os.listdir(d) if os.path.isdir(d) else []) if os.path.isdir(os.path.join(d, m)))
        have = ["gt" if m == "gt" else f"{m}/{l}" for m in ms
                for l in (["."] if m == "gt" else sorted(os.listdir(os.path.join(d, m))))
                if m == "gt" or os.path.isdir(os.path.join(d, m, l))]
        raise SystemExit(f"root not found: {root}\navailable under {a.dataset}:\n  " + "\n  ".join(have or ["(none)"]))
    return root, ref, gt_of(ref)


_MESH_BASE = 512
_MESH_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
_MESH_FIT = 0.99999
GLTF_TO_ZUP = "+X-Z+Y"


class EmptyMeshError(AssertionError):
    pass


def load_mesh_zup(file):
    import trimesh
    m = trimesh.load(file, process=False, force='mesh', skip_materials=True)
    if not isinstance(m, trimesh.Trimesh) or m.faces is None or len(m.faces) == 0:
        raise EmptyMeshError(f'mesh has no faces ({type(m).__name__}): {file}')
    v = np.asarray(m.vertices, dtype=np.float64)
    if str(file).lower().endswith(('.glb', '.gltf')):
        v = v @ _mat(GLTF_TO_ZUP).astype(np.float64).T
    return v, np.asarray(m.faces, dtype=np.int64)


def mesh_to_coord(file):
    import torch
    from o_voxel.convert import mesh_to_flexible_dual_grid
    verts, faces = load_mesh_zup(file)
    v = torch.as_tensor(verts, dtype=torch.float32, device='cuda')
    f = torch.as_tensor(faces, dtype=torch.int32, device='cuda')
    vmin, vmax = v.min(dim=0)[0], v.max(dim=0)[0]
    extent = (vmax - vmin).max()
    assert extent > 1e-8, f'degenerate mesh, longest side {float(extent)}'
    v = (v - (vmin + vmax) / 2) * (_MESH_FIT / extent)
    assert v.min() >= -0.5 and v.max() <= 0.5, 'out of bounds after normalization'
    coord, _, _ = mesh_to_flexible_dual_grid(
        v, f, grid_size=_MESH_BASE, aabb=_MESH_AABB, return_voxel_only=True)
    assert coord.numel() > 0, 'empty occupancy'
    assert coord.min() >= 0 and coord.max() < _MESH_BASE, \
        f'coordinates out of bounds [{int(coord.min())},{int(coord.max())}]'
    return coord.int().cpu().numpy()


_DEGEN_HEADER = "sample_id,reason,n_occ_raw\n"


def _load_degenerate(root):
    p = os.path.join(root, "degenerate.csv")
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return {ln.split(",", 1)[0] for ln in f.read().splitlines()[1:] if ln.strip()}


def _record_degenerate(root, sid, reason, n_occ_raw):
    p = os.path.join(root, "degenerate.csv")
    os.makedirs(root, exist_ok=True)
    first = not os.path.exists(p)
    with open(p, "a") as f:
        if first:
            f.write(_DEGEN_HEADER)
        f.write(f"{sid},{reason},{n_occ_raw}\n")
    print(f"  [{sid}] skipped ({reason}), recorded in degenerate.csv", flush=True)


def voxelize_gt(dataset: str, src: str, pattern: str):
    out_root = root_of(dataset, "gt")
    print(f"[cfg] root = {out_root}\n[cfg] source = {src}", flush=True)

    def _key(f):
        st = os.path.splitext(os.path.basename(f))[0]
        return (0, int(st)) if st.isdigit() else (1, st)
    fs = sorted(glob.glob(os.path.join(src, pattern)), key=_key)
    assert fs, f"no {pattern} files in {src}"
    done = skipped = degen = 0
    degen_ids = _load_degenerate(out_root)
    for f in fs:
        stem = os.path.splitext(os.path.basename(f))[0]
        sid = "%06d" % int(stem) if stem.isdigit() else stem
        if already_done(out_root, sid) or sid in degen_ids:
            skipped += 1; continue
        try:
            c = mesh_to_coord(f)
        except EmptyMeshError:
            _record_degenerate(out_root, sid, "empty_mesh", 0)
            degen += 1
            continue
        n = save_sample(out_root, sid, c, _MESH_BASE)
        done += 1
        if done <= 3:
            print(f"  [{sid}] {os.path.basename(f)}  512^3 occupied {len(c)} -> 64^3 {n}", flush=True)
    print(f"[voxelize-mesh] wrote {done}, skipped {skipped} -> {out_root}")
    n_deg_all = len(_load_degenerate(out_root))
    print(f"[degenerate] {degen} in this run; {n_deg_all} / {len(fs)} "
          f"= {100.0 * n_deg_all / len(fs):.3f}% in total", flush=True)
    vdir = os.path.join(out_root, "voxels")
    n_vox = len(os.listdir(vdir)) if os.path.isdir(vdir) else 0
    print(f"[reconcile] voxels {n_vox} + degenerate {n_deg_all} = {n_vox + n_deg_all} (source meshes {len(fs)})", flush=True)
    assert n_vox + n_deg_all == len(fs), \
        f"[reconcile] voxels {n_vox} + degenerate {n_deg_all} != source meshes {len(fs)}"
