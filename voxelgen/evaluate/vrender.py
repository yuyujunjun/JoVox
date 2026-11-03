import argparse
import glob
import json
import os
import time

import numpy as np

GRID = 64
N_VIEWS = 6
RDIR = "renders6_vox"
CACHE = "feats_vrender.npz"
RENDERER = "render_voxel_frame(pbr)|make_projection_mtx+Rz60|512|black"


def up_transform(c, up):
    if up == "z":
        return c
    assert up == "y", f"--up must be z or y, got {up!r}"
    return np.stack([c[:, 0], GRID - 1 - c[:, 2], c[:, 1]], 1)


def cameras(device):
    import torch
    import utils3d
    from voxelgen.interfaces.render import make_projection_mtx, perspective
    mtx0, orig0 = make_projection_mtx(device)
    fov = float(np.deg2rad(40.0))
    proj = perspective(fov, 0.01, 100.0).to(device)
    cams = []
    for k in range(N_VIEWS):
        a = np.deg2rad(60.0 * k)
        Rz = torch.tensor([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]],
                          dtype=torch.float32, device=device)
        o = Rz @ orig0
        extr = utils3d.torch.extrinsics_look_at(o, torch.zeros(3, device=device),
                                                torch.tensor([0.0, 0.0, 1.0], device=device))
        cams.append((proj @ extr, o))
    assert torch.allclose(cams[0][0], mtx0) and torch.allclose(cams[0][1], orig0), \
        "camera 0 does not match make_projection_mtx"
    return cams


def _meta_check(root, up):
    d = os.path.join(root, RDIR)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "_meta.json")
    want = {"up": up, "renderer": RENDERER, "grid": GRID, "n_views": N_VIEWS}
    if os.path.exists(p):
        have = json.load(open(p))
        assert have == want, f"{p} records {have}, this run uses {want}; one root cannot mix two render settings"
    else:
        tmp = p + f".tmp{os.getpid()}"
        json.dump(want, open(tmp, "w")); os.replace(tmp, p)


def _views(root, name):
    return [os.path.join(root, RDIR, name, f"{v:03d}.png") for v in range(N_VIEWS)]


def _to_png_array(img_u8):
    import torch
    img = torch.from_numpy(img_u8).float() / 255.0
    return (img.numpy() * 255).clip(0, 255).astype(np.uint8)


def render_views(b0, glctx, cams, device="cuda"):
    import copy
    import torch
    from voxelgen.interfaces.render import (coords_to_cube_mesh, vertex_normals, render_mesh_batch,
                                              _pbr_env_shading, _render_to_image)
    mesh = coords_to_cube_mesh(b0, GRID, device=device)
    assert mesh is not None, "cannot build a mesh from empty voxels"
    mesh = vertex_normals(mesh)
    mtx = torch.stack([m for m, _ in cams])
    cam = torch.stack([o for _, o in cams]).view(len(cams), 1, 1, 3)
    mb = copy.copy(mesh)
    mb.v_pos = mesh.v_pos[None].expand(len(cams), -1, -1).contiguous()

    def _shade(rast, rast_db, m, **kw):
        m1 = copy.copy(m); m1.v_pos = mesh.v_pos
        kw["cam_pos"] = cam
        return _pbr_env_shading(rast, rast_db, m1, **kw)

    buf = render_mesh_batch(glctx, mb, mtx, 512, shading_func=_shade, antialias=True,
                            triangle_colors=None, metallic=0.05, roughness=0.4,
                            sky_top=None, sky_horizon=None, ground_color=None)
    assert buf["shaded"].shape[0] == len(cams), buf["shaded"].shape
    return [_to_png_array(_render_to_image({k: v[i:i + 1] for k, v in buf.items()}, "shaded"))
            for i in range(len(cams))]


def render_root(root, names, coords, up, rank=0, world_size=1, device="cuda"):
    import torch
    import nvdiffrast.torch as dr
    from PIL import Image
    from voxelgen.interfaces.render import render_voxel_frame
    _meta_check(root, up)
    cams = cameras(device)
    glctx = dr.RasterizeCudaContext()
    mine = list(range(len(names)))[rank::world_size]
    todo = [i for i in mine if not all(os.path.exists(p) for p in _views(root, names[i]))]
    print(f"[vrender] {root}  up={up}  shard {rank}/{world_size}  to render {len(todo)} / {len(mine)}", flush=True)
    t0 = time.time()
    for j, i in enumerate(todo):
        c = up_transform(np.asarray(coords[i]), up)
        b0 = torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), torch.as_tensor(c, dtype=torch.int32)], 1).to(device)
        od = os.path.join(root, RDIR, names[i]); os.makedirs(od, exist_ok=True)
        arrs = render_views(b0, glctx, cams, device)
        if j == 0:
            def _fd(m, o):
                return (render_voxel_frame(b0, glctx, m, GRID, o).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            for v, (m, o) in enumerate(cams):
                r1, r2 = _fd(m, o), _fd(m, o)
                n_self = int((r1 != r2).any(-1).sum()); n_new = int((r1 != arrs[v]).any(-1).sum())
                tot = r1.shape[0] * r1.shape[1]
                assert n_new <= 3 * max(n_self, 50), (
                    f"view {v}: batched render differs from the per-view renderer by {n_new} pixels, more than the renderer's own run-to-run difference ({n_self} pixels)")
                print(f"[vrender] view {v}: differs from the per-view renderer by {n_new} pixels; renderer run-to-run difference {n_self} pixels / {tot}", flush=True)
        for v, arr in enumerate(arrs):
            p = os.path.join(od, f"{v:03d}.png"); tmp = p + f".tmp{os.getpid()}.png"
            Image.fromarray(arr).save(tmp); os.replace(tmp, p)
        if (j + 1) % 100 == 0 or j + 1 == len(todo):
            print(f"  [{j + 1}/{len(todo)}] {(time.time() - t0) / (j + 1):.2f} s/sample", flush=True)


FEATS = {
    "dinov2": CACHE,
    "dinov3": "feats_vrender3.npz",
}


def root_names(root):
    return sorted(os.path.basename(x)[:-4] for x in glob.glob(os.path.join(root, "voxels", "*.vxz")))


def main():
    from . import contract as GC
    from .eval_voxels import load_root
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    GC.add_identity_args(ap, ref=False)
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world_size", type=int, default=1)
    a = ap.parse_args()
    a.root, _, _ = GC.resolve(a)
    a.up = GC.UP_AXIS
    print(f"[vrender] root {a.root}  up axis = {a.up}", flush=True)
    names, coords = load_root(a.root)
    render_root(a.root, names, coords, a.up, a.rank, a.world_size)
    print("[vrender] done")
