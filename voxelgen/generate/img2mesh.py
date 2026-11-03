import glob
import os

import numpy as np

EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _names(d):
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))
    assert fs, f"no images ({EXTS}) in {d}"
    return fs


def prep(in_dir: str, out: str, fill: float):
    from PIL import Image
    from voxelgen.interfaces.trellis2 import background_remover
    rembg = background_remover()
    os.makedirs(f"{out}/prep", exist_ok=True)
    for f in _names(in_dir):
        img = Image.open(os.path.join(in_dir, f))
        has_alpha = img.mode == "RGBA" and not np.all(np.array(img)[:, :, 3] == 255)
        s = min(1, 1024 / max(img.size))
        if s < 1:
            img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
        im = img if has_alpha else rembg(img.convert("RGB"))
        assert im.mode == "RGBA" and (np.array(im)[:, :, 3] > 204).any(), f"{f}: no foreground after background removal ({im.mode})"
        al = np.array(im)[:, :, 3]
        ys, xs = np.nonzero(al > 204)
        obj = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        side = int(np.ceil(max(obj.size) / fill))
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(obj, ((side - obj.width) // 2, (side - obj.height) // 2))
        im = canvas.resize((512, 512), Image.Resampling.LANCZOS)
        im.save(f"{out}/prep/{os.path.splitext(f)[0]}.png")
        print(f"[prep] {f} -> prep/{os.path.splitext(f)[0]}.png {im.size} {'alpha kept' if has_alpha else 'background removed'};"
              f" foreground box {obj.size} centered on a {side}^2 canvas (object fraction {max(obj.size) / side:.2f}, fill={fill}), resized to 512^2",
              flush=True)


def voxel(out: str, spec_name: str, step, seed: int, batch: int):
    import torch
    from voxelgen.train.dataset import load_rgba_image
    from .generator import VoxelGenerator, get_spec
    spec = get_spec(spec_name, step)
    gen = VoxelGenerator(spec, torch.device("cuda"))
    print(f"[voxel] spec={spec_name} checkpoint={spec.ckpt_file()} weights={spec.weights} sampling={gen.sampling_kwargs} seed={seed}",
          flush=True)
    os.makedirs(f"{out}/voxels", exist_ok=True)
    R = gen.native_res
    pngs = sorted(glob.glob(f"{out}/prep/*.png"))
    assert pngs, f"no images in {out}/prep/: run the prep stage first"
    print(f"[voxel] batch size {batch}", flush=True)
    for i0 in range(0, len(pngs), batch):
        chunk = pngs[i0:i0 + batch]
        ims = [load_rgba_image(p) for p in chunk]
        assert len({tuple(r.shape) for r, _ in ims}) == 1, f"images in one batch have different sizes: {[tuple(r.shape) for r, _ in ims]}"
        res = gen.generate(torch.stack([r for r, _ in ims]), torch.stack([m for _, m in ims]), seed)
        for pth, o in zip(chunk, res):
            c = np.asarray(o).astype(np.int64)
            assert c.ndim == 2 and c.shape[1] == 3 and (len(c) == 0 or (c.min() >= 0 and c.max() < R)), c.shape
            n = os.path.splitext(os.path.basename(pth))[0]
            np.save(f"{out}/voxels/{n}.npy", c)
            print(f"[voxel] {n}: {len(c)} occupied cells at {R}^3", flush=True)


def mesh(out: str, seed: int):
    import trimesh
    from PIL import Image
    from voxelgen.interfaces.trellis2 import mesh_pipeline, voxels_to_mesh
    p = mesh_pipeline()
    os.makedirs(f"{out}/mesh", exist_ok=True)
    vs = sorted(glob.glob(f"{out}/voxels/*.npy"))
    assert vs, f"no voxels in {out}/voxels/: run the voxel stage first"
    for vp in vs:
        n = os.path.splitext(os.path.basename(vp))[0]
        c = np.load(vp)
        if len(c) == 0:
            print(f"[mesh] {n}: empty voxels, skipped", flush=True)
            continue
        v, f = voxels_to_mesh(p, Image.open(f"{out}/prep/{n}.png"), seed, c)
        trimesh.Trimesh(v, f, process=False).export(f"{out}/mesh/{n}.ply", file_type="ply")
        print(f"[mesh] {n}: {len(v)} vertices {len(f)} faces -> mesh/{n}.ply", flush=True)
