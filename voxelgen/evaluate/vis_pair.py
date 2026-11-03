import json
import os
import time

import numpy as np


GT_DINO = "dino_renders6_vox.npz"


def run_vis_pair(gen_names, gen, ref_names, ref, ref_root, up, device="cuda", batch=16, compute_missing=False):
    import nvdiffrast.torch as dr
    import torch
    from PIL import Image
    from .paired import align_to_ref
    from .vrender import cameras, render_views, up_transform, RDIR, RENDERER, GRID, N_VIEWS
    from .perceptual import lpips_pairs, dino_cls_images, dino_cls

    meta = os.path.join(ref_root, RDIR, "_meta.json")
    want = {"up": up, "renderer": RENDERER, "grid": GRID, "n_views": N_VIEWS}
    assert os.path.exists(meta) and json.load(open(meta)) == want, \
        f"reference renders {meta} do not match the expected settings {want}"
    ref_idx = {n: i for i, n in enumerate(ref_names)}
    missing = [n for n in gen_names if n not in ref_idx]
    assert not missing, f"{len(missing)} generated samples have no matching uid in the reference set, first 3: {missing[:3]}"
    gpath = os.path.join(ref_root, GT_DINO)
    gt_dino = dict(np.load(gpath)) if os.path.exists(gpath) else {}
    need = [n for n in gen_names if n not in gt_dino]
    if need:
        if not compute_missing:
            raise SystemExit(f"[cache] {len(need)} reference DINOv2 features missing ({gpath}); use --compute_missing")
        views = [os.path.join(ref_root, RDIR, n, f"{v:03d}.png") for n in need for v in range(N_VIEWS)]
        miss = [p for p in views if not os.path.exists(p)]
        assert not miss, f"{len(miss)} reference renders missing, first 3: {miss[:3]}"
        feats = dino_cls(views, device).reshape(len(need), N_VIEWS, -1)
        gt_dino.update({n: f for n, f in zip(need, feats)})
        np.savez(gpath[:-4] + ".tmp.npz", **gt_dino); os.replace(gpath[:-4] + ".tmp.npz", gpath)
        print(f"[cache] computed {len(need)} reference DINOv2 features -> {gpath}", flush=True)
    cams = cameras(device)
    glctx = dr.RasterizeCudaContext()
    rows, t0 = [], time.time()
    for s in range(0, len(gen_names), batch):
        names = gen_names[s:s + batch]
        al = align_to_ref([gen[i] for i in range(s, s + len(names))], [ref[ref_idx[n]] for n in names], device)
        g_imgs, r_imgs = [], []
        for j, n in enumerate(names):
            c = up_transform(np.asarray(al["idx_refined"][j]), up)
            b0 = torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), torch.as_tensor(c, dtype=torch.int32)], 1).to(device)
            g_imgs += [Image.fromarray(x) for x in render_views(b0, glctx, cams, device)]
            for v in range(N_VIEWS):
                p = os.path.join(ref_root, RDIR, n, f"{v:03d}.png")
                assert os.path.exists(p), f"reference render missing: {p}"
                r_imgs.append(Image.open(p).convert("RGB"))
        lp = lpips_pairs(g_imgs, r_imgs, device).reshape(len(names), N_VIEWS)
        fg = dino_cls_images(g_imgs, device)
        fr = np.concatenate([gt_dino[n] for n in names], 0).astype(np.float32)
        fg /= np.linalg.norm(fg, axis=1, keepdims=True); fr /= np.linalg.norm(fr, axis=1, keepdims=True)
        cos = (fg * fr).sum(1).reshape(len(names), N_VIEWS)
        for j, n in enumerate(names):
            rows.append({"sample_id": n, "lpips": float(lp[j].mean()), "dino_cos": float(cos[j].mean()),
                         "pose24": al["pose24"][j], "scale_refined": float(al["s"][j])})
        print(f"  [vis_pair {len(rows)}/{len(gen_names)}] {(time.time() - t0) / len(rows):.2f} s/sample", flush=True)
    res = {"lpips": float(np.mean([r["lpips"] for r in rows])), "dino_cos": float(np.mean([r["dino_cos"] for r in rows])),
           "n_pairs": len(rows), "n_views": N_VIEWS, "renderer": RENDERER, "up": up,
           "definition": "per pair: gen voxels aligned to GT (24-pose coarse + scaled ICP, paired_gpu.align_to_ref), "
                         "rounded to the grid, rendered with the FID_VR renderer/6 cameras as the GT renders6_vox; "
                         "per view LPIPS(alex) and dinov2_vitl14_reg CLS cosine; mean over views, then over pairs"}
    return res, rows
