import csv
import os
import time

import numpy as np
import torch

from voxelgen.evaluate import contract as GC
from voxelgen.train.dataset import load_rgba_image
from .generator import VoxelGenerator, get_spec


def generate_list(spec_name: str, step, dataset: str, n=None, batch: int = 8, rank: int = 0, world_size: int = 1):
    spec = get_spec(spec_name, step)
    out_root = GC.root_of(dataset, spec_name, step=step, n=n)
    degen_path = os.path.join(out_root, "degenerate.csv")
    degen = {r["sample_id"] for r in csv.DictReader(open(degen_path))} if os.path.exists(degen_path) else set()

    rows = GC.cond_rows(dataset)
    for r in rows[:3]:
        assert os.path.exists(r["image_path"]), f"condition image not found: {r['image_path']}"
    if n:
        rows = rows[:n]
    rows = rows[rank::world_size]
    rows = [r for r in rows if not (r["uid"] in degen or GC.already_done(out_root, r["uid"]))]
    print(f"[cfg] {spec_name}{f' @ {step}' if step else ''}  dataset {dataset}  todo {len(rows)} "
          f"batch={batch} shard={rank}/{world_size}\n[cfg] root {out_root}", flush=True)

    gen = VoxelGenerator(spec, torch.device("cuda"))
    done, n_occ = 0, 0
    t0 = time.time()
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        seed = int(chunk[0]["seed"])
        ims = [load_rgba_image(r["image_path"]) for r in chunk]
        outs = gen.generate(torch.stack([i for i, _ in ims]), torch.stack([m for _, m in ims]), seed)
        for r, out in zip(chunk, outs):
            if len(np.asarray(out)) == 0:
                os.makedirs(out_root, exist_ok=True)
                with open(os.path.join(out_root, "degenerate.csv"), "a") as f:
                    if f.tell() == 0:
                        f.write("sample_id,reason,n_occ_raw\n")
                    f.write(f"{r['uid']},empty_output,0\n")
                print(f"  [degenerate] {r['uid'][:12]} empty output, recorded", flush=True)
                done += 1
                continue
            n_occ = GC.save_sample(out_root, r["uid"], out, gen.native_res, seed=seed)
            done += 1
        if done <= batch or done % 100 < batch:
            print(f"  [{done}/{len(rows)}] {r['uid'][:12]} -> 64^3 {n_occ}  ({(time.time() - t0) / done:.1f} s/sample)", flush=True)
    print(f"[done] wrote {done}, {time.time() - t0:.0f}s", flush=True)
