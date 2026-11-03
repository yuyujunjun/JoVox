import argparse
import csv
import glob
import hashlib
import json
import re
import os
import time

import numpy as np

from .contract import EVAL_GRID as GRID


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_root(root, limit=0, grid=GRID):
    import o_voxel
    from .contract import voxel_dir
    d = voxel_dir(root)
    fs = sorted(glob.glob(os.path.join(d, "*.vxz")))
    if limit:
        fs = fs[:limit]
    assert fs, f"no .vxz files in {d}"
    assert os.path.exists(os.path.join(root, "manifest.csv")), f"{root} has no manifest.csv"
    names, coords = [], []
    for f in fs:
        c = o_voxel.io.read_vxz(f)[0].int().numpy().astype(np.int64)
        assert len(c) > 0, f"{f}: empty occupancy"
        assert c.min() >= 0 and c.max() < grid, f"{f}: indices [{c.min()},{c.max()}] not in [0,{grid})"
        span = int((c.max(0) - c.min(0)).max())
        assert span == grid - 1, (
            f"{f}: longest-axis span={span}, expected {grid - 1} (sample is not normalized)")
        names.append(os.path.basename(f)[:-4]); coords.append(c)
    return names, coords


def run_paired(gen_names, gen, ref_names, ref, device, batch=32):
    from .paired import paired_batch
    ref_idx = {n: i for i, n in enumerate(ref_names)}
    missing = [n for n in gen_names if n not in ref_idx]
    assert not missing, (f"{len(missing)}/{len(gen_names)} generated samples have no matching uid in the reference set, first 3: "
                        f"{missing[:3]}")
    rows = []
    t0 = time.time()
    for s in range(0, len(gen_names), batch):
        idx = list(range(s, min(len(gen_names), s + batch)))
        out = paired_batch([gen[i] for i in idx], [ref[ref_idx[gen_names[i]]] for i in idx], device)
        for i, o in zip(idx, out):
            o["sample_id"] = gen_names[i]; rows.append(o)
        print(f"  [{len(rows)}/{len(gen_names)}] {time.time() - t0:.0f}s  ({(time.time() - t0) / len(rows):.2f} s/sample)", flush=True)
    keys = [k for k in rows[0] if k not in ("sample_id", "pose24")]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    for k in ("cd_raw", "cd_p24", "cd_refined"):
        agg[k + "_sum"] = 2 * agg[k]
    hist = {}
    for r in rows:
        hist[r["pose24"]] = hist.get(r["pose24"], 0) + 1
    agg["pose24_hist"] = dict(sorted(hist.items(), key=lambda kv: -kv[1]))
    agg["n_pairs"] = len(rows)
    return agg, rows


METRICS = ("core", "vis_pair")


def write_result(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


_SECTIONS = {"core": ("mode", "result", "ref_vs_ref", "frame_probe", "per_sample_csv"),
             "vis_pair": ("vis_pair", "vis_pair_csv")}
_SUBSET = re.compile(r"_n\d+\.(json|csv)$")


_NOT_RESULT = ("_smoke", "_p3dcheck", "_partial", "_anchor", "_superseded", "_frameprobe", "mesh")


def _has_metric(x, m):
    if not isinstance(x, dict) or x.get("mode") == "mesh":
        return False
    if m == "core":
        return isinstance(x.get("result"), dict)
    if m == "vis_pair":
        return isinstance(x.get("vis_pair"), dict) and "lpips" in x["vis_pair"]


def existing_results(root, ref_dataset, metrics, sfx=""):
    from .contract import eval_dir
    d = eval_dir(root, ref_dataset)
    found = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".json") or f.startswith(_NOT_RESULT):
            continue
        if bool(_SUBSET.search(f)) != bool(sfx) or (sfx and not f.endswith(sfx + ".json")):
            continue
        x = json.load(open(os.path.join(d, f)))
        for m in metrics:
            if _has_metric(x, m):
                found.append((m, f, x.get("time", "?"), str(x.get("eval_script_md5", "?"))[:8]))
    return found


def refuse_duplicate(found, root, ref_dataset):
    lines = "\n".join(f"  {m:8s} {f}   (time {t}, code md5 {h})" for m, f, t, h in found)
    raise SystemExit(f"[exists] {root} already has results for reference {ref_dataset}:\n{lines}\n"
                     f"  use --overwrite to recompute and replace them")


def main():
    from .contract import add_identity_args, resolve, eval_dir, cond_rows, UP_AXIS as up
    ap = argparse.ArgumentParser(description="Compare generated structures with the ground truth by CD_V, IoU, LPIPS_V and DINO_V.")
    add_identity_args(ap)
    ap.add_argument("--metrics", default=None,
                    help="core (CD_V, IoU), vis_pair (LPIPS_V, DINO_V), or both comma-separated (default)")
    ap.add_argument("--device", default="cuda", help="torch device")
    ap.add_argument("--batch", type=int, default=32, help="objects per batch for CD_V / IoU, changes only memory and speed")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute metrics that already exist (otherwise the script stops)")
    ap.add_argument("--compute_missing", action="store_true",
                    help="compute the ground-truth DINOv2 features for DINO_V if they are not cached yet (needed on the first run)")
    a = ap.parse_args()
    subset_n = a.n if (a.n and a.method == "gt") else None
    if subset_n:
        a.n = None
    sfx = f"_n{subset_n}" if subset_n else ""
    root, ref_ds, gt = resolve(a)
    a.gen, a.ref = root, gt
    outdir = eval_dir(root, ref_ds)

    metrics = list(METRICS) if a.metrics is None else [m.strip() for m in a.metrics.split(",") if m.strip()]
    for m in metrics:
        assert m in METRICS, f"--metrics must be from {list(METRICS)}, got {m!r}"
    if subset_n:
        assert set(metrics) <= {"core", "vis_pair"}, f"a subset of {subset_n} samples only supports core / vis_pair, got {metrics}"
    dup = existing_results(root, ref_ds, metrics, sfx)
    if dup and not a.overwrite:
        refuse_duplicate(dup, root, ref_ds)
    if dup:
        print(f"[exists] --overwrite: recomputing {sorted({m for m, *_ in dup})}", flush=True)
    print(f"[cfg] root {root}\n[cfg] reference {gt}  results -> {outdir}/<metric>.json\n"
          f"[cfg] paired with {ref_ds}  up axis {up}  metrics {metrics}"
          f"  missing features: {'compute' if a.compute_missing else 'error'}", flush=True)

    t0 = time.time()
    gen_names, gen = load_root(a.gen)
    ref_names, ref = load_root(a.ref)
    if subset_n:
        keep = {r["uid"] for r in cond_rows(ref_ds)[:subset_n]}
        gen_names, gen = map(list, zip(*[(n, c) for n, c in zip(gen_names, gen) if n in keep])) if keep & set(gen_names) else ([], [])
        ref_names, ref = map(list, zip(*[(n, c) for n, c in zip(ref_names, ref) if n in keep]))
        print(f"[subset] first {subset_n} entries of the list: gen {len(gen)} / ref {len(ref)} (result suffix {sfx})", flush=True)
    print(f"[load] gen {len(gen)} <- {a.gen}\n[load] ref {len(ref)} <- {a.ref}", flush=True)

    degen_path = os.path.join(a.gen, "degenerate.csv")
    degen_rows = list(csv.DictReader(open(degen_path))) if os.path.exists(degen_path) else []
    if subset_n:
        degen_rows = [r for r in degen_rows if r["sample_id"] in keep]
    degen = {"n_degenerate": len(degen_rows), "n_valid": len(gen_names),
             "rate": len(degen_rows) / max(1, len(gen_names) + len(degen_rows)), "sample_ids": [r["sample_id"] for r in degen_rows]}
    if degen_rows:
        print(f"[degenerate] rate {degen['rate']:.4f}", flush=True)
    common = {
        "degenerate": degen,
        "gen_root": a.gen, "ref_root": a.ref,
        "gen_manifest_sha256": _sha256(os.path.join(a.gen, "manifest.csv")),
        "ref_manifest_sha256": _sha256(os.path.join(a.ref, "manifest.csv")),
        "gen_names_first3": gen_names[:3], "ref_names_first3": ref_names[:3],
        "identity": {"dataset": a.dataset, "method": a.method, "step": a.step,
                     "n": subset_n or a.n, "subset_of_full_root": bool(subset_n), "ref": ref_ds},
        "metrics": metrics, "up": up, "compute_missing": a.compute_missing, "overwrite": a.overwrite,
        "eval_script_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    count = {"n_gen_valid": len(gen_names), "n_degenerate": len(degen_rows), "n_ref": len(ref_names),
             "match": len(gen_names) + len(degen_rows) == len(ref_names)}
    if "core" not in metrics:
        print(f"[metrics] computing {metrics} only (core skipped)", flush=True)
        out = {"mode": "paired", "result": "skipped (core not in --metrics)", "count": count, **common}
    else:
        agg, rows = run_paired(gen_names, gen, ref_names, ref, a.device, a.batch)
        from . import paired as _pg
        lv = _pg.LEVELS
        print(f"[result] CD_V raw {agg['cd_raw']:.5f}  p24 {agg['cd_p24']:.5f}  refined {agg['cd_refined']:.5f}"
              f"  (mean of both directions)   " + " ".join(f"IoU{L} {agg[f'iou{L}']:.4f}" for L in lv)
              + f"   N {agg['n_pairs']}  {time.time() - t0:.0f}s", flush=True)
        print(f"[result] after scaled ICP: CD_V {agg['cd_refined']:.5f}  "
              + " ".join(f"IoU{L} {agg[f'iou{L}_refined']:.4f}" for L in lv)
              + f"  mean scale {agg['scale_refined']:.4f}", flush=True)
        print(f"[result] pose histogram: {agg['pose24_hist']}", flush=True)
        out = {"mode": "paired", "result": agg, **common}
        csv_path = os.path.join(outdir, f"core{sfx}.csv")
        os.makedirs(outdir, exist_ok=True)
        with open(csv_path + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        os.replace(csv_path + ".tmp", csv_path)
        out["per_sample_csv"] = csv_path


    if "vis_pair" in metrics:
        from .vis_pair import run_vis_pair
        t1 = time.time()
        vp, vrows = run_vis_pair(gen_names, gen, ref_names, ref, a.ref, up, a.device,
                                  compute_missing=a.compute_missing)
        vcsv = os.path.join(outdir, f"vis_pair{sfx}.csv")
        os.makedirs(outdir, exist_ok=True)
        with open(vcsv + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(vrows[0].keys())); w.writeheader(); w.writerows(vrows)
        os.replace(vcsv + ".tmp", vcsv)
        out["vis_pair"], out["vis_pair_csv"] = vp, vcsv
        print(f"[result] vis_pair LPIPS {vp['lpips']:.4f}  DINOv2 cosine {vp['dino_cos']:.4f}   "
              f"(N {vp['n_pairs']} x {vp['n_views']} views, {time.time() - t1:.0f}s)", flush=True)

    if "core" not in metrics and not count["match"] and not subset_n:
        msg = (f"generated {count['n_gen_valid']} + degenerate {count['n_degenerate']} != reference {count['n_ref']}")
        out["count_warning"] = msg
        print(f"[count mismatch] {msg}", flush=True)
    owned = {k for m in METRICS for k in _SECTIONS[m] if k != "mode"}
    base = {k: v for k, v in out.items() if k not in owned}
    written = []
    for m in metrics:
        p = os.path.join(outdir, f"{m}{sfx}.json")
        write_result(p, {**base, **{k: out[k] for k in _SECTIONS[m] if k in out}})
        written.append(os.path.basename(p))
    print(f"[done] -> {outdir}/{{{','.join(written)}}}", flush=True)
