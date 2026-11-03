import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelgen.generate import img2mesh


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Image -> 64^3 voxel structure -> mesh (TRELLIS.2 second stage).")
    ap.add_argument("--stage", required=True, choices=["prep", "voxel", "mesh"])
    ap.add_argument("--in_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill", type=float, default=0.74,
                    help="prep: fraction of the square canvas covered by the longest side of the object")
    ap.add_argument("--spec", default="voxelgen64")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1, help="voxel: number of images per sampling call")
    a = ap.parse_args()
    if a.stage == "prep":
        assert a.in_dir, "--stage prep requires --in_dir"
        img2mesh.prep(a.in_dir, a.out, a.fill)
    elif a.stage == "voxel":
        img2mesh.voxel(a.out, a.spec, a.step, a.seed, a.batch)
    else:
        img2mesh.mesh(a.out, a.seed)
    print(f"=== IMG2MESH {a.stage} DONE", flush=True)
