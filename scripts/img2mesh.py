import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelgen.generate import img2mesh


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Image -> 64^3 voxel structure -> mesh (TRELLIS.2 second stage).")
    ap.add_argument("--stage", required=True, choices=["prep", "voxel", "mesh"],
                    help="prep: remove the background and center the object; voxel: generate the structure; "
                         "mesh: turn the structure into a mesh")
    ap.add_argument("--in_dir", help="prep: folder of input images (.png, .jpg, .jpeg, .webp)")
    ap.add_argument("--out", required=True, help="output folder; each stage reads the results of the previous one from it")
    ap.add_argument("--seed", type=int, default=0, help="voxel and mesh: random seed")
    ap.add_argument("--fill", type=float, default=0.74,
                    help="prep: fraction of the image covered by the longest side of the object")
    ap.add_argument("--spec", default="voxelgen64",
                    help="voxel: voxelgen64 (the released model) or voxelgen64_trained (a checkpoint of your own training)")
    ap.add_argument("--step", type=int, default=None, help="voxel: checkpoint step, only with voxelgen64_trained")
    ap.add_argument("--batch", type=int, default=1, help="voxel: images per sampling call")
    a = ap.parse_args()
    if a.stage == "prep":
        assert a.in_dir, "--stage prep requires --in_dir"
        img2mesh.prep(a.in_dir, a.out, a.fill)
    elif a.stage == "voxel":
        img2mesh.voxel(a.out, a.spec, a.step, a.seed, a.batch)
    else:
        img2mesh.mesh(a.out, a.seed)
    print(f"=== IMG2MESH {a.stage} DONE", flush=True)
