import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelgen.evaluate.contract import voxelize_gt, DATASETS


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Voxelize the ground-truth meshes to 64^3 (written to data/gen/<dataset>/gt/).")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS), help="name of the evaluation set")
    ap.add_argument("--src", required=True, help="folder of ground-truth meshes named <uid>.<ext>")
    ap.add_argument("--pattern", default="*.glb", help="file pattern of the meshes in --src")
    a = ap.parse_args()
    voxelize_gt(a.dataset, a.src, a.pattern)
