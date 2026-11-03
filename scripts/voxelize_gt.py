import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelgen.evaluate.contract import voxelize_gt, DATASETS


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ground-truth meshes (<uid>.glb) -> 64^3 evaluation root of the dataset.")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--src", required=True, help="directory of ground-truth meshes named <uid>.<ext>")
    ap.add_argument("--pattern", default="*.glb")
    a = ap.parse_args()
    voxelize_gt(a.dataset, a.src, a.pattern)
