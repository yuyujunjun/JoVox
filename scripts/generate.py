import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelgen.generate.batch import generate_list
from voxelgen.generate.generator import SPECS


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate 64^3 voxel structures for every image of an evaluation list.")
    ap.add_argument("--spec", required=True, choices=sorted(SPECS))
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (voxelgen64_trained only)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=None, help="generate only the first N entries")
    ap.add_argument("--batch", type=int, default=8, help="images per batch")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    a = ap.parse_args()
    generate_list(a.spec, a.step, a.dataset, a.n, a.batch, a.rank, a.world_size)
