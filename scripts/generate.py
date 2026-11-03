import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelgen.evaluate.contract import DATASETS
from voxelgen.generate.batch import generate_list
from voxelgen.generate.generator import SPECS


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate one 64^3 structure for every object of data/splits/eval_cond.csv.")
    ap.add_argument("--spec", required=True, choices=sorted(SPECS),
                    help="voxelgen64: the released model; voxelgen64_trained: a checkpoint of your own training")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step, only with voxelgen64_trained")
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                    help="name of the evaluation set; results go to data/gen/<dataset>/<spec>/")
    ap.add_argument("--n", type=int, default=None, help="use only the first N objects of the list")
    ap.add_argument("--batch", type=int, default=8, help="images per sampling call")
    ap.add_argument("--rank", type=int, default=0, help="index of this job when the list is split over several jobs")
    ap.add_argument("--world_size", type=int, default=1, help="number of jobs the list is split over")
    a = ap.parse_args()
    generate_list(a.spec, a.step, a.dataset, a.n, a.batch, a.rank, a.world_size)
