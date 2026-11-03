import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelgen.train.trainer import Trainer


if __name__ == '__main__':
    with Trainer() as trainer:
        trainer.run()
