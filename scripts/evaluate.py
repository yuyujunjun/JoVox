import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxelgen.evaluate.eval_voxels import main


if __name__ == "__main__":
    main()
