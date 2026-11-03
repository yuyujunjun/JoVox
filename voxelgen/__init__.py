import os
import sys

TRELLIS2_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "TRELLIS.2")
assert os.path.isdir(os.path.join(TRELLIS2_ROOT, "trellis2")), (
    f"TRELLIS.2 not found at {TRELLIS2_ROOT}: clone https://github.com/microsoft/TRELLIS.2 there "
    f"and check out commit 5565d24 (see README)")
if TRELLIS2_ROOT not in sys.path:
    sys.path.insert(0, TRELLIS2_ROOT)
