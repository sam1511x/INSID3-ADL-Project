"""Task 2 - Multi-Prototype Reference Matching.

The annotated reference region is represented by M prototypes (clusters of its
debiased DINOv3 patch features) instead of a single mean prototype; candidate
target regions are matched against the whole set. Compares aggregation
strategies (max / weighted mean / softmax / many-to-many OT) and numbers of
prototypes with the original INSID3 matching. Same arguments as
inference_segmentation.py plus --variants / --max-episodes.

Examples:
    python experiments/task2_multiproto.py --dataset coco --fold 0 --exp-name t2-coco-f0
    python experiments/task2_multiproto.py --dataset pascal_part --variants "single (INSID3)" "M=4 max"
    python experiments/task2_multiproto.py --list-variants
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import main  # noqa: E402
from models.insid3_ext import DebiasConfig as D, ProtoConfig as P  # noqa: E402

REGISTRY = {"single (INSID3)": (D(), P(num_protos=1))}
# number of prototypes x aggregation strategy
for M in (2, 3, 4, 6, 8):
    for agg in ("max", "mean", "softmax", "ot"):
        REGISTRY[f"M={M} {agg}"] = (D(), P(num_protos=M, agg=agg))
# ablations on the best-performing structure
for M in (4,):
    REGISTRY[f"M={M} max +global"] = (D(), P(num_protos=M, agg="max", include_global=True))
    REGISTRY[f"M={M} softmax +global"] = (D(), P(num_protos=M, agg="softmax", include_global=True))
    REGISTRY[f"M={M} max agglo"] = (D(), P(num_protos=M, agg="max", cluster="agglo"))
    REGISTRY[f"M={M} max seed+agg only"] = (D(), P(num_protos=M, agg="max", stages="seed,agg"))
    REGISTRY[f"M={M} max agg only"] = (D(), P(num_protos=M, agg="max", stages="agg"))

DEFAULT = list(REGISTRY)

if __name__ == "__main__":
    main("INSID3 Task 2: multi-prototype reference matching", REGISTRY, DEFAULT)
