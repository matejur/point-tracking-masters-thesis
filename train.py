from tracker.training import train, parse_args
from tracker.utils import path_to_mast3r  # noqa: F401

from mast3r.utils import path_to_dust3r  # noqa: F401
from tracker.datasets import KubricSeq

import dust3r.datasets

dust3r.datasets.KubricSeq = KubricSeq

if __name__ == "__main__":
    args = parse_args()
    train(args)

