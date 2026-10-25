"""Matched CPU control using the historical recorded q/command convention.

Accepts the same arguments as r2p.policy.prove; use a separate --out directory.
Only the two P2 dataset convention markers are removed. The model, optimizer,
augmentation, evaluator and success checker are identical to the repaired fit.
An exit status indicating a failed proof is expected if this control scores <=50%.
"""
from __future__ import annotations

import importlib

from . import prove


def main():
    training = importlib.import_module("r2p.policy.train")
    original = training.export_config

    def recorded_config(*args, **kwargs):
        config = original(*args, **kwargs)
        config.pop("commanded_q_source")
        config.pop("command_alignment")
        return config

    training.export_config = recorded_config
    try:
        prove.main()
    finally:
        training.export_config = original


if __name__ == "__main__":
    main()
