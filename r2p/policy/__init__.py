"""Command-conditioned absolute-target policies. Inference requires only NumPy."""
from pathlib import Path

from .runtime import Policy


def load(name_or_dir):
    path = Path(name_or_dir)
    if not path.is_dir():
        path = Path(__file__).resolve().parents[2] / "exports" / str(name_or_dir)
    return Policy.load(path)


__all__ = ["Policy", "load"]
