"""Deterministic CPU synthetic bench; never opens a GPU rendering context."""
from .world import World, OracleDetector
from .composite import CompositeRenderer, RenderParams, compare_frames

__all__ = ["World", "OracleDetector", "CompositeRenderer", "RenderParams", "compare_frames"]
