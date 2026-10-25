"""Calibrated, task-driven perception. Images use OpenCV's BGR convention."""

from .camera import CameraModel, HttpCamera, LiveCamera, StillCamera, SynthCamera, open_camera
from .task_perception import TaskPerceiver

__all__ = ["CameraModel", "HttpCamera", "LiveCamera", "StillCamera", "SynthCamera", "open_camera", "TaskPerceiver"]
