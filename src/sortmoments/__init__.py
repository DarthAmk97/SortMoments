"""Sort Moments Python library API.

This package wraps the existing Sort Moments processing pipeline so it can be
used from Python code without launching the desktop UI.
"""

from .api import (
    FaceRecord,
    OrganizationResult,
    ProcessingResult,
    SortMomentsConfig,
    SortMomentsOrganizer,
    detect_faces,
    group_faces,
    organize_photos,
)
from .models import DetectedFace, EmbeddingModel, FaceModel, GroupingModel

__all__ = [
    "DetectedFace",
    "EmbeddingModel",
    "FaceModel",
    "FaceRecord",
    "GroupingModel",
    "OrganizationResult",
    "ProcessingResult",
    "SortMomentsConfig",
    "SortMomentsOrganizer",
    "detect_faces",
    "group_faces",
    "organize_photos",
]
