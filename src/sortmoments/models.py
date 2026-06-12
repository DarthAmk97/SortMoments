"""Model protocols and convenience face objects for Sort Moments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class DetectedFace:
    """Simple face object accepted by the Sort Moments pipeline.

    Custom face detectors can return this dataclass, InsightFace face objects,
    or any object with compatible ``bbox``, ``det_score``, and optional
    ``embedding`` attributes.
    """

    bbox: Sequence[float]
    det_score: float = 1.0
    embedding: Optional[Sequence[float]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        if self.embedding is not None:
            self.embedding = np.asarray(self.embedding, dtype=np.float32)


@runtime_checkable
class FaceModel(Protocol):
    """Detector protocol.

    Implement ``get(image_rgb)`` and return face objects with bbox/score and,
    if available, embeddings. A plain callable with the same signature also
    works when passed to ``organize_photos`` or ``detect_faces``.
    """

    def get(self, image_rgb: np.ndarray) -> Iterable[Any]:
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Embedding protocol for separating detection from embedding."""

    def embed(self, image_rgb: np.ndarray, face: Any) -> Sequence[float]:
        ...


@runtime_checkable
class GroupingModel(Protocol):
    """Grouping protocol.

    Implement ``group(face_records, similarity_threshold=...)`` and return a
    mapping of person/folder id to either face paths or FaceRecord objects.
    """

    def group(self, face_records: Sequence[Any], similarity_threshold: float = 0.5) -> Mapping[str, Iterable[Any]]:
        ...
