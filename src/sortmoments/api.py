"""Package-facing library facade for Sort Moments processing."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
import os
import pickle
import shutil

import numpy as np

from . import pipeline as processphotos


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class SortMomentsConfig:
    """Configuration for the library facade.

    Defaults match the desktop app's current processing settings. Pass custom
    models to ``organize_photos``/``detect_faces`` to replace InsightFace or the
    default cosine-similarity grouping without changing this config object.
    """

    min_face_size: int = 80
    min_confidence: float = 0.8
    min_face_ratio: float = 0.01
    foreground_ratio_threshold: float = 0.1
    blur_threshold: float = 60
    batch_size: int = 8
    max_workers: int = 4
    similarity_threshold: float = 0.5
    prefer_gpu: bool = True
    model_name: str = "buffalo_l"
    det_size: tuple[int, int] = (640, 640)
    keep_intermediate: bool = False
    session_id: Optional[str] = None
    intermediate_folder_name: str = "face_detection_output"
    final_output_folder: Optional[str] = None


@dataclass(frozen=True)
class FaceRecord:
    """A detected face plus source-photo metadata for custom grouping models."""

    face_path: str
    source_path: str
    embedding: np.ndarray
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ProcessingResult:
    """Result returned by ``organize_photos``."""

    persons: Mapping[str, Sequence[str]]
    output_folder: Path
    embeddings_file: Path
    person_count: int
    face_count: int


# More descriptive alias for documentation; kept alongside ProcessingResult.
OrganizationResult = ProcessingResult


def _config(config: Optional[SortMomentsConfig]) -> SortMomentsConfig:
    return config if config is not None else SortMomentsConfig()


class SortMomentsOrganizer:
    """Reusable organizer with configured model hooks.

    Use this when you want to run multiple folders with the same thresholds,
    progress callback, custom detector/embedder, or grouping strategy. The
    convenience functions below use the same pipeline for one-off calls.
    """

    def __init__(
        self,
        config: SortMomentsConfig | None = None,
        *,
        face_model: Any = None,
        embedding_model: Any = None,
        grouping_model: Any = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = _config(config)
        self.face_model = face_model
        self.embedding_model = embedding_model
        self.grouping_model = grouping_model
        self.progress_callback = progress_callback

    def detect_faces(
        self,
        input_folder: str | os.PathLike[str],
        output_folder: str | os.PathLike[str] | None = None,
    ) -> Mapping[str, Any]:
        return detect_faces(
            input_folder,
            output_folder,
            config=self.config,
            face_model=self.face_model,
            embedding_model=self.embedding_model,
            progress_callback=self.progress_callback,
        )

    def group_faces(
        self,
        intermediate_folder: str | os.PathLike[str],
        *,
        input_folder: str | os.PathLike[str] | None = None,
    ) -> tuple[Mapping[str, Sequence[str]], Path]:
        return group_faces(
            intermediate_folder,
            input_folder=input_folder,
            config=self.config,
            grouping_model=self.grouping_model,
        )

    def organize(
        self,
        input_folder: str | os.PathLike[str],
        *,
        work_folder: str | os.PathLike[str] | None = None,
    ) -> "ProcessingResult":
        return organize_photos(
            input_folder,
            work_folder=work_folder,
            config=self.config,
            face_model=self.face_model,
            embedding_model=self.embedding_model,
            grouping_model=self.grouping_model,
            progress_callback=self.progress_callback,
        )


def _default_intermediate_folder(input_folder: Path, config: SortMomentsConfig) -> Path:
    return input_folder / config.intermediate_folder_name


def _load_face_records(intermediate_folder: Path) -> list[FaceRecord]:
    embeddings_file = intermediate_folder / "face_embeddings_insightface.pkl"
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Face embeddings file not found: {embeddings_file}")

    with embeddings_file.open("rb") as fh:
        embeddings_dict = pickle.load(fh)

    records: list[FaceRecord] = []
    for face_path, value in embeddings_dict.items():
        if isinstance(value, dict) and "embedding" in value:
            embedding = value["embedding"]
            source_path = value.get("source_path", "")
            metadata = {k: v for k, v in value.items() if k not in {"embedding", "source_path"}}
        else:
            embedding = value
            source_path = _legacy_original_for_face(face_path)
            metadata = {}

        if embedding is None:
            raise ValueError(f"Missing embedding for detected face: {face_path}")

        records.append(
            FaceRecord(
                face_path=str(face_path),
                source_path=str(source_path),
                embedding=np.asarray(embedding),
                metadata=metadata,
            )
        )
    return records


def _legacy_original_for_face(face_path: str) -> str:
    image_folder = Path(face_path).parent
    matches = list(image_folder.glob("original_*"))
    return str(matches[0]) if matches else ""


def _call_grouping_model(grouping_model: Any, records: Sequence[FaceRecord], similarity_threshold: float) -> Mapping[str, Iterable[Any]]:
    if hasattr(grouping_model, "group"):
        try:
            return grouping_model.group(records, similarity_threshold=similarity_threshold)
        except TypeError:
            return grouping_model.group(records)
    if callable(grouping_model):
        try:
            return grouping_model(records, similarity_threshold=similarity_threshold)
        except TypeError:
            return grouping_model(records)
    raise TypeError("grouping_model must be callable or expose a .group(...) method")


def _normalize_groups(groups: Mapping[str, Iterable[Any]]) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for person_id, members in groups.items():
        face_paths: list[str] = []
        for member in members:
            if isinstance(member, FaceRecord):
                face_paths.append(member.face_path)
            elif hasattr(member, "face_path"):
                face_paths.append(str(member.face_path))
            else:
                face_paths.append(str(member))
        normalized[str(person_id)] = face_paths
    return normalized


def _copy_unique(src: str, destination_dir: Path) -> Optional[Path]:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None

    destination = destination_dir / src_path.name
    if destination.exists() and destination.resolve() != src_path.resolve():
        stem, suffix = src_path.stem, src_path.suffix
        counter = 2
        while destination.exists():
            destination = destination_dir / f"{stem}_{counter}{suffix}"
            counter += 1

    shutil.copy2(src_path, destination)
    return destination


def _organize_custom_groups(
    *,
    groups: Mapping[str, Iterable[Any]],
    records: Sequence[FaceRecord],
    input_folder: Path,
    config: SortMomentsConfig,
) -> tuple[dict[str, list[str]], Path]:
    normalized_groups = _normalize_groups(groups)
    record_by_face_path = {record.face_path: record for record in records}

    if config.final_output_folder:
        base_output = Path(config.final_output_folder)
        processed_folder = base_output / config.session_id if config.session_id else base_output
    elif config.session_id:
        processed_folder = input_folder / "all_images_processed" / config.session_id
    else:
        processed_folder = input_folder / "all_images_processed"
    processed_folder.mkdir(parents=True, exist_ok=True)

    group_photos_folder = processed_folder / "all_group_photos"
    group_photos_folder.mkdir(parents=True, exist_ok=True)

    image_face_counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.source_path:
            image_face_counts[record.source_path] += 1

    persons_to_originals: dict[str, list[str]] = {}
    for person_id, face_paths in normalized_groups.items():
        person_folder = processed_folder / person_id
        person_folder.mkdir(parents=True, exist_ok=True)

        originals: set[str] = set()
        existing_faces = [face_path for face_path in face_paths if face_path in record_by_face_path]
        for face_path in existing_faces:
            source_path = record_by_face_path[face_path].source_path
            if source_path:
                originals.add(source_path)

        persons_to_originals[person_id] = sorted(originals)

        if existing_faces:
            try:
                representative = max(existing_faces, key=lambda path: Path(path).stat().st_size)
                shutil.copy2(representative, person_folder / f"{person_id}_representative_face.jpg")
            except OSError:
                pass

        for original_path in sorted(originals):
            _copy_unique(original_path, person_folder)

    for original_path, count in image_face_counts.items():
        if count > 3:
            _copy_unique(original_path, group_photos_folder)

    with (group_photos_folder / "README.txt").open("w", encoding="utf-8") as fh:
        fh.write("GROUP PHOTOS\n=============\n\n")
        fh.write("This folder contains photos where more than 3 faces were detected.\n")

    with (processed_folder / "persons_to_originals.pkl").open("wb") as fh:
        pickle.dump(persons_to_originals, fh)

    with (processed_folder / "README.txt").open("w", encoding="utf-8") as fh:
        fh.write("PHOTO ORGANIZATION BY PERSON\n")
        fh.write("===========================\n\n")
        fh.write("Each folder contains a representative face and source photos for one grouped person.\n")
        fh.write("Folders are named by the active grouping model and can be renamed.\n")

    return persons_to_originals, processed_folder


def detect_faces(
    input_folder: str | os.PathLike[str],
    output_folder: str | os.PathLike[str] | None = None,
    *,
    config: SortMomentsConfig | None = None,
    face_model: Any = None,
    embedding_model: Any = None,
    progress_callback: ProgressCallback | None = None,
) -> Mapping[str, Any]:
    """Detect faces and write face crops/embeddings.

    ``face_model`` may be an InsightFace-compatible object with ``get`` or a
    callable. ``embedding_model`` may expose ``embed(image_rgb, face)`` or be a
    callable; it is useful when detection and embedding are separate models.
    """

    cfg = _config(config)
    input_path = Path(input_folder)
    output_path = Path(output_folder) if output_folder is not None else _default_intermediate_folder(input_path, cfg)

    return processphotos.detect_and_embed_faces(
        str(input_path),
        str(output_path),
        min_face_size=cfg.min_face_size,
        min_confidence=cfg.min_confidence,
        min_face_ratio=cfg.min_face_ratio,
        foreground_ratio_threshold=cfg.foreground_ratio_threshold,
        blur_threshold=cfg.blur_threshold,
        batch_size=cfg.batch_size,
        max_workers=cfg.max_workers,
        progress_callback=progress_callback,
        face_model=face_model,
        embedding_model=embedding_model,
        prefer_gpu=cfg.prefer_gpu,
        model_name=cfg.model_name,
        det_size=cfg.det_size,
    )


def group_faces(
    intermediate_folder: str | os.PathLike[str],
    *,
    input_folder: str | os.PathLike[str] | None = None,
    config: SortMomentsConfig | None = None,
    grouping_model: Any = None,
) -> tuple[Mapping[str, Sequence[str]], Path]:
    """Group previously detected faces and copy photos into person folders."""

    cfg = _config(config)
    intermediate_path = Path(intermediate_folder)
    input_path = Path(input_folder) if input_folder is not None else intermediate_path.parent

    if grouping_model is None:
        persons, processed_folder = processphotos.reorganize_by_person(
            str(intermediate_path),
            input_folder=str(input_path),
            similarity_threshold=cfg.similarity_threshold,
            session_id=cfg.session_id,
            final_output_folder=cfg.final_output_folder,
        )
        return persons, Path(processed_folder)

    records = _load_face_records(intermediate_path)
    groups = _call_grouping_model(grouping_model, records, cfg.similarity_threshold)
    return _organize_custom_groups(groups=groups, records=records, input_folder=input_path, config=cfg)


def organize_photos(
    input_folder: str | os.PathLike[str],
    *,
    work_folder: str | os.PathLike[str] | None = None,
    config: SortMomentsConfig | None = None,
    face_model: Any = None,
    embedding_model: Any = None,
    grouping_model: Any = None,
    progress_callback: ProgressCallback | None = None,
) -> ProcessingResult:
    """Run the full Sort Moments pipeline as an importable library call.

    By default this uses the existing InsightFace detector/embeddings and
    built-in cosine-similarity grouping. Supply custom ``face_model``,
    ``embedding_model``, or ``grouping_model`` to replace those stages.
    """

    cfg = _config(config)
    input_path = Path(input_folder)
    intermediate_path = Path(work_folder) if work_folder is not None else _default_intermediate_folder(input_path, cfg)

    embeddings = detect_faces(
        input_path,
        intermediate_path,
        config=cfg,
        face_model=face_model,
        embedding_model=embedding_model,
        progress_callback=progress_callback,
    )

    persons, processed_folder = group_faces(
        intermediate_path,
        input_folder=input_path,
        config=cfg,
        grouping_model=grouping_model,
    )

    processphotos.clean_filenames(str(processed_folder))

    embeddings_file = intermediate_path / "face_embeddings_insightface.pkl"
    result = ProcessingResult(
        persons=persons,
        output_folder=processed_folder,
        embeddings_file=embeddings_file,
        person_count=len(persons) if persons else 0,
        face_count=len(embeddings) if embeddings else 0,
    )

    if not cfg.keep_intermediate and intermediate_path.exists():
        shutil.rmtree(intermediate_path, ignore_errors=True)

    return result
