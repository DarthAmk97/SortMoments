"""Source checkout CLI for running Sort Moments without the desktop GUI.

This file is intentionally kept outside the PyPI library surface. The package
published as ``sortmoments`` is the importable Python API; this source wrapper is
for repo/git-clone based batch runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

"""
Command-line entry point for running Sort Moments without the PyQt GUI.

This module intentionally stays thin: it validates CLI/config input, prints a
clear execution plan, then delegates the real photo processing to
``processphotos.py`` so the GUI and CLI share the same pipeline.
"""

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EXCLUDED_DIRS = {"face_detection_output", "all_images_processed", "__pycache__", ".git"}
SAFE_TEMP_FOLDER_NAMES = {"face_detection_output", ".sortmoments_face_detection_output"}


class CliError(Exception):
    """A user-correctable CLI error."""


@dataclass(frozen=True)
class CliOptions:
    input_folder: Path
    output_folder: Path
    temp_folder: Path
    batch_size: int
    workers: int
    min_face_size: int
    min_confidence: float
    min_face_ratio: float
    foreground_ratio_threshold: float
    blur_threshold: int
    similarity_threshold: float
    prefer_gpu: bool
    model_name: str
    det_size: tuple[int, int]
    dry_run: bool
    confirm: bool
    yes: bool
    keep_temp: bool
    overwrite: bool


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise CliError(f"Config file not found: {config_path}")

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError(f"Config file is not valid JSON: {config_path} ({exc})") from exc

    if not isinstance(data, dict):
        raise CliError("Config file must contain a JSON object")

    return data


def _path_from(value: str | os.PathLike[str] | None, *, base: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _bounded_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sortmoments",
        description="Organize a photo folder by detected faces without launching the Sort Moments GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=defaults.get("input_folder"),
        help="Folder containing photos. May also be supplied by --config.",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        default=defaults.get("output_folder"),
        help="Final organized output folder. Defaults to <input>/all_images_processed.",
    )
    parser.add_argument(
        "--temp-folder",
        default=defaults.get("temp_folder"),
        help="Temporary face-crop/embedding folder. Defaults to <input>/face_detection_output.",
    )
    parser.add_argument(
        "--config",
        help="JSON config file. CLI flags override config values.",
    )
    parser.add_argument("--dry-run", action="store_true", default=bool(defaults.get("dry_run", False)),
                        help="Validate inputs and print the plan without importing models or writing output.")
    parser.add_argument("--confirm", action="store_true", default=bool(defaults.get("confirm", False)),
                        help="Ask for an interactive confirmation before processing.")
    parser.add_argument("-y", "--yes", action="store_true", default=bool(defaults.get("yes", False)),
                        help="Skip confirmation prompts.")
    parser.add_argument("--overwrite", action="store_true", default=bool(defaults.get("overwrite", False)),
                        help="Delete the final output folder before processing.")

    temp_group = parser.add_mutually_exclusive_group()
    temp_group.add_argument("--keep-temp", action="store_true", default=bool(defaults.get("keep_temp", False)),
                            help="Keep temporary face detection artifacts after processing.")
    temp_group.add_argument("--cleanup-temp", action="store_false", dest="keep_temp",
                            help="Remove the default temporary folder after processing.")

    parser.add_argument("--batch-size", type=_positive_int, default=int(defaults.get("batch_size", 8)),
                        help="Number of images to process per batch.")
    parser.add_argument("--workers", type=_positive_int, default=int(defaults.get("workers", 4)),
                        help="Parallel image loading workers.")
    parser.add_argument("--min-face-size", type=_positive_int, default=int(defaults.get("min_face_size", 80)),
                        help="Minimum detected face size in pixels.")
    parser.add_argument("--min-confidence", type=_bounded_float, default=float(defaults.get("min_confidence", 0.8)),
                        help="Minimum face detection confidence.")
    parser.add_argument("--min-face-ratio", type=float, default=float(defaults.get("min_face_ratio", 0.01)),
                        help="Minimum face size relative to image area.")
    parser.add_argument(
        "--foreground-ratio-threshold",
        type=float,
        default=float(defaults.get("foreground_ratio_threshold", 0.1)),
        help="Minimum ratio compared with the largest face in an image.",
    )
    parser.add_argument("--blur-threshold", type=_positive_int, default=int(defaults.get("blur_threshold", 60)),
                        help="Laplacian variance threshold used to filter blurry face crops.")
    parser.add_argument(
        "--similarity-threshold",
        type=_bounded_float,
        default=float(defaults.get("similarity_threshold", 0.5)),
        help="Face embedding similarity threshold used to group photos by person.",
    )

    gpu_default = bool(defaults.get("prefer_gpu", True))
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", action="store_true", dest="prefer_gpu", default=gpu_default,
                           help="Prefer GPU execution providers when available.")
    gpu_group.add_argument("--cpu", action="store_false", dest="prefer_gpu",
                           help="Force CPU-only model execution.")
    parser.add_argument("--model-name", default=str(defaults.get("model_name", "buffalo_l")),
                        help="InsightFace model pack name.")
    parser.add_argument(
        "--det-size",
        nargs=2,
        type=_positive_int,
        metavar=("WIDTH", "HEIGHT"),
        default=defaults.get("det_size", (640, 640)),
        help="Face detector input size.",
    )
    return parser


def _normalize_argv(argv: list[str] | None = None) -> list[str] | None:
    """Allow both `sortmoments <folder>` and `sortmoments organize <folder>`."""
    if argv is None:
        argv = sys.argv[1:]
    normalized = list(argv)
    if normalized and normalized[0] == "organize":
        return normalized[1:]
    return normalized


def parse_args(argv: list[str] | None = None) -> CliOptions:
    argv = _normalize_argv(argv)
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config")
    known, _ = config_probe.parse_known_args(argv)
    config = _load_config(known.config)

    parser = _build_parser(config)
    args = parser.parse_args(argv)

    input_folder = _path_from(args.input_folder)
    if input_folder is None:
        parser.error("input_folder is required unless provided in --config")

    input_folder = input_folder.resolve()
    output_folder = _path_from(args.output_folder, base=input_folder)
    if output_folder is None:
        output_folder = input_folder / "all_images_processed"
    temp_folder = _path_from(args.temp_folder, base=input_folder)
    if temp_folder is None:
        temp_folder = input_folder / "face_detection_output"

    det_size = tuple(int(part) for part in args.det_size)
    if len(det_size) != 2:
        parser.error("--det-size requires WIDTH HEIGHT")

    return CliOptions(
        input_folder=input_folder,
        output_folder=output_folder.resolve(),
        temp_folder=temp_folder.resolve(),
        batch_size=args.batch_size,
        workers=args.workers,
        min_face_size=args.min_face_size,
        min_confidence=args.min_confidence,
        min_face_ratio=args.min_face_ratio,
        foreground_ratio_threshold=args.foreground_ratio_threshold,
        blur_threshold=args.blur_threshold,
        similarity_threshold=args.similarity_threshold,
        prefer_gpu=args.prefer_gpu,
        model_name=args.model_name,
        det_size=(det_size[0], det_size[1]),
        dry_run=args.dry_run,
        confirm=args.confirm,
        yes=args.yes,
        keep_temp=args.keep_temp,
        overwrite=args.overwrite,
    )


def find_images(input_folder: Path) -> list[Path]:
    images: list[Path] = []
    for root, dirs, files in os.walk(input_folder):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for filename in files:
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS and "_representative_face" not in filename:
                images.append(Path(root) / filename)
    return images


def validate_options(options: CliOptions) -> None:
    if not options.input_folder.exists():
        raise CliError(f"Input folder does not exist: {options.input_folder}")
    if not options.input_folder.is_dir():
        raise CliError(f"Input path is not a folder: {options.input_folder}")
    if options.output_folder == options.input_folder:
        raise CliError("Output folder must be different from the input folder")
    if options.temp_folder == options.input_folder:
        raise CliError("Temporary folder must be different from the input folder")


def print_plan(options: CliOptions, image_count: int) -> None:
    noun = "image" if image_count == 1 else "images"
    print("Sort Moments CLI plan")
    print("=====================")
    print(f"Input folder:  {options.input_folder}")
    print(f"Output folder: {options.output_folder}")
    print(f"Temp folder:   {options.temp_folder}")
    print(f"Found:         {image_count} {noun}")
    print(f"Mode:          {'dry run' if options.dry_run else 'process photos'}")
    print(f"Model:         {options.model_name} @ {options.det_size[0]}x{options.det_size[1]}")
    print(f"Execution:     {'prefer GPU' if options.prefer_gpu else 'CPU only'}")
    print(f"Batch/workers: {options.batch_size}/{options.workers}")
    if options.output_folder.exists() and any(options.output_folder.iterdir()):
        action = "will be deleted first" if options.overwrite and not options.dry_run else "will be reused"
        print(f"Existing output: {action}")


def confirm_if_requested(options: CliOptions) -> None:
    if options.yes or not options.confirm:
        return
    answer = input("Start Sort Moments processing now? Type 'yes' to continue: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise CliError("Processing cancelled by user")


def _is_safe_to_delete_output(options: CliOptions) -> bool:
    output = options.output_folder
    input_folder = options.input_folder
    return (
        output != input_folder
        and (input_folder not in output.parents or output.name == "all_images_processed")
    )


def _is_safe_to_delete_temp(options: CliOptions) -> bool:
    temp = options.temp_folder
    inside_known_base = options.input_folder in temp.parents or options.output_folder in temp.parents
    return temp.name in SAFE_TEMP_FOLDER_NAMES and inside_known_base


def _remove_output_if_requested(options: CliOptions) -> None:
    if not options.overwrite or not options.output_folder.exists():
        return
    if not _is_safe_to_delete_output(options):
        raise CliError(f"Refusing to overwrite unsafe output folder: {options.output_folder}")
    shutil.rmtree(options.output_folder)


def _cleanup_temp_if_requested(options: CliOptions) -> None:
    if options.keep_temp or not options.temp_folder.exists():
        return
    if not _is_safe_to_delete_temp(options):
        print(f"Keeping temp folder because it does not look CLI-owned: {options.temp_folder}")
        return
    shutil.rmtree(options.temp_folder)


def run_sortmoments(options: CliOptions) -> Path:
    from sortmoments.pipeline import clean_filenames, detect_and_embed_faces, reorganize_by_person

    _remove_output_if_requested(options)
    options.temp_folder.mkdir(parents=True, exist_ok=True)
    options.output_folder.mkdir(parents=True, exist_ok=True)

    detect_and_embed_faces(
        str(options.input_folder),
        str(options.temp_folder),
        min_face_size=options.min_face_size,
        min_confidence=options.min_confidence,
        min_face_ratio=options.min_face_ratio,
        foreground_ratio_threshold=options.foreground_ratio_threshold,
        blur_threshold=options.blur_threshold,
        batch_size=options.batch_size,
        max_workers=options.workers,
        prefer_gpu=options.prefer_gpu,
        model_name=options.model_name,
        det_size=options.det_size,
    )

    _persons, processed_folder = reorganize_by_person(
        str(options.temp_folder),
        input_folder=str(options.input_folder),
        similarity_threshold=options.similarity_threshold,
        final_output_folder=str(options.output_folder),
    )
    final_folder = Path(processed_folder) if processed_folder else options.output_folder
    clean_filenames(str(final_folder))
    _cleanup_temp_if_requested(options)
    return final_folder


def main(argv: list[str] | None = None) -> int:
    try:
        options = parse_args(argv)
        validate_options(options)
        images = find_images(options.input_folder)
        print_plan(options, len(images))

        if options.dry_run:
            print("Dry run complete. No models were loaded and no files were written.")
            return 0

        if not images:
            raise CliError("No supported images found. Supported extensions: .jpg, .jpeg, .png, .bmp")

        confirm_if_requested(options)
        final_folder = run_sortmoments(options)
        print(f"Done. Organized photos are in: {final_folder}")
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except CliError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2 if "cancelled" in str(exc).lower() else 1


if __name__ == "__main__":
    raise SystemExit(main())
