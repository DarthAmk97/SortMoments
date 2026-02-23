import shutil
import subprocess
import sys
import argparse
from pathlib import Path

APP_NAME = "SortMoments"
ENTRYPOINT = "photo_organizer.py"


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd)


def ensure_macos_icon():
    """
    Create a .icns from logo.png if needed and return icon path.
    Returns None if no source icon is available.
    """
    png_path = Path("logo.png")
    icns_path = Path("logo.icns")

    if icns_path.exists():
        return str(icns_path)

    if not png_path.exists():
        return None

    iconset_dir = Path("build") / "logo.iconset"
    iconset_dir.mkdir(parents=True, exist_ok=True)

    sizes = [16, 32, 64, 128, 256, 512, 1024]

    for size in sizes:
        out_png = iconset_dir / f"icon_{size}x{size}.png"
        run(["sips", "-z", str(size), str(size), str(png_path), "--out", str(out_png)])

        if size <= 512:
            out_png_2x = iconset_dir / f"icon_{size}x{size}@2x.png"
            run(["sips", "-z", str(size * 2), str(size * 2), str(png_path), "--out", str(out_png_2x)])

    run(["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_path)])

    return str(icns_path)


def get_insightface_models_path():
    home = Path.home()
    models_path = home / ".insightface" / "models" / "buffalo_l"
    if models_path.exists():
        return str(models_path)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-models",
        action="store_true",
        help="Bundle InsightFace models into the app (offline friendly)",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("This script is intended to run on macOS.")
        sys.exit(1)

    # Clean old builds
    for p in ["build", "dist", f"{APP_NAME}.spec"]:
        path = Path(p)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    # Ensure dependencies
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Data files to bundle (safe tuple format)
    add_data = [
        ("processphotos.py", "."),
        ("logo.png", "."),
    ]

    icon_path = ensure_macos_icon()
    if icon_path:
        add_data.append((icon_path, "."))

    if args.bundle_models:
        models_path = get_insightface_models_path()
        if models_path:
            print(f"Bundling InsightFace models from: {models_path}")
            add_data.append((models_path, "insightface_models/buffalo_l"))
        else:
            print("WARNING: InsightFace models not found. Models will download at first run.")

    # Build command
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--windowed",
        "--name", APP_NAME,
        "--clean",

        "--collect-all", "insightface",
        "--collect-all", "onnxruntime",
        "--collect-all", "cv2",
        "--collect-all", "scipy",
        "--collect-all", "sklearn",
        "--collect-all", "albumentations",

        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "prettytable",
        "--hidden-import", "easydict",
    ]

    if icon_path:
        cmd += ["--icon", icon_path]

    cmd += [ENTRYPOINT]

    # Add data safely (no orphan flags possible)
    for src, dest in add_data:
        cmd += ["--add-data", f"{src}:{dest}"]

    run(cmd)

    app_path = Path("dist") / f"{APP_NAME}.app"
    if not app_path.exists():
        print("Build finished but .app not found at:", app_path)
        sys.exit(2)

    # Zip for GitHub Releases
    zip_name = f"{APP_NAME}-macos.zip"
    zip_path = Path(zip_name)

    if zip_path.exists():
        zip_path.unlink()

    run([
        "ditto",
        "-c",
        "-k",
        "--sequesterRsrc",
        "--keepParent",
        str(app_path),
        str(zip_path),
    ])

    print(f"✅ Built {app_path}")
    print(f"📦 Packaged {zip_path}")


if __name__ == "__main__":
    main()
