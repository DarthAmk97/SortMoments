import os
import shutil
import subprocess
import sys
from pathlib import Path

APP_NAME = "SortMoments"
ENTRYPOINT = "photo_organizer.py"   # change if your entry differs

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
    """Get the path to InsightFace models if they exist."""
    home = Path.home()
    models_path = home / ".insightface" / "models" / "buffalo_l"
    if models_path.exists():
        return str(models_path)
    return None

def main():
    if sys.platform != "darwin":
        print("This script is intended to run on macOS.")
        sys.exit(1)

    # clean old builds
    for p in ["build", "dist", f"{APP_NAME}.spec"]:
        if Path(p).exists():
            if Path(p).is_dir():
                shutil.rmtree(p)
            else:
                Path(p).unlink()

    # ensure deps
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # IMPORTANT: macOS uses ":" in --add-data (Windows uses ";")
    add_data = [
        "--add-data", "processphotos.py:.",
        "--add-data", "logo.png:.",
    ]

    icon_path = ensure_macos_icon()
    if icon_path:
        add_data += ["--add-data", f"{icon_path}:."]  # include icon in bundle

    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--windowed",                 # no terminal window
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
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "numpy.core._methods",
        "--hidden-import", "numpy.lib.format",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "onnxruntime.capi",
        "--hidden-import", "insightface",
        "--hidden-import", "insightface.app",
        "--hidden-import", "insightface.app.face_analysis",
        "--hidden-import", "insightface.model_zoo",
        "--hidden-import", "insightface.model_zoo.model_zoo",
        "--hidden-import", "insightface.utils",
        "--hidden-import", "insightface.utils.face_align",
        "--hidden-import", "sklearn",
        "--hidden-import", "sklearn.cluster",
        "--hidden-import", "sklearn.neighbors",
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.spatial",
        "--hidden-import", "scipy.special",
        "--hidden-import", "albumentations",
        "--hidden-import", "prettytable",
        "--hidden-import", "easydict",
        ENTRYPOINT,
        *add_data,
    ]

    if icon_path:
        cmd.insert(-2, "--icon")
        cmd.insert(-2, icon_path)

    models_path = get_insightface_models_path()
    if models_path:
        print(f"Found InsightFace models at: {models_path}")
        print("Including models in app bundle (recommended for offline use).")
        cmd.insert(-1, "--add-data")
        cmd.insert(-1, f"{models_path}:insightface_models/buffalo_l")
    else:
        print("WARNING: InsightFace models not found. First run will download models.")

    run(cmd)

    app_path = Path("dist") / f"{APP_NAME}.app"
    if not app_path.exists():
        print("Build finished but .app not found at:", app_path)
        sys.exit(2)

    # Optional: zip it for GitHub Releases
    zip_name = f"{APP_NAME}-macos.zip"
    if Path(zip_name).exists():
        Path(zip_name).unlink()

    # ditto preserves .app bundle correctly
    run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(app_path), zip_name])
    print(f"✅ Built {app_path} and packaged {zip_name}")

if __name__ == "__main__":
    main()
