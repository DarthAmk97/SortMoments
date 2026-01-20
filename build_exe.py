"""
Build script for creating Sort Moments Windows executable.
Creates a SINGLE standalone .exe file that works on any Windows PC.

Usage:
    1. Install build dependencies: pip install pyinstaller
    2. Run this script: python build_exe.py
    3. Find the executable in the 'dist' folder
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
        return True
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True


def get_insightface_models_path():
    """Get the path to InsightFace models if they exist."""
    home = Path.home()
    models_path = home / ".insightface" / "models" / "buffalo_l"
    if models_path.exists():
        return str(models_path)
    return None


def build_single_exe():
    """Build a single standalone .exe file."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("Sort Moments - Building Standalone Windows Executable")
    print("=" * 60)
    print()

    # Check PyInstaller
    check_pyinstaller()

    # Clean previous builds
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            print(f"Cleaning {folder}...")
            shutil.rmtree(folder)

    # Remove old spec file
    if os.path.exists("SortMoments.spec"):
        os.remove("SortMoments.spec")

    print()
    print("Building single executable file...")
    print("This may take 5-10 minutes due to large dependencies...")
    print()

    # PyInstaller command for single file
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=SortMoments",
        "--onefile",  # Single .exe file - no _internal folder needed
        "--windowed",  # No console window
        "--noconfirm",  # Replace output without asking

        # Hidden imports that PyInstaller might miss
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL._tkinter_finder",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=numpy.core._methods",
        "--hidden-import=numpy.lib.format",
        "--hidden-import=onnxruntime",
        "--hidden-import=onnxruntime.capi",
        "--hidden-import=insightface",
        "--hidden-import=insightface.app",
        "--hidden-import=insightface.app.face_analysis",
        "--hidden-import=insightface.model_zoo",
        "--hidden-import=insightface.model_zoo.model_zoo",
        "--hidden-import=insightface.utils",
        "--hidden-import=insightface.utils.face_align",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.cluster",
        "--hidden-import=sklearn.neighbors",
        "--hidden-import=scipy",
        "--hidden-import=scipy.spatial",
        "--hidden-import=scipy.special",
        "--hidden-import=albumentations",
        "--hidden-import=prettytable",
        "--hidden-import=easydict",

        # Collect all data from these packages (includes DLLs and models)
        "--collect-all=insightface",
        "--collect-all=onnxruntime",
        "--collect-all=cv2",

        # Include the processing module
        "--add-data=processphotos.py;.",

        # Main script
        "photo_organizer.py"
    ]

    # Check for InsightFace models and include them
    models_path = get_insightface_models_path()
    if models_path:
        print(f"Found InsightFace models at: {models_path}")
        print("Including models in executable (recommended for offline use)")
        # Add models to the bundle - they'll be extracted to temp on runtime
        cmd.insert(-1, f"--add-data={models_path};insightface_models/buffalo_l")
    else:
        print("=" * 60)
        print("WARNING: InsightFace models not found!")
        print("=" * 60)
        print()
        print("For best results, run the app once first to download models:")
        print("  python photo_organizer.py")
        print()
        print("Then run this build script again to include the models.")
        print("Without bundled models, users need internet on first run.")
        print()
        input("Press Enter to continue building without models, or Ctrl+C to cancel...")

    print()
    print("Running PyInstaller (this takes a while)...")
    print()

    try:
        subprocess.check_call(cmd)

        exe_path = os.path.join(script_dir, 'dist', 'SortMoments.exe')

        if os.path.exists(exe_path):
            # Get file size
            size_mb = os.path.getsize(exe_path) / (1024 * 1024)

            print()
            print("=" * 60)
            print("BUILD SUCCESSFUL!")
            print("=" * 60)
            print()
            print(f"Executable: {exe_path}")
            print(f"Size: {size_mb:.1f} MB")
            print()
            print("This is a STANDALONE executable:")
            print("- No Python installation required")
            print("- No additional files needed")
            print("- Just copy and run SortMoments.exe")
            print()

            if not models_path:
                print("NOTE: First run requires internet to download AI models (~300MB)")
            else:
                print("AI models are bundled - works completely offline!")

            print()
            print("=" * 60)
            return True
        else:
            print("ERROR: Executable was not created")
            return False

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print(f"BUILD FAILED: {e}")
        print("=" * 60)
        print()
        print("Troubleshooting tips:")
        print("1. Update PyInstaller: pip install pyinstaller --upgrade")
        print("2. Install all dependencies: pip install -r requirements.txt")
        print("3. Temporarily disable antivirus (it may block the build)")
        print("4. Run as Administrator if permission errors occur")
        print()
        return False


def build_folder_version():
    """Build folder version (faster startup, easier debugging)."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("Sort Moments - Building Folder Version")
    print("=" * 60)

    check_pyinstaller()

    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=SortMoments",
        "--onedir",  # Folder with exe + _internal
        "--windowed",
        "--noconfirm",

        "--hidden-import=PIL",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=onnxruntime",
        "--hidden-import=insightface",
        "--hidden-import=insightface.app",
        "--hidden-import=insightface.app.face_analysis",
        "--hidden-import=insightface.model_zoo",
        "--hidden-import=sklearn",
        "--hidden-import=scipy",

        "--collect-all=insightface",
        "--collect-all=onnxruntime",
        "--collect-all=cv2",

        "--add-data=processphotos.py;.",

        "photo_organizer.py"
    ]

    models_path = get_insightface_models_path()
    if models_path:
        cmd.insert(-1, f"--add-data={models_path};insightface_models/buffalo_l")

    try:
        subprocess.check_call(cmd)
        print()
        print("BUILD SUCCESSFUL!")
        print(f"Output: {os.path.join(script_dir, 'dist', 'SortMoments')}")
        print()
        print("To distribute: Copy the entire 'SortMoments' folder")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Sort Moments executable")
    parser.add_argument("--folder", action="store_true",
                       help="Create folder version instead of single exe (faster startup)")

    args = parser.parse_args()

    if args.folder:
        build_folder_version()
    else:
        # Default: build single standalone exe
        build_single_exe()
