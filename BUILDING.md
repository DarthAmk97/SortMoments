# Building Sort Moments Executable

This guide explains how to build the Sort Moments application into a standalone Windows executable.

## Prerequisites

- **Python 3.8 or higher** (3.10+ recommended)
- **pip** (Python package manager)
- **Windows 10/11** (building on the target platform is recommended)
- **~2GB disk space** for build artifacts

## Step 1: Set Up Environment

```bash
# Clone the repository (or download source code)
git clone https://github.com/abdullahkhawaja/SortMoments.git
cd SortMoments

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

## Step 2: Install Dependencies

```bash
# Install application dependencies
pip install -r requirements.txt

# Install PyInstaller for building
pip install pyinstaller>=5.0
```

## Step 3: Download AI Models (Optional but Recommended)

The InsightFace model (~300MB) can be pre-downloaded to bundle with the executable. If not bundled, it will download on first run.

```bash
# Pre-download the model
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis('buffalo_l'); app.prepare(ctx_id=-1)"
```

The model will be saved to `~/.insightface/models/buffalo_l/`

## Step 4: Build the Executable

### Option A: Using the Build Script (Recommended)

```bash
python build_exe.py
```

This will:
- Detect and bundle AI models if available
- Configure all hidden imports
- Create a single-file executable
- Output to `dist/SortMoments.exe`

### Option B: Using BUILD.bat

Simply double-click `BUILD.bat` or run:

```bash
BUILD.bat
```

### Option C: Manual PyInstaller Command

```bash
pyinstaller --onefile --windowed --name SortMoments ^
    --hidden-import=onnxruntime ^
    --hidden-import=insightface ^
    --hidden-import=cv2 ^
    --hidden-import=PIL ^
    --hidden-import=numpy ^
    --hidden-import=PyQt6 ^
    photo_organizer.py
```

## Build Output

After building, you'll find:

```
dist/
└── SortMoments.exe    # Standalone executable (~800MB with models)

build/
└── ...                # Intermediate build files (can be deleted)
```

## Build Options

### Single File vs Folder

**Single File (Default)** - One `.exe` file, slower startup but easier to distribute:
```bash
python build_exe.py --onefile
```

**Folder** - Faster startup, but multiple files:
```bash
python build_exe.py --onedir
```

### With/Without Console

**Windowed (Default)** - No console window:
```bash
python build_exe.py --windowed
```

**With Console** - Shows console for debugging:
```bash
python build_exe.py --console
```

## Troubleshooting

### Build fails with "ModuleNotFoundError"

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install pyinstaller
```

### Executable is very large (>1GB)

This is normal if AI models are bundled. To reduce size:
- Don't bundle models (they'll download on first run)
- Use `--onedir` instead of `--onefile`

### "Failed to execute script" error when running .exe

- Ensure you're running on Windows 10/11 64-bit
- Try running from command prompt to see error messages:
  ```bash
  cd dist
  SortMoments.exe
  ```

### Antivirus flags the executable

Some antivirus software flags PyInstaller executables. This is a false positive. You can:
- Add an exception in your antivirus
- Sign the executable with a code signing certificate

### DirectML/GPU not working

Ensure these are installed in your build environment:
```bash
pip install onnxruntime-directml>=1.15.0
```

## Distribution

After building:

1. **Test the executable** on a clean Windows machine
2. **Create a GitHub Release** with the `.exe` file
3. **Upload to your server** for direct downloads

The executable is fully standalone - users just download and run, no installation needed.

## File Sizes

| Configuration | Approximate Size |
|---------------|------------------|
| Without models | ~400MB |
| With models bundled | ~800MB |
| Folder build | ~600MB (multiple files) |

## Build Environment Tested

- Windows 11 64-bit
- Python 3.10.11
- PyInstaller 5.13.0
- PyQt6 6.5.0
