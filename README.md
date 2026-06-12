# Sort Moments

<p align="center">
  <img src="sortmoments_logo_transparent.png" alt="Sort Moments transparent logo" width="260" />
</p>

**Organize your photos by faces using AI**

A free, open-source desktop application that automatically groups your photos by the people in them using advanced face detection and recognition.

<video src="https://github.com/user-attachments/assets/f30ffacf-f4f0-4550-af09-1cee316e90b4" controls width="420"> 
  Demo
</video>


[Download](https://sortmoments.com) | [Website](https://sortmoments.com) | [Report Issues](https://github.com/DarthAmk97/SortMoments/issues)

---

## Features

- **AI-Powered Face Detection** - Uses InsightFace with the buffalo_l model for accurate face detection and recognition
- **Automatic Grouping** - Photos are automatically grouped by person based on face similarity
- **GPU Accelerated** - Supports DirectML (any GPU on Windows), CUDA, or CPU fallback
- **Privacy First** - All processing happens locally on your machine. No data leaves your computer
- **Modern Interface** - Clean, dark-themed PyQt6 interface with progress tracking
- **Free & Open Source** - MIT licensed, free forever

---

## Quick Start

### Option 1: Download Pre-built Executable (Recommended)

1. Download `SortMoments.exe` from [sortmoments.com](https://sortmoments.com) or [GitHub Releases](https://github.com/DarthAmk97/SortMoments/releases)
2. Run the executable (no installation required)
3. Drag & drop a folder containing your photos
4. Click "Start Processing"
5. Done! Your photos are organized by person

### Option 2: Run from Source

```bash
# Clone the repository
git clone https://github.com/DarthAmk97/SortMoments.git
cd SortMoments

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the application
python photo_organizer.py
```

### Option 3: Python Library Package (Coming Soon)

Use this when you want Sort Moments inside your own Python scripts, notebooks,
or backend jobs. This is the intended package surface, separate from the
repo-based CLI. It is **not on PyPI yet**; keep using the source CLI until the
wheel, metadata, and import surface are verified.

```bash
# planned, not live yet
python -m pip install sortmoments
```

```python
from sortmoments import SortMomentsConfig, organize_photos

result = organize_photos(
    "C:/path/to/photos",
    config=SortMomentsConfig(similarity_threshold=0.58),
)
print(result.output_folder)
```

### Option 4: Run the No-GUI CLI from Source

Use the source CLI when you want local batch processing from a terminal or
script without launching the desktop GUI:

```bash
git clone https://github.com/DarthAmk97/SortMoments.git
cd SortMoments
python -m pip install -r requirements.txt

# Preview what would be processed without loading the AI model
python sortmoments_cli.py organize "C:\path\to\photos" --dry-run

# Process photos into the default <input>/all_images_processed folder
python sortmoments_cli.py organize "C:\path\to\photos" --yes

# Choose a custom output folder and force CPU execution
python sortmoments_cli.py organize "C:\path\to\photos" --output-folder "D:\sorted-photos" --cpu --yes
```

Useful options:

- `--output-folder PATH` sets the final organized folder.
- `--dry-run` validates paths and prints the plan without writing files.
- `--confirm` asks for an interactive confirmation before processing.
- `--config sortmoments.json` loads defaults from JSON; CLI flags override it.
- `--cpu` / `--gpu`, `--model-name`, and `--det-size WIDTH HEIGHT` expose model hooks.

### Option 5: Build Your Own Executable

See [BUILDING.md](BUILDING.md) for detailed build instructions.

---

## System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: ~1GB for the application + space for your photos
- **GPU**: Optional but recommended for faster processing
  - Any DirectX 12 compatible GPU (NVIDIA, AMD, Intel)
  - NVIDIA CUDA-capable GPU for best performance

---

## How It Works

1. **Face Detection** - Scans all images in your folder and detects faces
2. **Embedding Creation** - Creates a unique mathematical representation for each face
3. **Clustering** - Groups similar faces together (same person)
4. **Organization** - Copies photos into person-specific folders

---

## Python Library API (Coming Soon)

Sort Moments is being shaped into an importable Python API. The facade keeps
the existing desktop pipeline defaults, while letting advanced users replace
the face detector, embedding model, or grouping model. This section documents
the target API shape; it is blocked on the PyPI release being verified.

```bash
# planned, not live yet
python -m pip install sortmoments
```

This is the Python package surface. It is intentionally separate from the
repo-based no-GUI CLI. For now, the CLI is the working terminal path and the
package stays on hold until publishing is sorted.

The package follows the Python Packaging User Guide pattern:

```text
SortMoments/
├── LICENSE
├── README.md
├── pyproject.toml
├── src/
│   └── sortmoments/
│       ├── __init__.py
│       ├── api.py
│       ├── models.py
│       └── pipeline.py
└── tests/
```

```python
from sortmoments import SortMomentsConfig, SortMomentsOrganizer, organize_photos

result = organize_photos(
    "C:/path/to/photos",
    config=SortMomentsConfig(similarity_threshold=0.5),
)

print(result.output_folder)
print(result.person_count)
```

Custom models can be passed stage-by-stage:

```python
result = organize_photos(
    "C:/path/to/photos",
    face_model=my_detector,        # .get(image_rgb) or callable(image_rgb)
    embedding_model=my_embedder,   # .embed(image_rgb, face) or callable
    grouping_model=my_grouper,     # .group(face_records, similarity_threshold=...)
)
```

For repeated jobs, create a configured organizer:

```python
organizer = SortMomentsOrganizer(
    SortMomentsConfig(similarity_threshold=0.62),
    face_model=my_detector,
    embedding_model=my_embedder,
    grouping_model=my_grouper,
)
result = organizer.organize("C:/path/to/photos")
```

Lower-level helpers are also available:

- `detect_faces(...)` writes face crops and embeddings.
- `group_faces(...)` groups previously detected faces.
- `DetectedFace` is a small convenience object for custom detectors.

---

## Output Structure

After processing, photos are organized in an `all_images_processed/` folder:

```
your_photos_folder/
└── all_images_processed/
    ├── rename_0/                    # Person 1
    │   ├── rename_0_representative_face.jpg
    │   ├── photo1.jpg
    │   └── photo2.jpg
    ├── rename_1/                    # Person 2
    │   ├── rename_1_representative_face.jpg
    │   └── photo3.jpg
    ├── all_group_photos/            # Photos with 3+ people
    │   └── group_photo.jpg
    └── README.txt
```

You can rename the folders to actual person names directly in the app!

---

## Tech Stack

- **GUI**: PyQt6
- **Face Detection/Recognition**: InsightFace (buffalo_l model)
- **Image Processing**: OpenCV, Pillow
- **GPU Acceleration**: ONNX Runtime with DirectML/CUDA
- **Build Tool**: PyInstaller

---

## Troubleshooting

### "Missing Dependencies" warning
```bash
pip install -r requirements.txt
```

### InsightFace model download issues
The model downloads automatically on first run. If issues occur:
```bash
python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis('buffalo_l'); app.prepare(ctx_id=-1)"
```

### Processing is slow
- Ensure GPU acceleration is being used (check console output)
- Processing time depends on the number and resolution of images
- First run may be slower due to model loading

### Error during processing
- Click "Send Error Logs" button to report the issue
- Check the log file location shown in the error dialog

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face detection and recognition
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the GUI framework
- [ONNX Runtime](https://onnxruntime.ai/) for GPU acceleration

---

Made with care by Abdullah Khawaja with dearest collabortator Claude Code
