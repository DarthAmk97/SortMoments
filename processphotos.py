"""
Optimized Photo Processing Pipeline
- Uses InsightFace for unified detection + embedding (single pass)
- GPU acceleration with automatic CPU fallback
- Batch processing for improved throughput
- Parallel image loading
- Maintains all filtering logic (blur, size ratios, election system)
"""

import os
import cv2
import numpy as np
from pathlib import Path
import pickle
from glob import glob
import argparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import shutil
import sys


# ============================================================================
# GPU Detection and Model Initialization
# ============================================================================

def get_execution_providers():
    """
    Detect available execution providers and return them in priority order.
    DirectML first (works with ANY GPU on Windows), then CUDA, then CPU as fallback.
    """
    providers = []

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()

        # Check for DirectML (works with ANY GPU - NVIDIA, AMD, Intel on Windows)
        if 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
            print("GPU detected: Using DirectML for acceleration (works with any GPU)")
        # Fallback to CUDA if available (NVIDIA only)
        elif 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
            print("GPU detected: Using CUDA for acceleration")

        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')

        if len(providers) == 1:
            print("No GPU detected: Using CPU")

    except ImportError:
        providers = ['CPUExecutionProvider']
        print("ONNX Runtime not found, defaulting to CPU")

    return providers


def initialize_face_analyzer(det_size=(640, 640)):
    """
    Initialize InsightFace FaceAnalysis with automatic GPU/CPU selection.

    Args:
        det_size: Detection input size (width, height). Smaller = faster but less accurate.

    Returns:
        FaceAnalysis app instance
    """
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        raise ImportError("InsightFace not installed. Please install with: pip install insightface onnxruntime-gpu")

    # Fix tqdm compatibility issue when stdout/stderr is None (common in GUI apps)
    if sys.stdout is None or sys.stderr is None:
        try:
            import tqdm.std
            # Patch tqdm's __init__ to use a dummy file when stdout is unavailable
            original_init = tqdm.std.tqdm.__init__

            def patched_init(self, iterable=None, desc=None, total=None, leave=True,
                           file=None, ncols=None, mininterval=0.1, maxinterval=10.0,
                           miniters=None, ascii=None, disable=False, unit='it',
                           unit_scale=False, dynamic_ncols=False, smoothing=0.3,
                           bar_format=None, initial=0, position=None, postfix=None,
                           unit_divisor=1000, write_bytes=None, lock_args=None,
                           nrows=None, colour=None, delay=0, **kwargs):
                # Force file to be None to disable output
                original_init(self, iterable=iterable, desc=desc, total=total,
                            leave=leave, file=None, ncols=ncols,
                            mininterval=mininterval, maxinterval=maxinterval,
                            miniters=miniters, ascii=ascii, disable=True,
                            unit=unit, unit_scale=unit_scale,
                            dynamic_ncols=dynamic_ncols, smoothing=smoothing,
                            bar_format=bar_format, initial=initial, position=position,
                            postfix=postfix, unit_divisor=unit_divisor,
                            write_bytes=write_bytes, lock_args=lock_args,
                            nrows=nrows, colour=colour, delay=delay, **kwargs)

            tqdm.std.tqdm.__init__ = patched_init
        except Exception:
            pass

    providers = get_execution_providers()

    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0, det_size=det_size)
        print(f"InsightFace initialized with detection size: {det_size}")
        return app
    except Exception as e:
        print(f"Error initializing with GPU, falling back to CPU: {e}")
        app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=det_size)
        return app


# ============================================================================
# Image Loading Utilities
# ============================================================================

def load_image(image_path, max_size=1920):
    """
    Load and optionally downsample an image for processing.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height)

    Returns:
        tuple: (image_bgr, original_image_bgr, scale_factor)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None, 1.0

    original = img.copy()
    h, w = img.shape[:2]

    # Downsample if needed
    scale = 1.0
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img, original, scale


def load_images_parallel(image_paths, max_size=1920, max_workers=4):
    """
    Load multiple images in parallel using ThreadPoolExecutor.

    Args:
        image_paths: List of image file paths
        max_size: Maximum dimension for downsampling
        max_workers: Number of parallel workers

    Returns:
        dict: {path: (processed_img, original_img, scale)}
    """
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(load_image, path, max_size): path
            for path in image_paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                if result[0] is not None:
                    results[path] = result
            except Exception as e:
                print(f"Error loading {path}: {e}")

    return results


# ============================================================================
# Face Filtering Utilities
# ============================================================================

def calculate_blur_value(face_img):
    """
    Calculate blur value using Laplacian variance.
    Higher value = sharper image.
    """
    if face_img is None or face_img.size == 0:
        return 0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def filter_faces(faces, image, img_area, min_face_size=80, min_confidence=0.8,
                 min_face_ratio=0.01, foreground_ratio_threshold=0.1, blur_threshold=60):
    """
    Filter detected faces based on multiple criteria.
    Includes election system for when no faces pass strict criteria.

    Args:
        faces: List of InsightFace face objects
        image: Original image (BGR)
        img_area: Total image area in pixels
        min_face_size: Minimum face dimension in pixels
        min_confidence: Minimum detection confidence (0-1)
        min_face_ratio: Minimum face area relative to image
        foreground_ratio_threshold: Minimum ratio compared to largest face
        blur_threshold: Minimum Laplacian variance for sharpness

    Returns:
        list: Filtered face objects with additional metadata
    """
    if not faces:
        return []

    # Gather face information
    face_info = []
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        face_area = w * h
        face_ratio = face_area / img_area
        score = face.det_score

        # Extract face region for blur check
        face_img = image[max(0, y1):min(image.shape[0], y2),
                        max(0, x1):min(image.shape[1], x2)]

        if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
            continue

        blur_value = calculate_blur_value(face_img)
        is_blurry = blur_value < blur_threshold

        # Composite score for election system
        composite_score = (
            score * 2.0 +
            face_ratio * 5.0 +
            (1.0 - int(is_blurry)) * 0.5 +
            min(blur_value / 200, 1.0)
        )

        face_info.append({
            "face": face,
            "bbox": bbox,
            "area": face_area,
            "ratio": face_ratio,
            "score": score,
            "is_blurry": is_blurry,
            "blur_value": blur_value,
            "composite_score": composite_score,
            "embedding": face.embedding
        })

    if not face_info:
        return []

    # Sort by area (largest first)
    face_info.sort(key=lambda x: x["area"], reverse=True)
    largest_face_area = face_info[0]["area"]

    # Apply filtering criteria
    filtered_faces = []
    for info in face_info:
        x1, y1, x2, y2 = info["bbox"]
        w, h = x2 - x1, y2 - y1
        area_ratio = info["area"] / largest_face_area

        passes_filter = (
            info["ratio"] >= min_face_ratio and
            w >= min_face_size and h >= min_face_size and
            area_ratio >= foreground_ratio_threshold and
            info["score"] >= min_confidence and
            not info["is_blurry"]
        )

        info["filtered"] = not passes_filter
        if passes_filter:
            filtered_faces.append(info)

    # Election system: if no faces pass, select best candidates
    if not filtered_faces:
        sorted_faces = sorted(face_info, key=lambda x: x["composite_score"], reverse=True)

        if len(sorted_faces) == 1:
            filtered_faces = [sorted_faces[0]]
        elif len(sorted_faces) == 2:
            filtered_faces = sorted_faces
        else:
            filtered_faces = sorted_faces[:2]

        for f in filtered_faces:
            f["elected"] = True

    return filtered_faces


# ============================================================================
# Main Processing Function (Unified Detection + Embedding)
# ============================================================================

def detect_and_embed_faces(input_folder, output_folder,
                           min_face_size=50, min_confidence=0.7,
                           min_face_ratio=0.005, foreground_ratio_threshold=0.05,
                           blur_threshold=50, batch_size=8, max_workers=4,
                           progress_callback=None):
    """
    Unified face detection and embedding using InsightFace.
    Performs detection and embedding in a single pass with GPU acceleration.

    Args:
        input_folder: Path to folder containing images
        output_folder: Path to store results
        min_face_size: Minimum face size in pixels
        min_confidence: Minimum detection confidence (0-1)
        min_face_ratio: Minimum face size relative to image
        foreground_ratio_threshold: Minimum ratio compared to largest face
        blur_threshold: Laplacian variance threshold for blur detection
        batch_size: Number of images to process in each batch
        max_workers: Number of parallel workers for image loading
        progress_callback: Optional callback(current, total, message)

    Returns:
        dict: Face embeddings dictionary {face_path: {'embedding': emb, 'source_path': original_path}}
    """
    os.makedirs(output_folder, exist_ok=True)

    # Initialize InsightFace
    print("Initializing face detection model...")
    app = initialize_face_analyzer()

    # Get all image files (Recursive)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    # Folders to exclude
    excluded_dirs = {'face_detection_output', 'all_images_processed', '__pycache__', '.git'}
    
    print(f"Scanning folder: {input_folder} (recursive)")
    
    for root, dirs, files in os.walk(input_folder):
        # Exclude directories in-place
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
        
        for f in files:
            if Path(f).suffix.lower() in valid_extensions:
                # Skip representative faces if re-scanning output
                if "_representative_face" in f:
                    continue
                image_files.append(os.path.join(root, f))

    total_images = len(image_files)
    print(f"Found {total_images} images to process")

    if total_images == 0:
        return {}

    embeddings_dict = {}
    processed_count = 0

    # Process in batches
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_paths = image_files[batch_start:batch_end]

        # Load images in parallel
        loaded_images = load_images_parallel(batch_paths, max_workers=max_workers)

        # Process each image in the batch
        for img_path, (processed_img, original_img, scale) in loaded_images.items():
            img_file = os.path.basename(img_path)
            # Create unique ID for this image based on path hash or relative path to handle duplicates in subfolders
            # We'll stick to a simple strategy: unique subfolder in output per processed image
            # But wait, we need face crops to be saved somewhere for representative face generation.
            
            # Simple hash for folder name to avoid collisions
            import hashlib
            rel_path_hash = hashlib.md5(img_path.encode('utf-8')).hexdigest()[:8]
            image_name = f"{Path(img_file).stem}_{rel_path_hash}"

            # Create output folder for this image's faces
            image_output_folder = os.path.join(output_folder, image_name)
            os.makedirs(image_output_folder, exist_ok=True)

            # NOTE: We NO LONGER save the full original image here to save space!
            # original_output_path = os.path.join(image_output_folder, "original_" + img_file)
            # cv2.imwrite(original_output_path, original_img)

            # Get image dimensions
            h, w = original_img.shape[:2]
            img_area = h * w

            # Detect faces and get embeddings (single pass!)
            # Convert to RGB for InsightFace
            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            faces = app.get(img_rgb)

            if not faces:
                # Clean up empty folder if created
                try:
                    os.rmdir(image_output_folder)
                except:
                    pass
                print(f"No faces detected in {img_file}")
                processed_count += 1
                continue

            # Scale bounding boxes back to original size if downsampled
            if scale != 1.0:
                for face in faces:
                    face.bbox = face.bbox / scale

            # Filter faces
            filtered_faces = filter_faces(
                faces, original_img, img_area,
                min_face_size=min_face_size,
                min_confidence=min_confidence,
                min_face_ratio=min_face_ratio,
                foreground_ratio_threshold=foreground_ratio_threshold,
                blur_threshold=blur_threshold
            )

            print(f"Kept {len(filtered_faces)} faces out of {len(faces)} detected in {img_file}")

            # Save face crops and store embeddings
            for i, face_info in enumerate(filtered_faces):
                bbox = face_info["bbox"].astype(int)
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # Add margin for face crop
                margin_h = int(0.4 * h)
                margin_w = int(0.3 * w)

                crop_y1 = max(0, y1 - margin_h)
                crop_y2 = min(original_img.shape[0], y2 + margin_h)
                crop_x1 = max(0, x1 - margin_w)
                crop_x2 = min(original_img.shape[1], x2 + margin_w)

                face_img = original_img[crop_y1:crop_y2, crop_x1:crop_x2]

                # Save face crop (still needed for UI thumbnails/representative faces)
                face_filename = f"face_{i + 1}_{img_file}"
                face_output_path = os.path.join(image_output_folder, face_filename)
                cv2.imwrite(face_output_path, face_img)

                # Store embedding AND source path in metadata
                # We store the source path so we can copy it later without duplicating now
                embeddings_dict[face_output_path] = {
                    'embedding': face_info["embedding"],
                    'source_path': img_path
                }

            processed_count += 1

            if progress_callback:
                progress_callback(processed_count, total_images, f"Processed {img_file}")

    # Save embeddings
    embeddings_file = os.path.join(output_folder, "face_embeddings_insightface.pkl")
    with open(embeddings_file, "wb") as f:
        pickle.dump(embeddings_dict, f)

    print(f"Processing complete. Created embeddings for {len(embeddings_dict)} faces.")
    print(f"Results saved to {output_folder}")

    return embeddings_dict


# ============================================================================
# Legacy Function Wrappers (for backward compatibility)
# ============================================================================

def detect_and_organize_faces_retina(input_folder, output_folder, min_face_size=80,
                                     min_confidence=0.8, min_face_ratio=0.03,
                                     foreground_ratio_threshold=0.4, blur_threshold=100):
    """
    Legacy wrapper - redirects to unified detect_and_embed_faces.
    Maintained for backward compatibility with existing code.
    """
    print("Note: Using optimized unified detection pipeline")
    return detect_and_embed_faces(
        input_folder, output_folder,
        min_face_size=min_face_size,
        min_confidence=min_confidence,
        min_face_ratio=min_face_ratio,
        foreground_ratio_threshold=foreground_ratio_threshold,
        blur_threshold=blur_threshold
    )


def create_face_embeddings(output_folder):
    """
    Legacy wrapper - embeddings are now created during detection.
    This function just loads and returns existing embeddings.
    """
    embeddings_file = os.path.join(output_folder, "face_embeddings_insightface.pkl")

    if os.path.exists(embeddings_file):
        print("Loading existing embeddings (created during detection)")
        with open(embeddings_file, "rb") as f:
            return pickle.load(f)

    print("Warning: No embeddings file found. Run detect_and_embed_faces first.")
    return {}


# ============================================================================
# Person Grouping and Organization
# ============================================================================

def reorganize_by_person(output_folder, input_folder=None, similarity_threshold=0.55, max_profile_angle=45, session_id=None):
    """
    Reorganizes the photos by person based on face embeddings from InsightFace.
    Uses improved handling for profile/side faces.
    Creates a special folder for group photos with more than 3 people.
    
    Args:
        output_folder (str): Path to the temp folder containing face crops/embeddings
        input_folder (str): Path to original input folder (where output will be created)
        similarity_threshold (float): Similarity threshold for matching faces (0-1)
        max_profile_angle (float): Maximum head pose angle to consider as same person
        session_id (str): Session ID to organize files under

    Returns:
        dict: A dictionary with person IDs as keys and lists of original image paths as values
        str: Path to the final output directory
    """
    print("Reorganizing photos by person using InsightFace embeddings with profile handling...")

    # Load embeddings
    embeddings_file = os.path.join(output_folder, "face_embeddings_insightface.pkl")
    if not os.path.exists(embeddings_file):
        print(f"InsightFace embeddings file not found: {embeddings_file}")
        return {}, ""

    with open(embeddings_file, "rb") as f:
        embeddings_dict = pickle.load(f)

    if not embeddings_dict:
        print("No embeddings found")
        return {}, ""

    # Check data format (handle legacy vs new format)
    first_val = next(iter(embeddings_dict.values()))
    is_new_format = isinstance(first_val, dict) and 'embedding' in first_val

    # Extract face paths and embeddings
    face_paths = list(embeddings_dict.keys())
    
    if is_new_format:
        face_embeddings = np.array([v['embedding'] for v in embeddings_dict.values()])
    else:
        face_embeddings = np.array(list(embeddings_dict.values()))

    # Create a persons dictionary
    persons = {}  # person_id -> [face_paths]
    person_counter = 0

    # First pass: Group faces by similarity
    for i, (face_path, embedding) in enumerate(zip(face_paths, face_embeddings)):
        print(f"Processing face {i + 1}/{len(face_paths)}")

        assigned_to_existing = False
        best_match_score = 0
        best_match_id = None

        for person_id, person_faces in persons.items():
            # Extract embeddings for this person
            if is_new_format:
                person_embeddings = np.array([embeddings_dict[f]['embedding'] for f in person_faces])
            else:
                person_embeddings = np.array([embeddings_dict[f] for f in person_faces])

            embedding_norm = embedding / np.linalg.norm(embedding)
            person_embeddings_norm = person_embeddings / np.linalg.norm(person_embeddings, axis=1, keepdims=True)

            similarities = np.dot(person_embeddings_norm, embedding_norm)

            if len(similarities) > 3:
                top_similarities = np.sort(similarities)[-3:]
                avg_similarity = np.mean(top_similarities)
            else:
                avg_similarity = np.max(similarities)

            if avg_similarity > best_match_score:
                best_match_score = avg_similarity
                best_match_id = person_id

            if avg_similarity > similarity_threshold:
                persons[person_id].append(face_path)
                assigned_to_existing = True
                print(f"  Assigned to existing person {person_id} (similarity: {avg_similarity:.3f})")
                break

        if not assigned_to_existing:
            profile_threshold = similarity_threshold * 0.9

            if best_match_score > profile_threshold and best_match_id is not None:
                persons[best_match_id].append(face_path)
                print(f"  Assigned to existing person {best_match_id} as possible profile (similarity: {best_match_score:.3f})")
            else:
                person_id = f"rename_{person_counter}"
                persons[person_id] = [face_path]
                person_counter += 1
                print(f"  Created new person {person_id}")

    print(f"Found {len(persons)} unique persons")

    # Second pass: Merge small clusters
    small_cluster_threshold = 2
    merge_happened = True

    while merge_happened:
        merge_happened = False

        small_clusters = {pid: faces for pid, faces in persons.items() if len(faces) <= small_cluster_threshold}
        large_clusters = {pid: faces for pid, faces in persons.items() if len(faces) > small_cluster_threshold}

        for small_pid, small_faces in small_clusters.items():
            if small_pid not in persons:
                continue

            best_match_score = 0
            best_match_pid = None

            for large_pid, large_faces in large_clusters.items():
                if small_pid == large_pid:
                    continue

                max_sim = 0
                for small_face in small_faces:
                    if is_new_format:
                        small_emb = embeddings_dict[small_face]['embedding']
                    else:
                        small_emb = embeddings_dict[small_face]
                        
                    small_emb_norm = small_emb / np.linalg.norm(small_emb)

                    for large_face in large_faces:
                        if is_new_format:
                            large_emb = embeddings_dict[large_face]['embedding']
                        else:
                            large_emb = embeddings_dict[large_face]
                            
                        large_emb_norm = large_emb / np.linalg.norm(large_emb)

                        sim = np.dot(small_emb_norm, large_emb_norm)
                        max_sim = max(max_sim, sim)

                if max_sim > best_match_score:
                    best_match_score = max_sim
                    best_match_pid = large_pid

            profile_merge_threshold = similarity_threshold * 0.85
            if best_match_score > profile_merge_threshold and best_match_pid is not None:
                print(f"Merging small cluster {small_pid} into {best_match_pid} (similarity: {best_match_score:.3f})")
                persons[best_match_pid].extend(persons[small_pid])
                del persons[small_pid]
                merge_happened = True
                break

    # Map face paths to original images
    face_to_original = {}
    
    if is_new_format:
        # Fast path: use metadata
        for face_path in face_paths:
            face_to_original[face_path] = embeddings_dict[face_path]['source_path']
    else:
        # Legacy path: look for files
        for face_path in face_paths:
            image_folder = os.path.dirname(face_path)
            original_files = glob(os.path.join(image_folder, "original_*"))
            if original_files:
                face_to_original[face_path] = original_files[0]

    # Create output directory
    # If input_folder provided, put results there. Else fallback to CWD.
    base_dir = input_folder if input_folder else "."
    
    if session_id:
        processed_folder = os.path.join(base_dir, "all_images_processed", session_id)
    else:
        processed_folder = os.path.join(base_dir, "all_images_processed")

    os.makedirs(processed_folder, exist_ok=True)

    # Group photos folder
    group_photos_folder = os.path.join(processed_folder, "all_group_photos")
    os.makedirs(group_photos_folder, exist_ok=True)

    # Track images with multiple people (based on face count, not just unique identities)
    image_face_counts = defaultdict(int)
    for face_path in face_paths:
        if face_path in face_to_original:
            image_face_counts[face_to_original[face_path]] += 1

    group_photos = {img_path: count for img_path, count in image_face_counts.items() if count > 3}
    print(f"Found {len(group_photos)} images with more than 3 detected faces")

    # Create person folders and copy images
    persons_to_originals = {}

    for person_id, person_face_paths in persons.items():
        original_paths = set()
        for face_path in person_face_paths:
            if face_path in face_to_original:
                original_paths.add(face_to_original[face_path])

        persons_to_originals[person_id] = list(original_paths)

        person_folder = os.path.join(processed_folder, person_id)
        os.makedirs(person_folder, exist_ok=True)

        # Representative face
        if person_face_paths:
            # We want the biggest/best face for the thumbnail
            # We can use file size of the crop as a proxy for quality/size
            try:
                rep_face_path = max(person_face_paths, key=lambda f: os.path.getsize(f))
                rep_face_dest = os.path.join(person_folder, f"{person_id}_representative_face.jpg")
                shutil.copy2(rep_face_path, rep_face_dest)
            except Exception as e:
                print(f"Error creating representative face: {e}")

        # Copy original images
        for original_path in original_paths:
            original_name = os.path.basename(original_path)
            destination = os.path.join(person_folder, original_name)
            
            # Avoid overwriting if multiple people in same folder structure (though we used unique IDs)
            # Just copy
            try:
                shutil.copy2(original_path, destination)
            except Exception as e:
                print(f"Error copying {original_path}: {e}")

        print(f"Copied {len(original_paths)} images to folder {person_id}")

    # Handle group photos
    group_readme_path = os.path.join(group_photos_folder, "README.txt")
    with open(group_readme_path, "w") as f:
        f.write("GROUP PHOTOS\n=============\n\n")
        f.write("This folder contains photos where more than 3 faces were detected.\n")

    for original_path in group_photos.keys():
        original_name = os.path.basename(original_path)
        destination = os.path.join(group_photos_folder, original_name)
        try:
            shutil.copy2(original_path, destination)
        except Exception as e:
            print(f"Error copying group photo {original_path}: {e}")

    # Save metadata
    persons_file = os.path.join(processed_folder, "persons_to_originals.pkl")
    with open(persons_file, "wb") as f:
        pickle.dump(persons_to_originals, f)

    readme_path = os.path.join(processed_folder, "README.txt")
    with open(readme_path, "w") as f:
        f.write("PHOTO ORGANIZATION BY PERSON\n")
        f.write("===========================\n\n")
        f.write("Each folder contains:\n")
        f.write("1. A representative face image showing who this folder is about\n")
        f.write("2. All original photos where this person appears\n\n")
        f.write("The 'all_group_photos' folder contains photos with more than 3 people.\n")
        f.write("Folders are named 'rename_X' - rename them to actual person names.\n")

    print(f"All images processed and organized at {processed_folder}")
    return persons_to_originals, processed_folder


def clean_filenames(processed_folder, session_id=None):
    """
    Removes the 'original_' prefix from all filenames in the organized folders.
    (Legacy function: mostly unused now as we don't add prefix anymore, 
    but kept for backward compatibility with old outputs)
    """
    if session_id:
        target_path = os.path.join(processed_folder, session_id)
    else:
        target_path = processed_folder

    print(f"Cleaning filenames in: {target_path}")

    if not os.path.exists(target_path):
        print(f"Warning: Directory {target_path} does not exist")
        return 0

    renamed_count = 0

    for folder_name in os.listdir(target_path):
        folder_path = os.path.join(target_path, folder_name)

        if not os.path.isdir(folder_path) or folder_name == "all_group_photos":
            continue

        for filename in os.listdir(folder_path):
            if not os.path.isfile(os.path.join(folder_path, filename)):
                continue

            if "representative_face" in filename or "alt_" in filename:
                continue

            if filename.startswith("original_"):
                old_path = os.path.join(folder_path, filename)
                new_filename = filename[9:]
                new_path = os.path.join(folder_path, new_filename)

                try:
                    if os.path.exists(new_path):
                        # Target already exists, just remove the original_ prefixed duplicate
                        os.remove(old_path)
                    else:
                        os.rename(old_path, new_path)
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {filename}: {str(e)}")

    print(f"Renamed {renamed_count} files")
    return renamed_count


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Process photos for face detection and organization.')
    parser.add_argument('input_folder', type=str, help='Folder containing your images')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = os.path.join(input_folder, 'face_detection_output')

    # Single unified detection + embedding pass
    detect_and_embed_faces(
        input_folder,
        output_folder,
        min_face_size=80,
        min_confidence=0.8,
        min_face_ratio=0.01,
        foreground_ratio_threshold=0.1,
        blur_threshold=60,
        batch_size=args.batch_size,
        max_workers=args.workers
    )

    # Reorganize by person
    reorganize_by_person(output_folder, input_folder=input_folder, similarity_threshold=0.5)

    # Clean up filenames
    clean_filenames(os.path.join(input_folder, "all_images_processed"))


if __name__ == "__main__":
    main()
