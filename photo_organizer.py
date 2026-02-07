"""
Sort Moments - Desktop Application
A modern PyQt6 GUI application for organizing photos by detected faces.
"""

import os
import sys
import time
import shutil
import json
import logging
import traceback
import webbrowser
import urllib.parse
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QScrollArea,
    QFrame, QGridLayout, QMessageBox, QGraphicsDropShadowEffect,
    QSizePolicy, QLineEdit, QDialog, QStackedWidget, QToolButton,
    QSpacerItem, QLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QRect, QPoint, QRunnable, QThreadPool, QObject
from PyQt6.QtGui import QPixmap, QFont, QColor, QPalette, QIcon, QCursor, QImage

# Import the processing functions
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from processphotos import (
        detect_and_organize_faces_retina,
        create_face_embeddings,
        reorganize_by_person,
        clean_filenames,
    )
    PROCESSING_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    PROCESSING_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Modern Color Palette
COLORS = {
    'bg_primary': '#09090b',
    'bg_secondary': '#18181b',
    'bg_tertiary': '#27272a',
    'bg_card': '#1c1c1f',
    'bg_hover': '#2a2a2e',
    'accent': '#8b5cf6',
    'accent_hover': '#a78bfa',
    'accent_light': '#7c3aed',
    'success': '#10b981',
    'success_hover': '#34d399',
    'danger': '#ef4444',
    'text_primary': '#fafafa',
    'text_secondary': '#a1a1aa',
    'text_muted': '#71717a',
    'border': '#27272a',
    'border_light': '#3f3f46',
    'warning': '#f59e0b',
}

# ============================================================================
# Logging Configuration
# ============================================================================

def get_log_file_path():
    """Get the path for the error log file."""
    # Store logs in user's app data directory
    if sys.platform == 'win32':
        app_data = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
        log_dir = os.path.join(app_data, 'SortMoments', 'logs')
    else:
        log_dir = os.path.join(os.path.expanduser('~'), '.sortmoments', 'logs')

    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, 'sort_moments_error.log')

def setup_logging():
    """Setup logging to capture errors to a file."""
    log_file = get_log_file_path()

    # Create a custom logger
    logger = logging.getLogger('SortMoments')
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler - captures all logs
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Format with timestamp and level
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Initialize the logger
APP_LOGGER = setup_logging()

# Email configuration for error reports
ERROR_REPORT_EMAIL = "abdullahlogos@gmail.com"

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg_primary']};
}}

QWidget {{
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    color: {COLORS['text_primary']};
}}

QLabel {{
    color: {COLORS['text_primary']};
    background: transparent;
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}

QScrollBar:vertical {{
    background-color: {COLORS['bg_secondary']};
    width: 10px;
    border-radius: 5px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['border_light']};
    border-radius: 5px;
    min-height: 40px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['text_muted']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['bg_secondary']};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['border_light']};
    border-radius: 5px;
    min-width: 40px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

QProgressBar {{
    background-color: {COLORS['bg_tertiary']};
    border: none;
    border-radius: 6px;
    height: 6px;
    text-align: center;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, stop:1 {COLORS['accent_hover']});
    border-radius: 6px;
}}

QLineEdit {{
    background-color: {COLORS['bg_tertiary']};
    border: 2px solid {COLORS['border']};
    border-radius: 10px;
    padding: 10px 14px;
    color: {COLORS['text_primary']};
    font-size: 13px;
    selection-background-color: {COLORS['accent']};
}}

QLineEdit:focus {{
    border-color: {COLORS['accent']};
}}

QMessageBox {{
    background-color: {COLORS['bg_secondary']};
}}

QMessageBox QLabel {{
    color: {COLORS['text_primary']};
}}

QMessageBox QPushButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    padding: 10px 28px;
    border-radius: 8px;
    font-weight: 600;
    min-width: 80px;
}}

QMessageBox QPushButton:hover {{
    background-color: {COLORS['accent_hover']};
}}
"""

# ============================================================================
# Background Thumbnail Worker
# ============================================================================

class WorkerSignals(QObject):
    """Signals for the ThumbnailWorker."""
    result = pyqtSignal(str, QImage)  # path, image

class ThumbnailWorker(QRunnable):
    """Worker to load and resize images in a background thread."""
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            # Load image into QImage (thread-safe, unlike QPixmap)
            image = QImage(self.image_path)
            if not image.isNull():
                # Resize here in the background thread!
                # Target size is 160x160 (thumbnail widget size)
                # Using FastTransformation for speed, or SmoothTransformation for quality
                scaled = image.scaled(
                    160, 160,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Crop center
                x = (scaled.width() - 160) // 2
                y = (scaled.height() - 160) // 2
                cropped = scaled.copy(x, y, 160, 160)
                
                self.signals.result.emit(self.image_path, cropped)
        except Exception:
            pass


class FlowLayout(QLayout):
    """A flow layout that arranges widgets in rows, wrapping to new rows as needed."""

    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def spacing(self):
        if self._spacing >= 0:
            return self._spacing
        return 20

    def setSpacing(self, spacing):
        self._spacing = spacing

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only):
        margins = self.contentsMargins()
        effective_rect = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._items:
            widget = item.widget()
            space_x = spacing
            space_y = spacing

            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y() + margins.bottom()


class ProcessingThread(QThread):
    """Background thread for photo processing with smooth progress updates."""

    status_update = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    finished_success = pyqtSignal(str, int)
    finished_error = pyqtSignal(str)
    finished_stopped = pyqtSignal()

    def __init__(self, input_folder):
        super().__init__()
        self.input_folder = input_folder
        self._current_progress = 0
        self._target_progress = 0
        self._stop_requested = False

    def request_stop(self):
        """Request the thread to stop processing."""
        self._stop_requested = True

    def smooth_progress(self, start, end, duration_hint=1.0):
        """Emit progress updates smoothly between start and end values."""
        steps = max(5, (end - start) // 2)
        delay = duration_hint / steps
        for i in range(steps):
            progress = start + int((end - start) * (i + 1) / steps)
            self.progress_update.emit(progress)
            time.sleep(delay)

    def run(self):
        # Log the start of processing
        APP_LOGGER.info(f"Starting processing for folder: {self.input_folder}")

        try:
            input_folder = self.input_folder

            # Step 1: Face Detection + Embedding (0-75%) - Unified pass, GPU accelerated
            if self._stop_requested:
                APP_LOGGER.info("Processing stopped by user before Step 1")
                self.finished_stopped.emit()
                return

            APP_LOGGER.info("Step 1: Detecting faces and creating embeddings...")
            self.status_update.emit("Detecting faces and creating embeddings...")
            self.progress_update.emit(2)

            output_folder = os.path.join(input_folder, 'face_detection_output')
            os.makedirs(output_folder, exist_ok=True)

            self.progress_update.emit(5)

            # Use the new unified detection + embedding function
            # This does detection AND embedding in a single pass with GPU acceleration
            from processphotos import detect_and_embed_faces
            detect_and_embed_faces(
                input_folder, output_folder,
                min_face_size=50, min_confidence=0.7,
                min_face_ratio=0.005, foreground_ratio_threshold=0.05,
                blur_threshold=50,
                batch_size=8, max_workers=4
            )

            APP_LOGGER.info("Step 1 completed: Face detection and embeddings done")
            self.smooth_progress(5, 75, 0.5)

            # Step 2: Grouping (75-95%) - Clustering faces by person
            if self._stop_requested:
                APP_LOGGER.info("Processing stopped by user before Step 2")
                self.finished_stopped.emit()
                return

            APP_LOGGER.info("Step 2: Grouping photos by person...")
            self.status_update.emit("Grouping photos by person...")
            self.progress_update.emit(76)

            persons_dict, processed_folder = reorganize_by_person(output_folder, input_folder=input_folder, similarity_threshold=0.5)

            APP_LOGGER.info(f"Step 2 completed: Grouped into {len(persons_dict) if persons_dict else 0} persons")
            self.smooth_progress(76, 95, 0.3)

            # Step 3: Cleanup (95-100%) - Very quick
            if self._stop_requested:
                APP_LOGGER.info("Processing stopped by user before Step 3")
                self.finished_stopped.emit()
                return

            APP_LOGGER.info("Step 3: Finalizing and cleaning up...")
            self.status_update.emit("Finalizing and cleaning up...")
            self.progress_update.emit(96)

            clean_filenames(processed_folder)

            APP_LOGGER.info("Step 3 completed: Cleanup done")
            self.smooth_progress(96, 99, 0.2)

            # Step 4: Delete intermediate face detection folder
            if os.path.exists(output_folder):
                try:
                    APP_LOGGER.info(f"Removing intermediate folder: {output_folder}")
                    shutil.rmtree(output_folder)
                except Exception as e:
                    APP_LOGGER.warning(f"Failed to remove intermediate folder: {e}")

            if self._stop_requested:
                APP_LOGGER.info("Processing stopped by user after Step 3")
                self.finished_stopped.emit()
                return

            self.progress_update.emit(100)
            person_count = len(persons_dict) if persons_dict else 0
            APP_LOGGER.info(f"Processing completed successfully. Found {person_count} persons.")
            self.finished_success.emit(processed_folder, person_count)

        except Exception as e:
            if not self._stop_requested:
                # Log the full error with traceback
                error_msg = str(e)
                full_traceback = traceback.format_exc()
                APP_LOGGER.error(f"Processing failed with error: {error_msg}")
                APP_LOGGER.error(f"Full traceback:\n{full_traceback}")
                self.finished_error.emit(error_msg)


class ModernButton(QPushButton):
    """Modern styled button with hover effects."""

    def __init__(self, text, variant="primary", parent=None):
        super().__init__(text, parent)
        self.variant = variant
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        self.apply_style()

    def apply_style(self):
        styles = {
            "primary": f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {COLORS['accent']}, stop:1 {COLORS['accent_light']});
                    color: white;
                    border: none;
                    padding: 14px 32px;
                    border-radius: 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {COLORS['accent_hover']}, stop:1 {COLORS['accent']});
                }}
                QPushButton:disabled {{
                    background: {COLORS['bg_tertiary']};
                    color: {COLORS['text_muted']};
                }}
            """,
            "success": f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {COLORS['success']}, stop:1 #059669);
                    color: white;
                    border: none;
                    padding: 14px 32px;
                    border-radius: 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {COLORS['success_hover']}, stop:1 {COLORS['success']});
                }}
                QPushButton:disabled {{
                    background: {COLORS['bg_tertiary']};
                    color: {COLORS['text_muted']};
                }}
            """,
            "secondary": f"""
                QPushButton {{
                    background: {COLORS['bg_tertiary']};
                    color: {COLORS['text_primary']};
                    border: 1px solid {COLORS['border_light']};
                    padding: 14px 32px;
                    border-radius: 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: {COLORS['bg_hover']};
                    border-color: {COLORS['text_muted']};
                }}
                QPushButton:disabled {{
                    background: {COLORS['bg_secondary']};
                    color: {COLORS['text_muted']};
                }}
            """,
            "ghost": f"""
                QPushButton {{
                    background: transparent;
                    color: {COLORS['text_secondary']};
                    border: none;
                    padding: 10px 16px;
                    border-radius: 8px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background: {COLORS['bg_tertiary']};
                    color: {COLORS['text_primary']};
                }}
            """,
            "icon": f"""
                QPushButton {{
                    background: {COLORS['bg_tertiary']};
                    color: {COLORS['text_secondary']};
                    border: 1px solid {COLORS['border']};
                    padding: 10px;
                    border-radius: 10px;
                    font-size: 16px;
                }}
                QPushButton:hover {{
                    background: {COLORS['bg_hover']};
                    color: {COLORS['text_primary']};
                    border-color: {COLORS['border_light']};
                }}
            """,
            "danger": f"""
                QPushButton {{
                    background: {COLORS['danger']};
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 10px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: #dc2626;
                }}
            """,
            "warning": f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 {COLORS['warning']}, stop:1 #d97706);
                    color: white;
                    border: none;
                    padding: 14px 32px;
                    border-radius: 12px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #fbbf24, stop:1 {COLORS['warning']});
                }}
                QPushButton:disabled {{
                    background: {COLORS['bg_tertiary']};
                    color: {COLORS['text_muted']};
                }}
            """
        }
        self.setStyleSheet(styles.get(self.variant, styles["primary"]))


class ImageViewer(QDialog):
    """Full-screen image viewer dialog."""

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Viewer")
        self.setModal(True)
        self.setStyleSheet(f"background-color: {COLORS['bg_primary']};")
        self.resize(1000, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Top bar
        top_bar = QFrame()
        top_bar.setStyleSheet(f"background-color: {COLORS['bg_secondary']}; padding: 8px;")
        top_bar.setFixedHeight(50)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 0, 16, 0)

        filename = QLabel(os.path.basename(image_path))
        filename.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        top_layout.addWidget(filename)

        top_layout.addStretch()

        close_btn = ModernButton("✕", variant="ghost")
        close_btn.setFixedSize(40, 40)
        close_btn.clicked.connect(self.close)
        top_layout.addWidget(close_btn)

        layout.addWidget(top_bar)

        # Image
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet("background: transparent; border: none;")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.width() - 40, self.height() - 100,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)

        scroll.setWidget(self.image_label)
        layout.addWidget(scroll)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()


class ImageThumbnail(QFrame):
    """Clickable image thumbnail."""

    clicked = pyqtSignal(str)

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(160, 160)
        self.setStyleSheet(f"""
            ImageThumbnail {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 12px;
                border: 2px solid transparent;
            }}
            ImageThumbnail:hover {{
                border-color: {COLORS['accent']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet("border-radius: 10px;")
        
        # Start with placeholder
        self.thumb_label.setText("Loading...")
        self.thumb_label.setStyleSheet(f"color: {COLORS['text_muted']};")

        layout.addWidget(self.thumb_label)

    def set_image(self, qimage):
        """Update with actual image from background thread."""
        if not qimage.isNull():
            # Convert to pixmap on UI thread (fast)
            self.thumb_label.setPixmap(QPixmap.fromImage(qimage))
            self.thumb_label.setText("")
        else:
            self.thumb_label.setText("Error")

    def mousePressEvent(self, event):
        self.clicked.emit(self.image_path)


class FolderView(QWidget):
    """View for displaying contents of a person folder."""

    back_clicked = pyqtSignal()
    folder_renamed = pyqtSignal(str, str)  # old_path, new_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder_path = None
        self.folder_name = None
        self.thread_pool = QThreadPool()
        # Limit threads to avoid disk thrashing
        self.thread_pool.setMaxThreadCount(4) 
        self.active_thumbnails = {}  # path -> widget
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # Header
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 16px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 16)
        header_layout.setSpacing(16)

        # Back button
        back_btn = ModernButton("←", variant="icon")
        back_btn.setFixedSize(44, 44)
        back_btn.setFont(QFont("Segoe UI", 16))
        back_btn.clicked.connect(self.back_clicked.emit)
        header_layout.addWidget(back_btn)

        # Folder icon and name
        self.folder_icon = QLabel("👤")
        self.folder_icon.setFont(QFont("Segoe UI", 24))
        header_layout.addWidget(self.folder_icon)

        # Editable name
        name_container = QVBoxLayout()
        name_container.setSpacing(4)

        self.name_input = QLineEdit()
        self.name_input.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                background: transparent;
                border: 2px solid transparent;
                border-radius: 8px;
                padding: 6px 12px;
                color: {COLORS['text_primary']};
                font-size: 16px;
            }}
            QLineEdit:hover {{
                background: {COLORS['bg_tertiary']};
            }}
            QLineEdit:focus {{
                background: {COLORS['bg_tertiary']};
                border-color: {COLORS['accent']};
            }}
        """)
        self.name_input.returnPressed.connect(self.rename_folder)
        name_container.addWidget(self.name_input)

        self.image_count_label = QLabel()
        self.image_count_label.setFont(QFont("Segoe UI", 11))
        self.image_count_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        name_container.addWidget(self.image_count_label)

        header_layout.addLayout(name_container, stretch=1)

        # Action buttons
        self.rename_btn = ModernButton("Rename", variant="secondary")
        self.rename_btn.clicked.connect(self.rename_folder)
        header_layout.addWidget(self.rename_btn)

        self.open_external_btn = ModernButton("Open in Explorer", variant="secondary")
        self.open_external_btn.clicked.connect(self.open_in_explorer)
        header_layout.addWidget(self.open_external_btn)

        layout.addWidget(header)

        # Images grid - with FlowLayout for proper expanding
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background: transparent;")
        self.grid_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.grid_layout = FlowLayout(self.grid_widget, margin=0, spacing=16)

        self.scroll_area.setWidget(self.grid_widget)
        layout.addWidget(self.scroll_area, stretch=1)
        
        # Connect scroll signal for lazy loading
        # Check on scroll
        self.scroll_area.verticalScrollBar().valueChanged.connect(self.check_scroll_position)
        # Check when range changes (e.g. window resize or content added)
        self.scroll_area.verticalScrollBar().rangeChanged.connect(self.check_scroll_range)
        
        # Batch loading state
        self.all_image_paths = []
        self.loaded_count = 0
        self.batch_size = 50
        self.is_loading_batch = False

    def check_scroll_position(self, value):
        """Load more images when scrolled near bottom."""
        scrollbar = self.scroll_area.verticalScrollBar()
        # If we are near the bottom (80%), load more
        if not self.is_loading_batch and value >= scrollbar.maximum() * 0.8:
            self.load_next_batch()

    def check_scroll_range(self, min_val, max_val):
        """Check if we need to load more when content/size changes."""
        # If scrollbar is missing (max=0) or we are still near bottom, load more
        if not self.is_loading_batch:
            current_val = self.scroll_area.verticalScrollBar().value()
            if max_val <= 0 or current_val >= max_val * 0.8:
                self.load_next_batch()

    def clear_content(self):
        """Immediately clear all thumbnails from the view."""
        # Cancel pending tasks if possible (though QThreadPool doesn't support easy cancellation)
        self.active_thumbnails.clear()
        
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()  # Hide immediately
                widget.setParent(None)  # Remove from parent
                widget.deleteLater()
        # Force immediate visual update
        self.grid_widget.update()
        QApplication.processEvents()

    def load_folder(self, folder_path, folder_name):
        self.clear_content()
        self.folder_path = folder_path
        self.folder_name = folder_name
        
        # Reset batch state
        self.all_image_paths = []
        self.loaded_count = 0

        # Set display name
        display_name = folder_name
        if folder_name.startswith("rename_"):
            display_name = f"Person {folder_name.split('_')[1]}"
        elif folder_name == "all_group_photos":
            display_name = "Group Photos"
            self.folder_icon.setText("👥")
        else:
            self.folder_icon.setText("👤")

        self.name_input.setText(display_name)

        # 1. Fast List: Get all file paths instantly
        try:
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    if not f.endswith('_representative_face.jpg'):
                        self.all_image_paths.append(os.path.join(folder_path, f))
        except Exception:
            pass

        self.image_count_label.setText(f"{len(self.all_image_paths)} photos")
        
        # 2. Start initial batch
        self.load_next_batch()

    def on_thumbnail_ready(self, path, image):
        """Callback when a background worker finishes resizing an image."""
        if path in self.active_thumbnails:
            # This is safe because signals slot into the UI thread automatically
            self.active_thumbnails[path].set_image(image)

    def load_next_batch(self):
        """Load the next batch of images."""
        if self.loaded_count >= len(self.all_image_paths):
            return
            
        self.is_loading_batch = True
        
        end_index = min(self.loaded_count + self.batch_size, len(self.all_image_paths))
        batch_paths = self.all_image_paths[self.loaded_count:end_index]
        
        for img_path in batch_paths:
            thumb = ImageThumbnail(img_path)
            thumb.clicked.connect(self.show_image)
            self.grid_layout.addWidget(thumb)
            self.active_thumbnails[img_path] = thumb
            
            # Queue Worker: Start background loading immediately
            worker = ThumbnailWorker(img_path)
            worker.signals.result.connect(self.on_thumbnail_ready)
            self.thread_pool.start(worker)
            
        self.loaded_count = end_index
        self.is_loading_batch = False

        # Force layout update
        self.grid_layout.invalidate()
        self.grid_widget.adjustSize()
        self.grid_widget.updateGeometry()
        
        # Check if we need to load more immediately (if screen isn't full)
        # We use a timer to allow the layout to actually update/repaint first
        QTimer.singleShot(50, lambda: self.check_scroll_range(0, self.scroll_area.verticalScrollBar().maximum()))

    def show_image(self, image_path):
        # Open in default OS viewer
        try:
            if sys.platform == "win32":
                os.startfile(image_path)
            elif sys.platform == "darwin":
                os.system(f'open "{image_path}"')
            else:
                os.system(f'xdg-open "{image_path}"')
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open image: {str(e)}")

    def rename_folder(self):
        if not self.folder_path:
            return

        new_name = self.name_input.text().strip()
        if not new_name:
            return

        # Sanitize name for filesystem
        safe_name = "".join(c for c in new_name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not safe_name:
            return

        parent_dir = os.path.dirname(self.folder_path)
        new_path = os.path.join(parent_dir, safe_name)

        if new_path == self.folder_path:
            return

        if os.path.exists(new_path):
            QMessageBox.warning(self, "Error", "A folder with this name already exists.")
            return

        try:
            old_folder_name = self.folder_name
            os.rename(self.folder_path, new_path)

            # Also rename the representative face file to match new folder name
            old_rep_face = os.path.join(new_path, f"{old_folder_name}_representative_face.jpg")
            new_rep_face = os.path.join(new_path, f"{safe_name}_representative_face.jpg")
            if os.path.exists(old_rep_face):
                os.rename(old_rep_face, new_rep_face)

            old_path = self.folder_path
            self.folder_path = new_path
            self.folder_name = safe_name
            self.folder_renamed.emit(old_path, new_path)
            QMessageBox.information(self, "Success", f"Folder renamed to '{safe_name}'")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to rename folder: {e}")

    def open_in_explorer(self):
        if self.folder_path and os.path.exists(self.folder_path):
            if sys.platform == "win32":
                os.startfile(self.folder_path)
            elif sys.platform == "darwin":
                os.system(f'open "{self.folder_path}"')
            else:
                os.system(f'xdg-open "{self.folder_path}"')

    def showEvent(self, event):
        """Force layout recalculation when view becomes visible."""
        super().showEvent(event)
        # Defer the update to ensure geometry is correct
        QTimer.singleShot(0, self._update_layout)

    def _update_layout(self):
        """Update FlowLayout after view is shown."""
        if self.grid_layout.count() > 0:
            # Force recalculation with actual viewport width
            viewport_width = self.scroll_area.viewport().width()
            if viewport_width > 100:  # Only if we have a reasonable width
                rect = QRect(0, 0, viewport_width, 0)
                self.grid_layout.setGeometry(rect)
            self.grid_layout.invalidate()
            self.grid_widget.adjustSize()
            self.grid_widget.updateGeometry()
            self.scroll_area.updateGeometry()


class PersonCard(QFrame):
    """Modern card widget for displaying a person folder."""

    clicked = pyqtSignal(str, str)  # folder_name, folder_path

    def __init__(self, folder_name, folder_path, image_count, parent=None):
        super().__init__(parent)
        self.folder_path = folder_path
        self.folder_name = folder_name
        self.image_count = image_count

        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setFixedSize(200, 260)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            PersonCard {{
                background-color: {COLORS['bg_card']};
                border-radius: 16px;
                border: 1px solid {COLORS['border']};
            }}
            PersonCard:hover {{
                border-color: {COLORS['accent']};
                background-color: {COLORS['bg_hover']};
            }}
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Thumbnail container
        thumb_container = QFrame()
        thumb_container.setFixedSize(176, 150)
        thumb_container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 12px;
            }}
        """)

        self.thumbnail_label = QLabel(thumb_container)
        self.thumbnail_label.setFixedSize(176, 150)
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setStyleSheet("border-radius: 12px;")
        self.load_thumbnail()

        layout.addWidget(thumb_container)

        # Info
        info = QVBoxLayout()
        info.setSpacing(4)

        display_name = folder_name
        if folder_name.startswith("rename_"):
            display_name = f"Person {folder_name.split('_')[1]}"
        elif folder_name == "all_group_photos":
            display_name = "Group Photos"

        name_label = QLabel(display_name)
        name_label.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        name_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        info.addWidget(name_label)

        count_label = QLabel(f"{self.image_count} photos")
        count_label.setFont(QFont("Segoe UI", 11))
        count_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        info.addWidget(count_label)

        layout.addLayout(info)

    def load_thumbnail(self):
        rep_face_path = os.path.join(self.folder_path, f"{self.folder_name}_representative_face.jpg")

        pixmap = None
        if os.path.exists(rep_face_path):
            pixmap = QPixmap(rep_face_path)

        if pixmap is None or pixmap.isNull():
            for f in os.listdir(self.folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    pixmap = QPixmap(os.path.join(self.folder_path, f))
                    if not pixmap.isNull():
                        break

        if pixmap and not pixmap.isNull():
            scaled = pixmap.scaled(
                176, 150,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (scaled.width() - 176) // 2
            y = (scaled.height() - 150) // 2
            cropped = scaled.copy(max(0, x), max(0, y), 176, 150)
            self.thumbnail_label.setPixmap(cropped)
        else:
            self.thumbnail_label.setText("No Preview")
            self.thumbnail_label.setStyleSheet(f"color: {COLORS['text_muted']};")

    def mousePressEvent(self, event):
        self.clicked.emit(self.folder_name, self.folder_path)


class DropZone(QFrame):
    """Modern drag-and-drop zone."""

    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(200)
        self.is_hover = False
        self.update_style()

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)

        icon = QLabel("📁")
        icon.setFont(QFont("Segoe UI", 56))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        main_text = QLabel("Drop folder here or click to browse")
        main_text.setFont(QFont("Segoe UI", 15, QFont.Weight.DemiBold))
        main_text.setStyleSheet(f"color: {COLORS['text_primary']};")
        main_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(main_text)

        sub_text = QLabel("Supports JPG, PNG, BMP images")
        sub_text.setFont(QFont("Segoe UI", 11))
        sub_text.setStyleSheet(f"color: {COLORS['text_muted']};")
        sub_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub_text)

    def update_style(self):
        border = COLORS['accent'] if self.is_hover else COLORS['border_light']
        bg = COLORS['bg_tertiary'] if self.is_hover else COLORS['bg_secondary']
        self.setStyleSheet(f"""
            DropZone {{
                background-color: {bg};
                border: 2px dashed {border};
                border-radius: 20px;
            }}
        """)

    def mousePressEvent(self, event):
        self.clicked.emit()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.is_hover = True
            self.update_style()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.is_hover = False
        self.update_style()

    def dropEvent(self, event):
        self.is_hover = False
        self.update_style()
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                # Find main window and set folder
                parent = self.parent()
                while parent:
                    if hasattr(parent, 'set_folder'):
                        parent.set_folder(path)
                        break
                    parent = parent.parent()


class AnimatedProgressBar(QFrame):
    """Progress bar with animated indeterminate mode and smooth transitions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(8)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_tertiary']};
                border-radius: 4px;
            }}
        """)

        self._progress = 0
        self._target_progress = 0
        self._is_animating = False

        # Inner progress bar
        self.bar = QFrame(self)
        self.bar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {COLORS['accent']}, stop:1 {COLORS['accent_hover']});
                border-radius: 4px;
            }}
        """)
        self.bar.setGeometry(0, 0, 0, 8)

        # Animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)

    def set_progress(self, value):
        """Set progress value (0-100) with smooth animation."""
        self._target_progress = max(0, min(100, value))
        if not self._is_animating:
            self._is_animating = True
            self.anim_timer.start(16)  # ~60fps

    def _animate(self):
        """Animate progress bar smoothly."""
        diff = self._target_progress - self._progress
        if abs(diff) < 0.5:
            self._progress = self._target_progress
            self._is_animating = False
            self.anim_timer.stop()
        else:
            # Ease out animation
            self._progress += diff * 0.15

        # Update bar width
        width = int((self._progress / 100) * self.width())
        self.bar.setGeometry(0, 0, width, 8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        width = int((self._progress / 100) * self.width())
        self.bar.setGeometry(0, 0, width, 8)


class StatusCard(QFrame):
    """Modern status/progress card with animated progress and rotating status text."""

    # Dynamic phrases that cycle during each processing step (3 steps with unified pipeline)
    STEP_PHRASES = {
        1: ["Scanning images...", "Finding faces...", "Creating embeddings...", "Analyzing photos...", "Processing faces...", "Detecting features..."],
        2: ["Grouping by person...", "Matching faces...", "Organizing photos...", "Sorting by identity...", "Building groups..."],
        3: ["Cleaning up...", "Finalizing results...", "Organizing folders...", "Almost done...", "Finishing up..."],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            StatusCard {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 16px;
                border: 1px solid {COLORS['border']};
            }}
        """)

        self._current_step = 0
        self._phrase_index = 0
        self._is_processing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(14)

        header = QHBoxLayout()

        self.status_icon = QLabel("⏳")
        self.status_icon.setFont(QFont("Segoe UI", 22))
        header.addWidget(self.status_icon)

        self.status_label = QLabel("Ready to process")
        self.status_label.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        self.status_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        header.addWidget(self.status_label)

        header.addStretch()

        self.step_label = QLabel("")
        self.step_label.setFont(QFont("Segoe UI", 11))
        self.step_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        header.addWidget(self.step_label)

        self.percent_label = QLabel("")
        self.percent_label.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        self.percent_label.setStyleSheet(f"color: {COLORS['accent']};")
        self.percent_label.setFixedWidth(45)
        header.addWidget(self.percent_label)

        layout.addLayout(header)

        # Use animated progress bar
        self.progress_bar = AnimatedProgressBar()
        layout.addWidget(self.progress_bar)

        # Timer for rotating phrases
        self.phrase_timer = QTimer(self)
        self.phrase_timer.timeout.connect(self._rotate_phrase)

    def set_status(self, text, icon="⏳", step=""):
        self.status_icon.setText(icon)
        self.step_label.setText(step)

        # Determine current step from step string (3-step optimized pipeline)
        if "1/3" in step:
            self._current_step = 1
            self._start_phrase_rotation()
        elif "2/3" in step:
            self._current_step = 2
            self._start_phrase_rotation()
        elif "3/3" in step:
            self._current_step = 3
            self._start_phrase_rotation()
        else:
            self._stop_phrase_rotation()
            self.status_label.setText(text)

    def _start_phrase_rotation(self):
        if not self._is_processing:
            self._is_processing = True
            self._phrase_index = 0
            self._update_phrase()
            self.phrase_timer.start(2300)  # Rotate every 2.3 seconds

    def _stop_phrase_rotation(self):
        self._is_processing = False
        self.phrase_timer.stop()

    def _rotate_phrase(self):
        if self._is_processing and self._current_step in self.STEP_PHRASES:
            phrases = self.STEP_PHRASES[self._current_step]
            self._phrase_index = (self._phrase_index + 1) % len(phrases)
            self._update_phrase()

    def _update_phrase(self):
        if self._current_step in self.STEP_PHRASES:
            phrases = self.STEP_PHRASES[self._current_step]
            self.status_label.setText(phrases[self._phrase_index])

    def set_progress(self, value):
        self.progress_bar.set_progress(value)
        if value > 0 and value < 100:
            self.percent_label.setText(f"{value}%")
        else:
            self.percent_label.setText("")
            if value >= 100:
                self._stop_phrase_rotation()


class MainView(QWidget):
    """Main view with drop zone and results."""

    folder_selected = pyqtSignal(str, str)  # folder_name, folder_path
    input_folder_changed = pyqtSignal(str)  # Emitted when user selects an input folder

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_folder = None
        self.processed_folder = None
        self.processing_thread = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        # Top section - fixed height
        top_section = QHBoxLayout()
        top_section.setSpacing(24)

        # Left: Drop zone container
        drop_container = QVBoxLayout()
        drop_container.setSpacing(12)

        self.drop_zone = DropZone()
        self.drop_zone.clicked.connect(self.browse_folder)
        drop_container.addWidget(self.drop_zone)

        # Selected folder display
        self.folder_display = QFrame()
        self.folder_display.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 14px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        self.folder_display.setVisible(False)

        folder_layout = QHBoxLayout(self.folder_display)
        folder_layout.setContentsMargins(20, 14, 20, 14)
        folder_layout.setSpacing(12)

        folder_icon = QLabel("📂")
        folder_icon.setFont(QFont("Segoe UI", 18))
        folder_layout.addWidget(folder_icon)

        self.folder_path_label = QLabel("")
        self.folder_path_label.setFont(QFont("Segoe UI", 12))
        self.folder_path_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        folder_layout.addWidget(self.folder_path_label, stretch=1)

        self.image_count_label = QLabel("")
        self.image_count_label.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        self.image_count_label.setStyleSheet(f"color: {COLORS['accent']};")
        folder_layout.addWidget(self.image_count_label)

        change_btn = ModernButton("Change", variant="ghost")
        change_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(change_btn)

        drop_container.addWidget(self.folder_display)

        top_section.addLayout(drop_container, stretch=2)

        # Right: Actions panel
        actions_panel = QFrame()
        actions_panel.setFixedWidth(300)
        actions_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_secondary']};
                border-radius: 20px;
            }}
        """)

        actions_layout = QVBoxLayout(actions_panel)
        actions_layout.setContentsMargins(24, 24, 24, 24)
        actions_layout.setSpacing(16)

        actions_title = QLabel("Actions")
        actions_title.setFont(QFont("Segoe UI", 15, QFont.Weight.DemiBold))
        actions_title.setStyleSheet(f"color: {COLORS['text_primary']};")
        actions_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actions_layout.addWidget(actions_title)

        self.process_btn = ModernButton("Start Processing", variant="success")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.start_processing)
        actions_layout.addWidget(self.process_btn)

        self.stop_btn = ModernButton("Stop Processing", variant="danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        actions_layout.addWidget(self.stop_btn)

        self.open_output_btn = ModernButton("Open Output Folder", variant="secondary")
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.clicked.connect(self.open_output_folder)
        actions_layout.addWidget(self.open_output_btn)

        # Send error logs button - only visible after an error occurs
        self.send_logs_btn = ModernButton("Send Error Logs", variant="warning")
        self.send_logs_btn.setEnabled(False)
        self.send_logs_btn.setVisible(False)
        self.send_logs_btn.clicked.connect(self.send_error_logs)
        actions_layout.addWidget(self.send_logs_btn)

        actions_layout.addStretch()

        info = QLabel("Output will be saved alongside your photos")
        info.setFont(QFont("Segoe UI", 10))
        info.setStyleSheet(f"color: {COLORS['text_muted']};")
        info.setWordWrap(True)
        actions_layout.addWidget(info)

        top_section.addWidget(actions_panel)

        layout.addLayout(top_section)

        # Status card
        self.status_card = StatusCard()
        layout.addWidget(self.status_card)

        # Results section header
        results_header = QHBoxLayout()

        results_title = QLabel("People")
        results_title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        results_title.setStyleSheet(f"color: {COLORS['text_primary']};")
        results_header.addWidget(results_title)

        self.results_count = QLabel("")
        self.results_count.setFont(QFont("Segoe UI", 14))
        self.results_count.setStyleSheet(f"color: {COLORS['text_muted']};")
        results_header.addWidget(self.results_count)

        results_header.addStretch()

        layout.addLayout(results_header)

        # Results scroll area - PROPERLY EXPANDING with FlowLayout
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.results_widget = QWidget()
        self.results_widget.setStyleSheet("background: transparent;")
        self.results_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Use FlowLayout for proper wrapping and filling width
        self.results_layout = FlowLayout(self.results_widget, margin=0, spacing=20)

        # Placeholder
        self.placeholder = QLabel("Select a folder and click 'Start Processing' to organize your photos")
        self.placeholder.setFont(QFont("Segoe UI", 13))
        self.placeholder.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 80px;")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setMinimumWidth(600)
        self.results_layout.addWidget(self.placeholder)

        self.scroll_area.setWidget(self.results_widget)
        layout.addWidget(self.scroll_area, stretch=1)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Photos", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.set_folder(folder)

    def set_folder(self, folder):
        self.selected_folder = folder
        image_count = self.count_images(folder)

        self.drop_zone.setVisible(False)
        self.folder_display.setVisible(True)

        display_path = folder
        if len(display_path) > 50:
            display_path = "..." + display_path[-47:]
        self.folder_path_label.setText(display_path)
        self.image_count_label.setText(f"{image_count} images")

        self.process_btn.setEnabled(True)
        self.status_card.set_status(f"Ready to process {image_count} images", "✅", "")

        # Notify that input folder changed (for config saving)
        self.input_folder_changed.emit(folder)

    def count_images(self, folder):
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        count = 0
        try:
            for f in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, f)):
                    if Path(f).suffix.lower() in valid_extensions:
                        count += 1
        except Exception:
            pass
        return count

    def count_folder_images(self, folder_path):
        """Count images in a person folder (excluding representative face)."""
        count = 0
        try:
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    if not f.endswith('_representative_face.jpg'):
                        count += 1
        except Exception:
            pass
        return count

    def start_processing(self):
        if not self.selected_folder or not PROCESSING_AVAILABLE:
            return

        image_count = self.count_images(self.selected_folder)
        if image_count == 0:
            QMessageBox.critical(self, "Error", "No images found in the selected folder.")
            return

        result = QMessageBox.question(
            self, "Confirm Processing",
            f"Process {image_count} images?\n\nThis may take several minutes.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if result != QMessageBox.StandardButton.Yes:
            return

        # Delete previous output folder before starting fresh
        output_folder = os.path.join(self.selected_folder, 'all_images_processed')
        if os.path.exists(output_folder):
            try:
                shutil.rmtree(output_folder)
            except Exception as e:
                print(f"Warning: Could not delete previous output folder: {e}")

        self.process_btn.setEnabled(False)
        self.process_btn.setVisible(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setVisible(True)
        # Hide the send logs button when starting new processing
        self.send_logs_btn.setEnabled(False)
        self.send_logs_btn.setVisible(False)
        self.status_card.set_status("Starting...", "🔄", "Step 1/4")
        self.status_card.set_progress(0)
        self.clear_results()

        self.processing_thread = ProcessingThread(self.selected_folder)
        self.processing_thread.status_update.connect(self.on_status_update)
        self.processing_thread.progress_update.connect(self.on_progress_update)
        self.processing_thread.finished_success.connect(self.on_processing_success)
        self.processing_thread.finished_error.connect(self.on_processing_error)
        self.processing_thread.finished_stopped.connect(self.on_processing_stopped)
        self.processing_thread.start()

    def on_status_update(self, message):
        step = ""
        # 3-step optimized pipeline: Detection+Embedding -> Grouping -> Cleanup
        if "Detecting" in message or "embeddings" in message:
            step = "Step 1/3"
        elif "Grouping" in message:
            step = "Step 2/3"
        elif "Finalizing" in message:
            step = "Step 3/3"
        self.status_card.set_status(message, "🔄", step)

    def on_progress_update(self, value):
        self.status_card.set_progress(value)

    def on_processing_success(self, processed_folder, person_count):
        self.processed_folder = processed_folder
        self.process_btn.setEnabled(True)
        self.process_btn.setVisible(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.open_output_btn.setEnabled(True)

        self.status_card.set_status(f"Complete! Found {person_count} people", "✅", "")
        self.status_card.set_progress(100)

        displayed_count = self.load_person_folders(processed_folder)

        QMessageBox.information(
            self, "Success",
            f"Photos organized successfully!\n\nFound {displayed_count} people with multiple photos."
        )

    def on_processing_error(self, error_msg):
        self.process_btn.setEnabled(True)
        self.process_btn.setVisible(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)

        # Show the Send Logs button when an error occurs
        self.send_logs_btn.setEnabled(True)
        self.send_logs_btn.setVisible(True)

        # Store the error message for the log
        self.last_error_msg = error_msg

        self.status_card.set_status(f"Error: {error_msg}", "❌", "")
        self.status_card.set_progress(0)
        QMessageBox.critical(
            self, "Error",
            f"Processing failed:\n\n{error_msg}\n\n"
            "You can send the error logs to help us fix this issue."
        )

    def send_error_logs(self):
        """Open email client with error logs attached."""
        log_file = get_log_file_path()

        # Read the log contents
        log_contents = ""
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_contents = f.read()
            except Exception as e:
                log_contents = f"Could not read log file: {e}"

        # Get system info for debugging
        system_info = (
            f"System Information:\n"
            f"- Platform: {sys.platform}\n"
            f"- Python Version: {sys.version}\n"
            f"- Log File Path: {log_file}\n"
            f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        # Get the last error message if available
        last_error = getattr(self, 'last_error_msg', 'No error message captured')

        # Compose the email body
        email_body = (
            f"Sort Moments Error Report\n"
            f"{'=' * 40}\n\n"
            f"Error Message:\n{last_error}\n\n"
            f"{system_info}\n"
            f"{'=' * 40}\n"
            f"Log Contents:\n"
            f"{'=' * 40}\n\n"
            f"{log_contents if log_contents else 'No logs available'}\n"
        )

        # URL encode the email body (limit to avoid URL length issues)
        max_body_length = 1500  # Most email clients handle this length
        if len(email_body) > max_body_length:
            email_body = email_body[:max_body_length] + "\n\n... [Log truncated - full log saved at: " + log_file + "]"

        subject = urllib.parse.quote("Sort Moments - Error Report")
        body = urllib.parse.quote(email_body)

        # Open the default email client
        mailto_url = f"mailto:{ERROR_REPORT_EMAIL}?subject={subject}&body={body}"

        try:
            webbrowser.open(mailto_url)
            QMessageBox.information(
                self, "Email Client Opened",
                f"Your email client should open with the error report.\n\n"
                f"If the log is truncated, the full log file is located at:\n{log_file}\n\n"
                f"You can attach this file to the email manually if needed."
            )
        except Exception as e:
            # Fallback: show the log file location
            QMessageBox.warning(
                self, "Could Not Open Email",
                f"Could not open email client: {e}\n\n"
                f"Please manually send the log file to:\n{ERROR_REPORT_EMAIL}\n\n"
                f"Log file location:\n{log_file}"
            )

    def stop_processing(self):
        """Stop the current processing operation."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.request_stop()
            self.stop_btn.setEnabled(False)
            self.status_card.set_status("Stopping...", "⏹️", "")

    def on_processing_stopped(self):
        """Handle processing stopped by user."""
        self.process_btn.setEnabled(True)
        self.process_btn.setVisible(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setVisible(False)
        self.status_card.set_status("Processing stopped by user", "⏹️", "")
        self.status_card.set_progress(0)

    def clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()  # Hide immediately
                widget.setParent(None)  # Remove from parent
                widget.deleteLater()
        # Force immediate visual update
        self.results_widget.update()
        QApplication.processEvents()

    def load_person_folders(self, processed_folder):
        """Load person folders, filtering out those with only 1 photo."""
        self.clear_results()

        if not os.path.exists(processed_folder):
            return 0

        folders = []
        for item in os.listdir(processed_folder):
            folder_path = os.path.join(processed_folder, item)
            if os.path.isdir(folder_path):
                # Skip internal folders
                if item in ['face_detection_output']:
                    continue

                # Count images in this folder
                image_count = self.count_folder_images(folder_path)

                # Only include folders with more than 1 photo, or group photos if any exist
                if image_count > 1 or (item == "all_group_photos" and image_count > 0):
                    if item.startswith("rename_") or item == "all_group_photos":
                        folders.append((item, folder_path, image_count))
                    elif not item.endswith('.pkl') and not item.endswith('.txt'):
                        # Include renamed folders
                        folders.append((item, folder_path, image_count))

        self.results_count.setText(f"{len(folders)} people")

        if not folders:
            placeholder = QLabel("No person folders with multiple photos found")
            placeholder.setStyleSheet(f"color: {COLORS['text_muted']}; padding: 80px;")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setMinimumWidth(600)
            self.results_layout.addWidget(placeholder)
            return 0

        # Sort folders by image count (descending - most photos first)
        folders.sort(key=lambda x: x[2], reverse=True)

        # FlowLayout handles positioning automatically
        for folder_name, folder_path, image_count in folders:
            card = PersonCard(folder_name, folder_path, image_count)
            card.clicked.connect(self.folder_selected.emit)
            self.results_layout.addWidget(card)

        # Force layout update for FlowLayout
        self.results_layout.invalidate()
        self.results_widget.adjustSize()
        self.results_widget.updateGeometry()

        return len(folders)

    def open_output_folder(self):
        if self.processed_folder and os.path.exists(self.processed_folder):
            if sys.platform == "win32":
                os.startfile(self.processed_folder)
            elif sys.platform == "darwin":
                os.system(f'open "{self.processed_folder}"')
            else:
                os.system(f'xdg-open "{self.processed_folder}"')

    def refresh_folders(self):
        if self.processed_folder:
            self.load_person_folders(self.processed_folder)

    def showEvent(self, event):
        """Force layout recalculation when view becomes visible."""
        super().showEvent(event)
        # Defer the update to ensure geometry is correct
        QTimer.singleShot(0, self._update_layout)

    def _update_layout(self):
        """Update FlowLayout after view is shown."""
        if self.results_layout.count() > 0:
            # Force recalculation with actual viewport width
            viewport_width = self.scroll_area.viewport().width()
            if viewport_width > 100:  # Only if we have a reasonable width
                rect = QRect(0, 0, viewport_width, 0)
                self.results_layout.setGeometry(rect)
            self.results_layout.invalidate()
            self.results_widget.adjustSize()
            self.results_widget.updateGeometry()
            self.scroll_area.updateGeometry()


class SortMomentsApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sort Moments")
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)

        # Get the directory where the script is located
        self.app_directory = os.path.dirname(os.path.abspath(__file__))

        # Cleanup temporary folders from previous sessions on startup
        self.cleanup_temp_folders_on_startup()

        self.init_ui()
        self.check_dependencies()

    def get_config_path(self):
        """Get the path to the config file."""
        return os.path.join(self.app_directory, ".sort_moments_config.json")

    def load_last_used_folder(self):
        """Load the last used folder path from config file."""
        config_path = self.get_config_path()
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('last_used_folder')
            except Exception:
                pass
        return None

    def save_last_used_folder(self, folder_path):
        """Save the last used folder path to config file."""
        config_path = self.get_config_path()
        try:
            config = {'last_used_folder': folder_path}
            with open(config_path, 'w') as f:
                json.dump(config, f)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")

    def cleanup_temp_folders_on_startup(self):
        """Clean up temporary folders from previous sessions on application startup."""
        # Clean up all_images_processed in the application directory
        all_images_processed_path = os.path.join(self.app_directory, "all_images_processed")
        if os.path.exists(all_images_processed_path):
            try:
                shutil.rmtree(all_images_processed_path)
                print(f"Cleaned up: {all_images_processed_path}")
            except Exception as e:
                print(f"Warning: Could not delete {all_images_processed_path}: {e}")

        # Clean up face_detection_output from last used folder (if any)
        last_folder = self.load_last_used_folder()
        if last_folder and os.path.exists(last_folder):
            face_detection_output_path = os.path.join(last_folder, "face_detection_output")
            if os.path.exists(face_detection_output_path):
                try:
                    shutil.rmtree(face_detection_output_path)
                    print(f"Cleaned up: {face_detection_output_path}")
                except Exception as e:
                    print(f"Warning: Could not delete {face_detection_output_path}: {e}")

    def cleanup_temp_folders_on_close(self):
        """Clean up temporary folders when the application is closed."""
        # Clean up all_images_processed in the application directory
        all_images_processed_path = os.path.join(self.app_directory, "all_images_processed")
        if os.path.exists(all_images_processed_path):
            try:
                shutil.rmtree(all_images_processed_path)
                print(f"Cleaned up: {all_images_processed_path}")
            except Exception as e:
                print(f"Warning: Could not delete {all_images_processed_path}: {e}")

        # Clean up face_detection_output in the user's selected folder
        if hasattr(self, 'main_view') and self.main_view.selected_folder:
            face_detection_output_path = os.path.join(
                self.main_view.selected_folder, "face_detection_output"
            )
            if os.path.exists(face_detection_output_path):
                try:
                    shutil.rmtree(face_detection_output_path)
                    print(f"Cleaned up: {face_detection_output_path}")
                except Exception as e:
                    print(f"Warning: Could not delete {face_detection_output_path}: {e}")

    def closeEvent(self, event):
        """Handle application close event to clean up temporary folders."""
        self.cleanup_temp_folders_on_close()
        event.accept()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(32, 28, 32, 28)
        main_layout.setSpacing(24)

        # Header - centered
        header = QVBoxLayout()
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("Sort Moments")
        title.setFont(QFont("Segoe UI", 32, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title)

        subtitle = QLabel("Organize your photos by faces using AI")
        subtitle.setFont(QFont("Segoe UI", 14))
        subtitle.setStyleSheet(f"color: {COLORS['text_secondary']};")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(subtitle)

        main_layout.addLayout(header)

        # Stacked widget for views - EXPANDS
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Main view
        self.main_view = MainView()
        self.main_view.folder_selected.connect(self.show_folder)
        self.main_view.input_folder_changed.connect(self.save_last_used_folder)
        self.stack.addWidget(self.main_view)

        # Folder view
        self.folder_view = FolderView()
        self.folder_view.back_clicked.connect(self.show_main)
        self.folder_view.folder_renamed.connect(self.on_folder_renamed)
        self.stack.addWidget(self.folder_view)

        main_layout.addWidget(self.stack, stretch=1)

    def check_dependencies(self):
        if not PROCESSING_AVAILABLE:
            QMessageBox.warning(
                self, "Missing Dependencies",
                f"Some dependencies are missing:\n{IMPORT_ERROR}\n\n"
                "Please install required packages:\n"
                "pip install -r requirements.txt"
            )

    def show_folder(self, folder_name, folder_path):
        # Clear content before switching views to prevent flicker
        self.folder_view.clear_content()
        self.stack.setCurrentWidget(self.folder_view)
        # Load folder content after switch
        QTimer.singleShot(10, lambda: self.folder_view.load_folder(folder_path, folder_name))

    def show_main(self):
        # Clear folder view content before switching back
        self.folder_view.clear_content()
        self.stack.setCurrentWidget(self.main_view)
        # Refresh main view after switch
        QTimer.singleShot(10, self.main_view.refresh_folders)

    def on_folder_renamed(self, old_path, new_path):
        # Refresh will happen when returning to main view
        pass


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    def resource_path(relative_path: str) -> Path:
        base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        return base_path / relative_path

    icon_path = resource_path("logo.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS['bg_primary']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS['bg_secondary']))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS['bg_tertiary']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS['accent']))
    app.setPalette(palette)

    window = SortMomentsApp()
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
