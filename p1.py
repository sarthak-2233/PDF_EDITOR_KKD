import sys
import os
import mimetypes
import psd_tools

import fitz  # PyMuPDF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsPathItem, QToolBar, QAction, QFileDialog, QColorDialog, QComboBox,
    QLabel, QHBoxLayout, QWidget, QSizePolicy, QMessageBox, QListWidget,
    QListWidgetItem, QSplitter, QVBoxLayout, QTableWidget, QTableWidgetItem, QTextEdit
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QPainterPath, QBrush, QColor, QIcon, QFont
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, QEvent, pyqtSignal
from PyQt5.QtWidgets import QGestureRecognizer
from PyQt5.QtPrintSupport import QPrinter

# Optional: CSV & PSD
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from psd_tools import PSDImage
except ImportError:
    PSDImage = None

class PDFDrawingView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.drawing = False
        self.current_path = None
        self.current_item = None
        self.pen = QPen(Qt.red, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.highlighter_mode = False
        self.page_item = None
        self.undo_stack = []
        self.redo_stack = []
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        
        # Enable touch gestures
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.grabGesture(Qt.PinchGesture)
        
        # Touch gesture variables
        self.pinch_scale_factor = 1.0
        self.last_pinch_scale = 1.0
        
        # Touch event variables for manual pinch detection
        self.touch_points = {}
        self.initial_distance = 0
        self.is_pinching = False
        self.max_zoom = 5.0
        self.zoom_step = 0.01  # Further reduced from 0.1 to make zoom much slower
        self.setup_high_dpi()
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.grabGesture(Qt.PanGesture)
        self.setTabletTracking(True)

    def setup_high_dpi(self):
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() if screen else 96
        scale_factor = dpi / 96.0
        self.pen.setWidthF(3.0 * scale_factor)
        self.scale_factor = scale_factor

    def set_pdf_page(self, pixmap):
        try:
            self.scene().clear()
            self.undo_stack = []
            self.redo_stack = []
            self.page_item = self.scene().addPixmap(pixmap)
            self.page_item.setZValue(0)
            rect = QRectF(pixmap.rect())
            self.setSceneRect(rect)
            self.zoom_factor = 1.0
            self.fitInView(rect, Qt.KeepAspectRatio)
        except Exception as e:
            print(f"Error setting image/PDF page: {e}")

    def set_pen_color(self, color):
        if self.highlighter_mode:
            transparent_color = QColor(color.red(), color.green(), color.blue(), 120)
            self.pen.setColor(transparent_color)
        else:
            self.pen.setColor(color)

    def set_pen_size(self, size):
        self.pen.setWidthF(float(size) * getattr(self, 'scale_factor', 1.0))

    def set_drawing_tool(self, tool):
        self.highlighter_mode = (tool == "Highlighter")
        scale = getattr(self, 'scale_factor', 1.0)
        if self.highlighter_mode:
            color = self.pen.color()
            if color.alpha() == 255:
                color.setAlpha(120)
            self.pen.setColor(color)
            self.pen.setWidthF(15.0 * scale)
        elif tool == "Marker":
            self.pen.setWidthF(8.0 * scale)
            color = self.pen.color()
            color.setAlpha(255)
            self.pen.setColor(color)
        else:  # Pen
            self.pen.setWidthF(3.0 * scale)
            color = self.pen.color()
            color.setAlpha(255)
            self.pen.setColor(color)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.dragMode() == QGraphicsView.NoDrag:
            try:
                scene_pos = self.mapToScene(event.pos())
                self.drawing = True
                self.current_path = QPainterPath()
                self.current_path.moveTo(scene_pos)
                brush = QBrush(self.pen.color()) if self.highlighter_mode else QBrush(Qt.NoBrush)
                self.current_item = self.scene().addPath(self.current_path, self.pen, brush)
                self.current_item.setZValue(1)
            except Exception as e:
                print(f"Error in mousePressEvent: {e}")
                self.drawing = False
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_path and self.current_item:
            try:
                scene_pos = self.mapToScene(event.pos())
                self.current_path.lineTo(scene_pos)
                self.current_item.setPath(self.current_path)
            except Exception as e:
                print(f"Error in mouseMoveEvent: {e}")
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_item:
                self.undo_stack.append(self.current_item)
                self.redo_stack.clear()
                self.current_path = None
                self.current_item = None
        super().mouseReleaseEvent(event)

    # ... (Tablet and touch support unchanged, omitted here for brevity)

    def undo(self):
        if self.undo_stack:
            item = self.undo_stack.pop()
            if item.scene():
                self.scene().removeItem(item)
            self.redo_stack.append(item)

    def redo(self):
        if self.redo_stack:
            item = self.redo_stack.pop()
            self.scene().addItem(item)
            item.setZValue(1)
            self.undo_stack.append(item)

    def clear_annotations(self):
        items_to_remove = []
        for item in self.scene().items():
            if isinstance(item, QGraphicsPathItem) and item != self.page_item:
                items_to_remove.append(item)
        for item in items_to_remove:
            self.scene().removeItem(item)
        self.undo_stack = []
        self.redo_stack = []

    def save_as_image(self):
        if not self.page_item:
            QMessageBox.warning(self.parent(), "Warning", "No image/PDF page loaded!")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save as Image", "", "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if file_path:
            try:
                rect = self.sceneRect()
                pixmap = QPixmap(int(rect.width()), int(rect.height()))
                pixmap.fill(Qt.white)
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                self.scene().render(painter, QRectF(pixmap.rect()), rect)
                painter.end()
                if pixmap.save(file_path):
                    QMessageBox.information(self.parent(), "Success", f"Image saved to {file_path}")
                else:
                    QMessageBox.warning(self.parent(), "Error", "Failed to save image!")
            except Exception as e:
                QMessageBox.critical(self.parent(), "Error", f"Error saving image: {str(e)}")

    def save_as_pdf(self):
        if not self.page_item:
            QMessageBox.warning(self.parent(), "Warning", "No image/PDF page loaded!")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save as PDF", "", "PDF Files (*.pdf)"
        )
        if file_path:
            try:
                printer = QPrinter(QPrinter.HighResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)
                rect = self.sceneRect()
                printer.setPaperSize(rect.size(), QPrinter.Point)
                printer.setPageMargins(0, 0, 0, 0, QPrinter.Point)
                painter = QPainter(printer)
                painter.setRenderHint(QPainter.Antialiasing)
                self.scene().render(painter)
                painter.end()
                QMessageBox.information(self.parent(), "Success", f"PDF saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self.parent(), "Error", f"Error saving PDF: {str(e)}")

    def wheelEvent(self, event):
        try:
            if event.modifiers() & Qt.ControlModifier:
                zoom_direction = 1 if event.angleDelta().y() > 0 else -1
                self.apply_zoom(zoom_direction)
            else:
                super().wheelEvent(event)
        except Exception as e:
            print(f"Error in wheelEvent: {e}")

    def event(self, event):
        """Handle gesture and touch events for zoom"""
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)
        elif event.type() == QEvent.TouchBegin:
            return self.touchEvent(event)
        elif event.type() == QEvent.TouchUpdate:
            return self.touchEvent(event)
        elif event.type() == QEvent.TouchEnd:
            return self.touchEvent(event)
        return super().event(event)

    def touchEvent(self, event):
        """Handle touch events for manual pinch detection"""
        try:
            import math
            
            touch_points = event.touchPoints()
            print(f"Touch event: {len(touch_points)} points")
            
            if len(touch_points) == 2:
                # Two finger touch - potential pinch
                point1 = touch_points[0].pos()
                point2 = touch_points[1].pos()
                
                # Calculate distance between touch points
                distance = math.sqrt((point1.x() - point2.x())**2 + (point1.y() - point2.y())**2)
                
                if event.type() == QEvent.TouchBegin:
                    self.initial_distance = distance
                    self.is_pinching = True
                    print(f"Pinch started: initial distance = {distance}")
                    
                elif event.type() == QEvent.TouchUpdate and self.is_pinching:
                    if self.initial_distance > 0:
                        scale_factor = distance / self.initial_distance
                        print(f"Pinch update: distance = {distance}, scale = {scale_factor}")
                        
                        # Apply zoom based on scale change (less sensitive for slower zoom)
                        if scale_factor > 1.15:  # Zoom in threshold (increased from 1.05 to require more movement)
                            self.apply_zoom(1)
                            self.initial_distance = distance  # Reset for next calculation
                            print("Touch zoom in")
                        elif scale_factor < 0.85:  # Zoom out threshold (decreased from 0.95 to require more movement)
                            self.apply_zoom(-1)
                            self.initial_distance = distance  # Reset for next calculation
                            print("Touch zoom out")
                            
                elif event.type() == QEvent.TouchEnd:
                    self.is_pinching = False
                    print("Pinch ended")
                    
                event.accept()
                return True
                
            elif event.type() == QEvent.TouchEnd:
                self.is_pinching = False
                
        except Exception as e:
            print(f"Error in touchEvent: {e}")
            import traceback
            traceback.print_exc()
            
        return False

    def gestureEvent(self, event):
        """Handle pinch gesture for touch zoom"""
        try:
            from PyQt5.QtWidgets import QPinchGesture
            
            pinch = event.gesture(Qt.PinchGesture)
            if pinch and isinstance(pinch, QPinchGesture):
                print(f"Pinch gesture detected: state={pinch.state()}, scale={pinch.scaleFactor()}")
                
                if pinch.state() == Qt.GestureStarted:
                    self.last_pinch_scale = 1.0
                    print("Pinch started")
                    
                elif pinch.state() == Qt.GestureUpdated:
                    current_scale = pinch.scaleFactor()
                    total_scale = pinch.totalScaleFactor()
                    
                    print(f"Pinch updated: current={current_scale}, total={total_scale}")
                    
                    # Use total scale factor for more reliable zooming
                    if total_scale > 1.1:  # Zoom in threshold
                        self.apply_zoom(1)
                        print("Zooming in")
                    elif total_scale < 0.9:  # Zoom out threshold
                        self.apply_zoom(-1)
                        print("Zooming out")
                        
                elif pinch.state() == Qt.GestureFinished:
                    print("Pinch finished")
                
                event.accept()
                return True
        except Exception as e:
            print(f"Error in gestureEvent: {e}")
            import traceback
            traceback.print_exc()
        
        return False

    def apply_zoom(self, direction):
        new_zoom = self.zoom_factor + (direction * self.zoom_step)
        if new_zoom < self.min_zoom:
            new_zoom = self.min_zoom
        elif new_zoom > self.max_zoom:
            new_zoom = self.max_zoom
        if new_zoom != self.zoom_factor:
            scale = new_zoom / self.zoom_factor
            self.scale(scale, scale)
            self.zoom_factor = new_zoom
            if self.parent() and hasattr(self.parent(), 'update_zoom_status'):
                self.parent().update_zoom_status()

    def apply_touch_zoom(self, direction, zoom_amount):
        """Apply zoom for touch gestures with custom zoom amount"""
        # Use smaller increments for smoother touch zoom
        touch_zoom_step = min(zoom_amount, 0.1)  # Cap the zoom step for smooth experience
        new_zoom = self.zoom_factor + (direction * touch_zoom_step)
        
        if new_zoom < self.min_zoom:
            new_zoom = self.min_zoom
        elif new_zoom > self.max_zoom:
            new_zoom = self.max_zoom
            
        if new_zoom != self.zoom_factor:
            scale = new_zoom / self.zoom_factor
            self.scale(scale, scale)
            self.zoom_factor = new_zoom
            if self.parent() and hasattr(self.parent(), 'update_zoom_status'):
                self.parent().update_zoom_status()

    def reset_zoom(self):
        if self.page_item:
            self.zoom_factor = 1.0
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            if self.parent() and hasattr(self.parent(), 'update_zoom_status'):
                self.parent().update_zoom_status()

    def zoom_in(self):
        self.apply_zoom(1)

    def zoom_out(self):
        self.apply_zoom(-1)

class PDFDrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Format Drawing Application")
        self.setup_high_dpi()
        screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry() if screen else QRectF(0, 0, 1200, 800)
        self.setGeometry(100, 100, int(screen_rect.width() * 0.8), int(screen_rect.height() * 0.8))
        self.doc = None
        self.current_page = 0
        self.page_annotations = {}
        self.current_widget_type = 'graphics'
        self.setup_ui()
        self.create_toolbar()
        self.create_statusbar()
        self.show_placeholder()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter)
        self.create_sidebar()
        self.view = PDFDrawingView(self)
        self.splitter.addWidget(self.view)
        self.splitter.setSizes([250, 950])
        self.table_widget = None
        self.text_widget = None

    def setup_high_dpi(self):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        screen = QApplication.primaryScreen()
        self.dpi = screen.logicalDotsPerInch() if screen else 96
        self.scale_factor = self.dpi / 96.0
        font = QApplication.font()
        font.setPointSizeF(font.pointSizeF() * self.scale_factor)
        QApplication.setFont(font)

    def create_sidebar(self):
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        title_label = QLabel("Pages")
        title_label.setStyleSheet(f"font-weight: bold; font-size: {int(14 * self.scale_factor)}pt; padding: 5px;")
        sidebar_layout.addWidget(title_label)
        self.page_list = QListWidget()
        self.page_list.setIconSize(QSize(int(120 * self.scale_factor), int(150 * self.scale_factor)))
        self.page_list.setViewMode(QListWidget.IconMode)
        self.page_list.setResizeMode(QListWidget.Adjust)
        self.page_list.setSpacing(int(5 * self.scale_factor))
        self.page_list.itemClicked.connect(self.page_selected)
        sidebar_layout.addWidget(self.page_list)
        self.splitter.addWidget(sidebar_widget)

    def create_statusbar(self):
        self.status_bar = self.statusBar()
        self.zoom_label = QLabel("Zoom: 100%")
        self.page_label = QLabel("Page: -")
        self.status_bar.addPermanentWidget(self.zoom_label)
        self.status_bar.addPermanentWidget(self.page_label)
        self.status_bar.showMessage("Ready - Open a file to start drawing/annotating")

    def update_zoom_status(self):
        zoom_percent = int(getattr(self.view, 'zoom_factor', 1) * 100)
        self.zoom_label.setText(f"Zoom: {zoom_percent}%")

    def update_page_status(self):
        if self.doc:
            self.page_label.setText(f"Page: {self.current_page + 1}/{len(self.doc)}")
        else:
            self.page_label.setText("Page: -")

    def create_toolbar(self):
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        icon_size = int(32 * self.scale_factor)
        self.toolbar.setIconSize(QSize(icon_size, icon_size))
        self.open_action = QAction("📂 Open File", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_file)
        self.toolbar.addAction(self.open_action)
        self.toolbar.addSeparator()
        self.draw_action = QAction("✏️ Draw", self)
        self.draw_action.setCheckable(True)
        self.draw_action.setChecked(False)
        self.draw_action.toggled.connect(self.toggle_drawing)
        self.toolbar.addAction(self.draw_action)
        self.toolbar.addSeparator()
        tool_label = QLabel("Tool:")
        tool_label.setStyleSheet(f"font-size: {int(10 * self.scale_factor)}pt;")
        self.toolbar.addWidget(tool_label)
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["Pen", "Marker", "Highlighter"])
        self.tool_combo.setMinimumWidth(int(100 * self.scale_factor))
        self.tool_combo.setStyleSheet(f"font-size: {int(10 * self.scale_factor)}pt;")
        self.tool_combo.currentTextChanged.connect(self.change_tool)
        self.toolbar.addWidget(self.tool_combo)
        color_label = QLabel("Color:")
        color_label.setStyleSheet(f"font-size: {int(10 * self.scale_factor)}pt;")
        self.toolbar.addWidget(color_label)
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Red", "Blue", "Green", "Yellow", "Purple", "Black", "Custom"])
        self.color_combo.setMinimumWidth(int(80 * self.scale_factor))
        self.color_combo.setStyleSheet(f"font-size: {int(10 * self.scale_factor)}pt;")
        self.color_combo.currentTextChanged.connect(self.change_color)
        self.toolbar.addWidget(self.color_combo)
        size_label = QLabel("Size:")
        size_label.setStyleSheet(f"font-size: {int(10 * self.scale_factor)}pt;")
        self.toolbar.addWidget(size_label)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["Thin", "Medium", "Thick"])
        self.size_combo.setCurrentText("Medium")
        self.size_combo.setMinimumWidth(int(80 * self.scale_factor))
        self.size_combo.setStyleSheet(f"font-size: {int(10 * self.scale_factor)}pt;")
        self.size_combo.currentTextChanged.connect(self.change_size)
        self.toolbar.addWidget(self.size_combo)
        self.toolbar.addSeparator()
        self.undo_action = QAction("↶ Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo)
        self.toolbar.addAction(self.undo_action)
        self.redo_action = QAction("↷ Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.triggered.connect(self.redo)
        self.toolbar.addAction(self.redo_action)
        self.clear_action = QAction("🗑️ Clear", self)
        self.clear_action.triggered.connect(self.clear_annotations)
        self.toolbar.addAction(self.clear_action)
        self.toolbar.addSeparator()
        self.save_image_action = QAction("💾 Save Image", self)
        self.save_image_action.triggered.connect(self.view.save_as_image)
        self.toolbar.addAction(self.save_image_action)
        self.save_pdf_action = QAction("📄 Save PDF", self)
        self.save_pdf_action.triggered.connect(self.view.save_as_pdf)
        self.toolbar.addAction(self.save_pdf_action)
        self.toolbar.addSeparator()
        self.zoom_in_action = QAction("➕ Zoom In", self)
        self.zoom_in_action.setShortcut("Ctrl++")
        self.zoom_in_action.triggered.connect(self.view.zoom_in)
        self.toolbar.addAction(self.zoom_in_action)
        self.zoom_out_action = QAction("➖ Zoom Out", self)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.triggered.connect(self.view.zoom_out)
        self.toolbar.addAction(self.zoom_out_action)
        self.reset_zoom_action = QAction("↺ Reset Zoom", self)
        self.reset_zoom_action.setShortcut("Ctrl+0")
        self.reset_zoom_action.triggered.connect(self.view.reset_zoom)
        self.toolbar.addAction(self.reset_zoom_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

    def change_color(self, color_text):
        color_map = {
            "Red": Qt.red,
            "Blue": Qt.blue,
            "Green": Qt.green,
            "Yellow": Qt.yellow,
            "Purple": QColor(128, 0, 128),
            "Black": Qt.black
        }
        if color_text == "Custom":
            color = QColorDialog.getColor(Qt.red, self, "Select Drawing Color")
            if color.isValid():
                self.view.set_pen_color(color)
        elif color_text in color_map:
            self.view.set_pen_color(color_map[color_text])

    def change_tool(self, tool):
        self.view.set_drawing_tool(tool)

    def change_size(self, size_text):
        size_map = {
            "Thin": 2.0,
            "Medium": 5.0,
            "Thick": 8.0
        }
        if size_text in size_map:
            self.view.set_pen_size(size_map[size_text])

    def toggle_drawing(self, enabled):
        if enabled:
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.status_bar.showMessage("Drawing mode enabled - Click and drag to draw")
        else:
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.status_bar.showMessage("Pan mode enabled - Drag to move, Ctrl+Wheel to zoom")

    def undo(self):
        self.view.undo()

    def redo(self):
        self.view.redo()

    def clear_annotations(self):
        reply = QMessageBox.question(
            self, "Clear Annotations",
            "Are you sure you want to clear all annotations on this page?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.view.clear_annotations()
            if self.current_page in self.page_annotations:
                del self.page_annotations[self.current_page]

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "All Supported Files (*.pdf *.png *.jpg *.jpeg *.bmp *.gif *.csv *.psd);;PDF Files (*.pdf);;Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;CSV Files (*.csv);;PSD Files (*.psd);;All Files (*)"
        )
        if file_path:
            mime, _ = mimetypes.guess_type(file_path)
            ext = os.path.splitext(file_path)[1].lower()
            if mime == "application/pdf" or ext == ".pdf":
                self.open_pdf(file_path)
            elif mime and mime.startswith("image"):
                self.open_image(file_path)
            elif ext == ".csv":
                self.open_csv(file_path)
            elif ext == ".psd":
                self.open_psd(file_path)
            else:
                QMessageBox.warning(self, "Unsupported Format", "That file type is not supported.")

    # ---- PDF Handler and Logic remains unchanged, just refactored to accept file_path ---- #
    def open_pdf(self, file_path):
        try:
            if self.doc:
                self.doc.close()
            self.doc = fitz.open(file_path)
            if len(self.doc) == 0:
                QMessageBox.warning(self, "Warning", "The PDF file appears to be empty.")
                return
            self.page_annotations = {}
            self.current_page = 0
            self.generate_thumbnails()
            self.load_page(0)
            self.page_list.setCurrentRow(0)
            self.draw_action.setChecked(True)
            self.toggle_drawing(True)
            filename = os.path.basename(file_path)
            self.status_bar.showMessage(f"Opened: {filename} ({len(self.doc)} pages)")
            self.switch_widget('graphics')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open PDF file:\n{str(e)}")
            if self.doc:
                self.doc.close()
                self.doc = None

    def generate_thumbnails(self):
        self.page_list.clear()
        try:
            for page_num in range(len(self.doc)):
                page = self.doc.load_page(page_num)
                zoom = 0.3 * self.scale_factor
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.samples
                img = QImage(img_data, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                item = QListWidgetItem(QIcon(pixmap), f"Page {page_num + 1}")
                item.setData(Qt.UserRole, page_num)
                item.setToolTip(f"Page {page_num + 1}")
                self.page_list.addItem(item)
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Error generating thumbnails: {str(e)}")

    def load_page(self, page_index):
        if not self.doc or page_index < 0 or page_index >= len(self.doc):
            return
        try:
            self.save_current_annotations()
            page = self.doc.load_page(page_index)
            zoom = 9.0 * self.scale_factor  # or 4.0 for extra high resolution
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_data = pix.samples
            img = QImage(img_data, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.view.set_pdf_page(pixmap)
            self.restore_annotations(page_index)
            self.current_page = page_index
            self.update_page_status()
            self.update_zoom_status()
            self.switch_widget('graphics')
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading page {page_index + 1}: {str(e)}")

    def save_current_annotations(self):
        if not self.doc:
            return
        annotations = []
        for item in self.view.scene().items():
            if isinstance(item, QGraphicsPathItem) and item != self.view.page_item:
                annotations.append({
                    'path': item.path(),
                    'pen': item.pen(),
                    'brush': item.brush()
                })
        if annotations:
            self.page_annotations[self.current_page] = annotations
        elif self.current_page in self.page_annotations:
            del self.page_annotations[self.current_page]

    def restore_annotations(self, page_index):
        if page_index not in self.page_annotations:
            return
        try:
            annotations = self.page_annotations[page_index]
            for annotation in annotations:
                item = self.view.scene().addPath(
                    annotation['path'],
                    annotation['pen'],
                    annotation['brush']
                )
                item.setZValue(1)
                self.view.undo_stack.append(item)
        except Exception as e:
            print(f"Error restoring annotations: {e}")

    def page_selected(self, item):
        page_index = item.data(Qt.UserRole)
        if page_index is not None and page_index != self.current_page:
            self.load_page(page_index)

    def show_placeholder(self):
        try:
            scene = QGraphicsScene()
            self.view.setScene(scene)
            font_size = int(16 * self.scale_factor)
            text_item = scene.addText(
                "Multi-Format Drawing Application\n\n"
                "📂 Click 'Open File' to load a document or image\n"
                "✏️ Use drawing tools to annotate\n"
                "💾 Save your work as image or PDF\n\n"
                f"Display Scale: {self.scale_factor:.1f}x",
                QFont("Arial", font_size)
            )
            text_item.setDefaultTextColor(QColor(100, 100, 100))
            rect = text_item.boundingRect()
            text_item.setPos(-rect.width() / 2, -rect.height() / 2)
            scene.setSceneRect(QRectF(-300, -200, 600, 400))
            self.view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.switch_widget('graphics')
        except Exception as e:
            print(f"Error showing placeholder: {e}")

    # ---- New Handlers for Images, CSV, PSD ---- #
    def switch_widget(self, widget_type):
        # Remove the right side widget and re-add the appropriate one
        if self.current_widget_type == widget_type:
            return
        if widget_type == 'graphics':
            idx = self.splitter.indexOf(self.view)
            if idx == -1:
                self.splitter.insertWidget(1, self.view)
            if self.table_widget:
                self.table_widget.hide()
            if self.text_widget:
                self.text_widget.hide()
            self.view.show()
            self.draw_action.setEnabled(True)
            self.tool_combo.setEnabled(True)
            self.size_combo.setEnabled(True)
            self.color_combo.setEnabled(True)
        elif widget_type == 'table' and self.table_widget:
            idx = self.splitter.indexOf(self.table_widget)
            if idx == -1:
                self.splitter.insertWidget(1, self.table_widget)
            self.table_widget.show()
            self.view.hide()
            if self.text_widget:
                self.text_widget.hide()
            self.draw_action.setEnabled(False)
            self.tool_combo.setEnabled(False)
            self.size_combo.setEnabled(False)
            self.color_combo.setEnabled(False)
        elif widget_type == 'text' and self.text_widget:
            idx = self.splitter.indexOf(self.text_widget)
            if idx == -1:
                self.splitter.insertWidget(1, self.text_widget)
            self.text_widget.show()
            self.view.hide()
            if self.table_widget:
                self.table_widget.hide()
            self.draw_action.setEnabled(False)
            self.tool_combo.setEnabled(False)
            self.size_combo.setEnabled(False)
            self.color_combo.setEnabled(False)
        self.current_widget_type = widget_type

    def open_image(self, file_path):
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", "Could not open image file.")
            return
        self.view.set_pdf_page(pixmap)
        self.doc = None
        self.page_label.setText("Image")
        self.status_bar.showMessage(f"Opened: {os.path.basename(file_path)} (Image)")
        self.page_list.clear()
        self.switch_widget('graphics')

    def open_csv(self, file_path):
        if pd is None:
            QMessageBox.warning(self, "Error", "pandas is not installed. Cannot open CSV.")
            return
        df = pd.read_csv(file_path)
        if not hasattr(self, 'table_widget') or self.table_widget is None:
            self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(df.shape[1])
        self.table_widget.setRowCount(df.shape[0])
        self.table_widget.setHorizontalHeaderLabels(df.columns)
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                self.table_widget.setItem(row, col, QTableWidgetItem(str(df.iat[row, col])))
        self.page_label.setText("CSV")
        self.status_bar.showMessage(f"Opened: {os.path.basename(file_path)} (CSV Table)")
        self.page_list.clear()
        self.switch_widget('table')

    def open_psd(self, file_path):
        if PSDImage is None:
            QMessageBox.warning(self, "Error", "psd-tools is not installed. Cannot open PSD.")
            return
        psd = PSDImage.open(file_path)
        img = psd.composite()
        img = img.convert('RGBA')
        data = img.tobytes("raw", "RGBA")
        qimage = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        self.view.set_pdf_page(pixmap)
        self.doc = None
        self.page_label.setText("PSD")
        self.status_bar.showMessage(f"Opened: {os.path.basename(file_path)} (PSD Image)")
        self.page_list.clear()
        self.switch_widget('graphics')

    def closeEvent(self, event):
        if self.doc:
            reply = QMessageBox.question(
                self, "Close Application",
                "Do you want to save your annotations before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self.view.save_as_pdf()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Multi-Format Drawing Application")
    app.setApplicationVersion("1.0")
    app.setStyle('Fusion')
    try:
        window = PDFDrawingApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
