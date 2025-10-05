import sys
import os
import mimetypes
import pyaudio
import threading
import time
import numpy as np
from datetime import datetime
import tempfile
import json
import base64
from io import BytesIO
import fitz  # PyMuPDF
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsPathItem, QToolBar, QAction, QFileDialog, QColorDialog, QComboBox,
    QLabel, QHBoxLayout, QWidget, QSizePolicy, QMessageBox, QListWidget,
    QListWidgetItem, QSplitter, QVBoxLayout, QTableWidget, QTableWidgetItem, QTextEdit,
    QProgressDialog, QCheckBox, QSpinBox, QTabWidget, QScrollArea, QLineEdit
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QPainterPath, QBrush, QColor, QIcon, QFont
)
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, QEvent, pyqtSignal, QTimer, QThread
from PyQt5.QtPrintSupport import QPrinter

# Screen recording and AI processing dependencies
try:
    import cv2
    import pyaudio
    import wave
    import pyautogui
    import subprocess
    from PIL import Image, ImageGrab
    RECORDING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Screen recording dependencies not available: {e}")
    RECORDING_AVAILABLE = False

# AI Processing dependencies
try:
    import speech_recognition as sr
    import openai
    from transformers import pipeline
    AI_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AI processing dependencies not available: {e}")
    AI_PROCESSING_AVAILABLE = False

# Optional: CSV & PSD
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from psd_tools import PSDImage
except ImportError:
    PSDImage = None

class VideoProcessor(QThread):
    """Process recorded video to extract key points and annotation screenshots"""
    processing_progress = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    processing_error = pyqtSignal(str)
    
    def __init__(self, video_path, audio_path=None):
        super().__init__()
        self.video_path = video_path
        self.audio_path = audio_path
        self.openai_api_key = None
        
    def set_openai_key(self, api_key):
        self.openai_api_key = api_key
        if api_key:
            openai.api_key = api_key
            
    def run(self):
        try:
            results = {
                'key_points': [],
                'annotation_screenshots': [],
                'transcript': '',
                'processing_time': 0
            }
            
            start_time = time.time()
            
            # Step 1: Extract audio and get transcript (20% progress)
            self.processing_progress.emit(10)
            transcript = self.extract_audio_and_transcribe()
            results['transcript'] = transcript
            self.processing_progress.emit(20)
            
            # Step 2: Detect annotation events in video (60% progress)
            annotation_events = self.detect_annotation_events()
            self.processing_progress.emit(60)
            
            # Step 3: Extract screenshots at annotation points (80% progress)
            screenshots = self.extract_annotation_screenshots(annotation_events)
            results['annotation_screenshots'] = screenshots
            self.processing_progress.emit(80)
            
            # Step 4: Process transcript with AI to extract key points (100% progress)
            key_points = self.extract_key_points_with_ai(transcript, annotation_events)
            results['key_points'] = key_points
            
            results['processing_time'] = time.time() - start_time
            self.processing_progress.emit(100)
            self.processing_finished.emit(results)
            
        except Exception as e:
            self.processing_error.emit(f"Video processing error: {str(e)}")
            
    def extract_audio_and_transcribe(self):
        """Extract audio from video and transcribe it"""
        try:
            if not AI_PROCESSING_AVAILABLE:
                return "AI processing not available - install speech_recognition and transformers"
                
            # Extract audio from video using ffmpeg
            temp_audio = tempfile.mktemp(suffix='.wav')
            
            cmd = [
                'ffmpeg', '-i', self.video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                temp_audio, '-y'
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return "Could not extract audio - ffmpeg not available"
                
            # Transcribe audio
            recognizer = sr.Recognizer()
            transcript = ""
            
            with sr.AudioFile(temp_audio) as source:
                # Process audio in chunks for better accuracy
                duration = self.get_audio_duration(temp_audio)
                chunk_duration = 30  # 30 second chunks
                
                for start_time in range(0, int(duration), chunk_duration):
                    end_time = min(start_time + chunk_duration, duration)
                    
                    with sr.AudioFile(temp_audio) as chunk_source:
                        audio_data = recognizer.record(chunk_source, offset=start_time, duration=end_time-start_time)
                        
                    try:
                        # Try Google Speech Recognition first, fallback to others
                        text = recognizer.recognize_google(audio_data)
                        transcript += f"[{start_time}s-{end_time}s] {text}\n"
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError:
                        # Fallback to offline recognition
                        try:
                            text = recognizer.recognize_sphinx(audio_data)
                            transcript += f"[{start_time}s-{end_time}s] {text}\n"
                        except:
                            continue
                            
            # Clean up temporary file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
                
            return transcript
            
        except Exception as e:
            return f"Transcription error: {str(e)}"
            
    def get_audio_duration(self, audio_path):
        """Get duration of audio file"""
        try:
            cmd = ['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', 
                   '-v', 'quiet', '-of', 'csv=p=0']
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 60  # Default fallback
            
    def detect_annotation_events(self):
        """Detect when annotations/drawings are made in the video"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            annotation_events = []
            prev_frame = None
            
            for frame_idx in range(0, frame_count, max(1, int(fps/2))):  # Check every 0.5 seconds
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                if prev_frame is not None:
                    # Calculate difference between frames to detect drawing activity
                    diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                      cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
                    
                    # Focus on likely drawing areas (exclude UI elements)
                    h, w = diff.shape
                    roi = diff[int(h*0.1):int(h*0.9), int(w*0.2):int(w*0.8)]  # Central area
                    
               
               
                    # Detect significant changes that might indicate drawing
                    threshold = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY)[1]
                    change_pixels = cv2.countNonZero(threshold)
                    
                    # If significant change detected (likely drawing/annotation)
                    if change_pixels > 0:  # 0.1% of ROI changed
                        timestamp = frame_idx / fps
                        annotation_events.append({
                            'timestamp': timestamp,
                            'frame_index': frame_idx,
                            'change_intensity': change_pixels
                        })
                        
                prev_frame = frame.copy()
                
                # Update progress
                progress = 20 + int((frame_idx / frame_count) * 40)
                self.processing_progress.emit(progress)
                
            cap.release()
            
            # Filter out events that are too close together (within 2 seconds)
            filtered_events = []
            for event in annotation_events:
                if not filtered_events or (event['timestamp'] - filtered_events[-1]['timestamp']) > 2:
                    filtered_events.append(event)
                    
            return filtered_events
            
        except Exception as e:
            print(f"Error detecting annotation events: {e}")
            return []
            
    def extract_annotation_screenshots(self, annotation_events):
        """Extract screenshots at annotation timestamps"""
        try:
            screenshots = []
            cap = cv2.VideoCapture(self.video_path)
            
            for i, event in enumerate(annotation_events):
                cap.set(cv2.CAP_PROP_POS_FRAMES, event['frame_index'])
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB for proper display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to base64 for storage
                    pil_image = Image.fromarray(frame_rgb)
                    buffer = BytesIO()
                    pil_image.save(buffer, format='PNG')
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    screenshots.append({
                        'timestamp': event['timestamp'],
                        'image_base64': img_base64,
                        'description': f"Annotation activity at {event['timestamp']:.1f}s"
                    })
                    
                # Update progress
                progress = 60 + int((i / max(len(annotation_events), 1)) * 20)
                self.processing_progress.emit(progress)
                
            cap.release()
            return screenshots
            
        except Exception as e:
            print(f"Error extracting screenshots: {e}")
            return []
            
    def extract_key_points_with_ai(self, transcript, annotation_events):
        """Use AI to extract key points from transcript"""
        try:
            if not transcript or not AI_PROCESSING_AVAILABLE:
                return ["AI processing not available or no transcript"]
                
            key_points = []
            
            # Method 1: Try OpenAI GPT if API key is provided
            if self.openai_api_key:
                try:
                    # Note: Update this to use the new OpenAI API format
                    from openai import OpenAI
                    client = OpenAI(api_key=self.openai_api_key)
                    
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an AI assistant that extracts key points from video transcripts. Focus on important information, decisions, explanations, and actionable items."},
                            {"role": "user", "content": f"Extract the key points from this video transcript in bullet point format. Focus on important information and insights:\n\n{transcript}"}
                        ],
                        max_tokens=3500,
                        temperature=0.3
                    )
                    
                    gpt_response = response.choices[0].message.content
                    # Parse bullet points
                    points = [point.strip() for point in gpt_response.split('\n') if point.strip() and ('•' in point or '-' in point or point.strip().startswith(('1.', '2.', '3.')))]
                    key_points.extend(points[:10])  # Limit to 10 points
                    
                except Exception as e:
                    print(f"OpenAI processing failed: {e}")
                    
            # Method 2: Fallback to local text processing
            if not key_points:
                # Simple keyword-based extraction as fallback
                sentences = transcript.replace('\n', ' ').split('.')
                important_keywords = ['important', 'key', 'note', 'remember', 'significant', 
                                    'crucial', 'main', 'primary', 'essential', 'critical']
                
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in important_keywords):
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 10:
                            key_points.append(f"• {clean_sentence}")
                            
                # If no keyword-based points found, extract first few sentences
                if not key_points:
                    for sentence in sentences[:5]:
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) > 20:
                            key_points.append(f"• {clean_sentence}")
                            
            # Add annotation timing information
            if annotation_events:
                key_points.append(f"• {len(annotation_events)} annotation/drawing events detected during the recording")
                for event in annotation_events[:3]:  # Show first 3 events
                    key_points.append(f"  - Annotation activity at {event['timestamp']:.1f} seconds")
                    
            return key_points[:15]  # Limit total points
            
        except Exception as e:
            return [f"Error extracting key points: {str(e)}"]

class EnhancedScreenRecorder(QThread):
    """Enhanced screen recorder with video processing capabilities"""
    recording_finished = pyqtSignal(str)
    recording_error = pyqtSignal(str)
    processing_requested = pyqtSignal(str, str)  # video_path, audio_path
    
    def __init__(self):
        super().__init__()
        self.recording = False
        self.output_path = ""
        self.fps = 30
        self.audio_format = pyaudio.paInt16
        self.channels = 2
        self.rate = 44100
        self.chunk = 1024
        self.audio_frames = []
        self.video_frames = []
        self.process_after_recording = False
        
    def set_processing_enabled(self, enabled):
        self.process_after_recording = enabled
        
    def start_recording(self, output_path):
        self.output_path = output_path
        self.recording = True
        self.audio_frames = []
        self.video_frames = []
        self.start()
        
    def stop_recording(self):
        self.recording = False
        
    def run(self):
        try:
            # Initialize audio recording
            audio_thread = threading.Thread(target=self._record_audio)
            audio_thread.daemon = True
            audio_thread.start()
            
            # Record video
            self._record_video()
            
            # Wait for audio thread to finish
            audio_thread.join(timeout=2.0)
            
            # Combine audio and video
            audio_path = self._combine_audio_video()
            
            # Emit signal for processing if enabled
            if self.process_after_recording and AI_PROCESSING_AVAILABLE:
                self.processing_requested.emit(self.output_path, audio_path)
            
        except Exception as e:
            self.recording_error.emit(f"Recording error: {str(e)}")
            
    def _record_audio(self):
        """Record system audio and microphone"""
        try:
            p = pyaudio.PyAudio()
            default_input = p.get_default_input_device_info()
            
            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=default_input['index'],
                frames_per_buffer=self.chunk
            )
            
            while self.recording:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    self.audio_frames.append(data)
                except Exception as e:
                    print(f"Audio recording error: {e}")
                    break
                    
            stream.stop_stream()
            stream.close()
            p.terminate()
            
        except Exception as e:
            print(f"Audio initialization error: {e}")
            
    def _record_video(self):
        """Record screen video"""
        try:
            screen = pyautogui.screenshot()
            width, height = screen.size
            
            temp_video_path = self.output_path.replace('.mp4', '_temp_video.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_video_path, fourcc, self.fps, (width, height))
            
            frame_duration = 1.0 / self.fps
            last_time = time.time()
            
            while self.recording:
                current_time = time.time()
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
                self.video_frames.append(frame)
                
                elapsed = time.time() - last_time
                sleep_time = max(0, frame_duration - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_time = time.time()
                
            out.release()
            
        except Exception as e:
            print(f"Video recording error: {e}")
            raise
            
    def _combine_audio_video(self):
        """Combine audio and video using ffmpeg"""
        try:
            temp_video_path = self.output_path.replace('.mp4', '_temp_video.avi')
            temp_audio_path = self.output_path.replace('.mp4', '_temp_audio.wav')
            
            # Save audio to temporary file
            if self.audio_frames:
                p = pyaudio.PyAudio()
                wf = wave.open(temp_audio_path, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.audio_frames))
                wf.close()
                p.terminate()
                
                try:
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video_path,
                        '-i', temp_audio_path,
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-strict', 'experimental',
                        self.output_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                        
                    self.recording_finished.emit(self.output_path)
                    return temp_audio_path  # Return audio path for processing
                    
                except (subprocess.CalledProcessError, FileNotFoundError):
                    print("ffmpeg not available, saving video only")
                    if os.path.exists(temp_video_path):
                        os.rename(temp_video_path, self.output_path)
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
            else:
                if os.path.exists(temp_video_path):
                    os.rename(temp_video_path, self.output_path)
                    
            self.recording_finished.emit(self.output_path)
            return None
            
        except Exception as e:
            self.recording_error.emit(f"Error combining audio/video: {str(e)}")
            return None

class ResultsViewer(QWidget):
    """Widget to display video processing results"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Key Points Tab
        self.key_points_widget = QTextEdit()
        self.key_points_widget.setReadOnly(True)
        self.tab_widget.addTab(self.key_points_widget, "Key Points")
        
        # Transcript Tab
        self.transcript_widget = QTextEdit()
        self.transcript_widget.setReadOnly(True)
        self.tab_widget.addTab(self.transcript_widget, "Transcript")
        
        # Screenshots Tab
        self.screenshots_widget = QScrollArea()
        self.screenshots_widget.setWidgetResizable(True)
        self.screenshots_content = QWidget()
        self.screenshots_layout = QVBoxLayout(self.screenshots_content)
        self.screenshots_widget.setWidget(self.screenshots_content)
        self.tab_widget.addTab(self.screenshots_widget, "Annotation Screenshots")
        
    def display_results(self, results):
        """Display the processing results"""
        # Display key points
        if results['key_points']:
            key_points_text = "🔑 Key Points Extracted from Video:\n\n"
            for point in results['key_points']:
                key_points_text += f"{point}\n"
            key_points_text += f"\n⏱️ Processing completed in {results['processing_time']:.1f} seconds"
        else:
            key_points_text = "No key points extracted from the video."
        self.key_points_widget.setPlainText(key_points_text)
        
        # Display transcript
        if results['transcript']:
            self.transcript_widget.setPlainText(f"🎙️ Audio Transcript:\n\n{results['transcript']}")
        else:
            self.transcript_widget.setPlainText("No transcript available.")
            
        # Display annotation screenshots
        self.clear_screenshots()
        if results['annotation_screenshots']:
            for i, screenshot in enumerate(results['annotation_screenshots']):
                self.add_screenshot(screenshot, i)
        else:
            no_screenshots_label = QLabel("No annotation screenshots captured.")
            no_screenshots_label.setAlignment(Qt.AlignCenter)
            self.screenshots_layout.addWidget(no_screenshots_label)
            
    def clear_screenshots(self):
        """Clear existing screenshots"""
        while self.screenshots_layout.count():
            child = self.screenshots_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def add_screenshot(self, screenshot, index):
        """Add a screenshot to the display"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(screenshot['image_base64'])
            qimage = QImage()
            qimage.loadFromData(image_data)
            
            if not qimage.isNull():
                pixmap = QPixmap.fromImage(qimage)
                
                # Scale image if too large
                if pixmap.width() > 800 or pixmap.height() > 600:
                    pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Create container widget
                container = QWidget()
                container_layout = QVBoxLayout(container)
                
                # Add timestamp label
                timestamp_label = QLabel(f"📸 Screenshot {index + 1} - {screenshot['timestamp']:.1f}s")
                timestamp_label.setFont(QFont("Arial", 12, QFont.Bold))
                container_layout.addWidget(timestamp_label)
                
                # Add image label
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(image_label)
                
                # Add description
                desc_label = QLabel(screenshot.get('description', ''))
                desc_label.setWordWrap(True)
                container_layout.addWidget(desc_label)
                
                # Add separator
                separator = QLabel("─" * 50)
                separator.setAlignment(Qt.AlignCenter)
                container_layout.addWidget(separator)
                
                self.screenshots_layout.addWidget(container)
                
        except Exception as e:
            error_label = QLabel(f"Error displaying screenshot {index + 1}: {str(e)}")
            self.screenshots_layout.addWidget(error_label)

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
        self.max_zoom = 5.0
        self.zoom_step = 0.01
        
        self.setup_high_dpi()

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


class EnhancedPDFDrawingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Enhanced Multi-Format Drawing Application")
        self.setup_high_dpi()
        screen = QApplication.primaryScreen()
        screen_rect = screen.availableGeometry() if screen else QRectF(0, 0, 1200, 800)
        self.setGeometry(100, 100, int(screen_rect.width() * 0.85), int(screen_rect.height() * 0.85))
        self.doc = None
        self.current_page = 0
        self.page_annotations = {}
        self.current_widget_type = 'graphics'
        
        # Enhanced recording setup
        self.screen_recorder = None
        self.video_processor = None
        self.is_recording = False
        self.recording_start_time = None
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_status)
        
        # AI processing settings
        self.openai_api_key = ""
        self.auto_process_recordings = True
        
        self.setup_ui()
        self.create_toolbar()
        self.create_statusbar()
        self.show_placeholder()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.main_splitter)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create content area with tabs
        self.content_tabs = QTabWidget()
        self.main_splitter.addWidget(self.content_tabs)
        
        # Drawing/PDF view tab
        self.view = PDFDrawingView(self)
        self.content_tabs.addTab(self.view, "📄 Document/Drawing")
        
        # Results viewer tab
        self.results_viewer = ResultsViewer()
        self.content_tabs.addTab(self.results_viewer, "🤖 AI Analysis")
        
        # Set initial sizes
        self.main_splitter.setSizes([250, 1000])
        
        # Additional widgets for different file types
        self.table_widget = None
        self.text_widget = None

    def setup_high_dpi(self):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        screen = QApplication.primaryScreen()
        self.dpi = screen.logicalDotsPerInch() if screen else 96
        self.scale_factor = self.dpi / 96.0

    def create_sidebar(self):
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        
        # Pages section
        title_label = QLabel("📄 Pages")
        title_label.setStyleSheet(f"font-weight: bold; font-size: {int(14 * self.scale_factor)}pt; padding: 5px;")
        sidebar_layout.addWidget(title_label)
        
        self.page_list = QListWidget()
        self.page_list.setIconSize(QSize(int(120 * self.scale_factor), int(150 * self.scale_factor)))
        self.page_list.setViewMode(QListWidget.IconMode)
        self.page_list.setResizeMode(QListWidget.Adjust)
        self.page_list.setSpacing(int(5 * self.scale_factor))
        self.page_list.itemClicked.connect(self.page_selected)
        sidebar_layout.addWidget(self.page_list)
        
        # AI Settings section
        ai_label = QLabel("🤖 AI Settings")
        ai_label.setStyleSheet(f"font-weight: bold; font-size: {int(12 * self.scale_factor)}pt; padding: 5px;")
        sidebar_layout.addWidget(ai_label)
        
        # Auto-process checkbox
        self.auto_process_check = QCheckBox("Auto-process recordings")
        self.auto_process_check.setChecked(self.auto_process_recordings)
        self.auto_process_check.toggled.connect(self.toggle_auto_processing)
        sidebar_layout.addWidget(self.auto_process_check)
        
        if not AI_PROCESSING_AVAILABLE:
            self.auto_process_check.setEnabled(False)
            self.auto_process_check.setToolTip("AI processing not available - install required dependencies")
        
        # OpenAI API Key input
        api_key_label = QLabel("OpenAI API Key:")
        sidebar_layout.addWidget(api_key_label)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter OpenAI API key (optional)")
        self.api_key_input.textChanged.connect(self.update_api_key)
        sidebar_layout.addWidget(self.api_key_input)
        
        # Add stretch to push everything to top
        sidebar_layout.addStretch()
        
        self.main_splitter.addWidget(sidebar_widget)

    def toggle_auto_processing(self, checked):
        self.auto_process_recordings = checked
        if self.screen_recorder:
            self.screen_recorder.set_processing_enabled(checked)

    def update_api_key(self, text):
        self.openai_api_key = text
        if self.video_processor:
            self.video_processor.set_openai_key(text)

    def create_statusbar(self):
        self.status_bar = self.statusBar()
        self.zoom_label = QLabel("Zoom: 100%")
        self.page_label = QLabel("Page: -")
        self.recording_label = QLabel("")
        self.ai_status_label = QLabel("")
        
        self.status_bar.addPermanentWidget(self.ai_status_label)
        self.status_bar.addPermanentWidget(self.recording_label)
        self.status_bar.addPermanentWidget(self.zoom_label)
        self.status_bar.addPermanentWidget(self.page_label)
        
        ai_status = "🤖 AI Ready" if AI_PROCESSING_AVAILABLE else "🤖 AI Unavailable"
        self.ai_status_label.setText(ai_status)
        
        self.status_bar.showMessage("Ready - Open a file to start drawing/annotating")

    def update_zoom_status(self):
        zoom_percent = int(getattr(self.view, 'zoom_factor', 1) * 100)
        self.zoom_label.setText(f"Zoom: {zoom_percent}%")

    def update_page_status(self):
        if self.doc:
            self.page_label.setText(f"Page: {self.current_page + 1}/{len(self.doc)}")
        else:
            self.page_label.setText("Page: -")

    def update_recording_status(self):
        if self.is_recording and self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.recording_label.setText(f"🔴 REC {minutes:02d}:{seconds:02d}")
        else:
            self.recording_label.setText("")

    def create_toolbar(self):
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        icon_size = int(32 * self.scale_factor)
        self.toolbar.setIconSize(QSize(icon_size, icon_size))
        
        # File operations
        self.open_action = QAction("📂 Open File", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_file)
        self.toolbar.addAction(self.open_action)
        self.toolbar.addSeparator()
        
        # Drawing toggle
        self.draw_action = QAction("✏️ Draw", self)
        self.draw_action.setCheckable(True)
        self.draw_action.setChecked(False)
        self.draw_action.toggled.connect(self.toggle_drawing)
        self.toolbar.addAction(self.draw_action)
        self.toolbar.addSeparator()
        
        # Drawing tools
        tool_label = QLabel("Tool:")
        self.toolbar.addWidget(tool_label)
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["Pen", "Marker", "Highlighter"])
        self.tool_combo.currentTextChanged.connect(self.change_tool)
        self.toolbar.addWidget(self.tool_combo)
        
        color_label = QLabel("Color:")
        self.toolbar.addWidget(color_label)
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Red", "Blue", "Green", "Yellow", "Purple", "Black", "Custom"])
        self.color_combo.currentTextChanged.connect(self.change_color)
        self.toolbar.addWidget(self.color_combo)
        
        size_label = QLabel("Size:")
        self.toolbar.addWidget(size_label)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["Thin", "Medium", "Thick"])
        self.size_combo.setCurrentText("Medium")
        self.size_combo.currentTextChanged.connect(self.change_size)
        self.toolbar.addWidget(self.size_combo)
        self.toolbar.addSeparator()
        
        # Edit operations
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
        
        # Enhanced Screen Recording
        self.record_action = QAction("🎥 Smart Record", self)
        self.record_action.setShortcut("Ctrl+R")
        self.record_action.triggered.connect(self.toggle_screen_recording)
        if not RECORDING_AVAILABLE:
            self.record_action.setEnabled(False)
            self.record_action.setToolTip("Screen recording not available - missing dependencies")
        else:
            self.record_action.setToolTip("Record screen with AI analysis")
        self.toolbar.addAction(self.record_action)
        
        # Process existing video
        self.process_video_action = QAction("🤖 Process Video", self)
        self.process_video_action.triggered.connect(self.process_existing_video)
        if not AI_PROCESSING_AVAILABLE:
            self.process_video_action.setEnabled(False)
            self.process_video_action.setToolTip("AI processing not available")
        self.toolbar.addAction(self.process_video_action)
        self.toolbar.addSeparator()
        
        # Save operations
        self.save_image_action = QAction("💾 Save Image", self)
        self.save_image_action.triggered.connect(self.view.save_as_image)
        self.toolbar.addAction(self.save_image_action)
        
        self.save_pdf_action = QAction("📄 Save PDF", self)
        self.save_pdf_action.triggered.connect(self.view.save_as_pdf)
        self.toolbar.addAction(self.save_pdf_action)
        self.toolbar.addSeparator()
        
        # Zoom operations
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

    def toggle_screen_recording(self):
        if not RECORDING_AVAILABLE:
            QMessageBox.warning(self, "Recording Unavailable", 
                              "Screen recording is not available. Please install required dependencies:\n"
                              "pip install opencv-python pyaudio pyautogui pillow")
            return
            
        if not self.is_recording:
            self.start_enhanced_recording()
        else:
            self.stop_enhanced_recording()

    def start_enhanced_recording(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"smart_recording_{timestamp}.mp4"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Smart Screen Recording", default_filename, 
                "MP4 Files (*.mp4);;All Files (*)"
            )
            
            if not file_path:
                return
                
            if not file_path.lower().endswith('.mp4'):
                file_path += '.mp4'
            
            # Initialize enhanced recorder
            self.screen_recorder = EnhancedScreenRecorder()
            self.screen_recorder.set_processing_enabled(self.auto_process_recordings and AI_PROCESSING_AVAILABLE)
            self.screen_recorder.recording_finished.connect(self.on_recording_finished)
            self.screen_recorder.recording_error.connect(self.on_recording_error)
            self.screen_recorder.processing_requested.connect(self.start_video_processing)
            
            self.screen_recorder.start_recording(file_path)
            
            # Update UI
            self.is_recording = True
            self.recording_start_time = time.time()
            self.record_action.setText("⏹️ Stop Recording")
            self.record_action.setToolTip("Click to stop smart recording")
            
            self.recording_timer.start(1000)
            
            self.status_bar.showMessage(f"Smart recording started - {os.path.basename(file_path)}")
            
            # Show enhanced notification
            ai_info = " with AI analysis" if (self.auto_process_recordings and AI_PROCESSING_AVAILABLE) else ""
            QMessageBox.information(self, "Smart Recording Started", 
                                  f"Enhanced screen recording has started{ai_info}!\n\n"
                                  f"Output: {os.path.basename(file_path)}\n\n"
                                  f"Features:\n"
                                  f"• Screen capture with audio\n"
                                  f"• Automatic annotation detection\n"
                                  f"• AI-powered key point extraction\n"
                                  f"• Screenshot capture at annotation moments\n\n"
                                  f"Click 'Stop Recording' when done.")
            
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to start enhanced recording:\n{str(e)}")
            self.reset_recording_ui()

    def stop_enhanced_recording(self):
        try:
            if self.screen_recorder:
                self.screen_recorder.stop_recording()
                self.status_bar.showMessage("Stopping recording and preparing for AI analysis...")
                
                self.record_action.setText("⏳ Processing...")
                self.record_action.setEnabled(False)
                
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", f"Failed to stop recording:\n{str(e)}")
            self.reset_recording_ui()

    def start_video_processing(self, video_path, audio_path):
        """Start AI processing of the recorded video"""
        if not AI_PROCESSING_AVAILABLE:
            return
            
        try:
            self.video_processor = VideoProcessor(video_path, audio_path)
            self.video_processor.set_openai_key(self.openai_api_key)
            self.video_processor.processing_progress.connect(self.update_processing_progress)
            self.video_processor.processing_finished.connect(self.on_processing_finished)
            self.video_processor.processing_error.connect(self.on_processing_error)
            
            # Show progress dialog
            self.progress_dialog = QProgressDialog("Processing video with AI...", "Cancel", 0, 100, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.canceled.connect(self.cancel_processing)
            self.progress_dialog.show()
            
            self.video_processor.start()
            self.ai_status_label.setText("🤖 AI Processing...")
            
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Failed to start video processing:\n{str(e)}")

    def update_processing_progress(self, progress):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(progress)

    def cancel_processing(self):
        if self.video_processor and self.video_processor.isRunning():
            self.video_processor.terminate()
            self.video_processor.wait()
        self.ai_status_label.setText("🤖 AI Ready" if AI_PROCESSING_AVAILABLE else "🤖 AI Unavailable")

    def on_processing_finished(self, results):
        """Handle completion of video processing"""
        try:
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.close()
                
            self.ai_status_label.setText("🤖 AI Complete")
            
            # Display results in the results viewer
            self.results_viewer.display_results(results)
            
            # Switch to results tab
            self.content_tabs.setCurrentWidget(self.results_viewer)
            
            # Show completion notification
            QMessageBox.information(
                self, "AI Processing Complete", 
                f"Video analysis completed successfully!\n\n"
                f"📝 {len(results['key_points'])} key points extracted\n"
                f"📸 {len(results['annotation_screenshots'])} annotation screenshots captured\n"
                f"⏱️ Processing time: {results['processing_time']:.1f} seconds\n\n"
                f"Check the 'AI Analysis' tab to view results."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Display Error", f"Error displaying results: {str(e)}")
        finally:
            self.video_processor = None

    def on_processing_error(self, error_message):
        """Handle video processing errors"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            
        self.ai_status_label.setText("🤖 AI Error")
        QMessageBox.critical(self, "Processing Error", f"Video processing failed:\n{error_message}")
        self.video_processor = None

    def process_existing_video(self):
        """Process an existing video file"""
        if not AI_PROCESSING_AVAILABLE:
            QMessageBox.warning(self, "AI Unavailable", 
                              "AI processing is not available. Please install required dependencies:\n"
                              "pip install speechrecognition openai transformers torch")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video to Process", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.start_video_processing(file_path, None)

    def on_recording_finished(self, file_path):
        """Handle recording completion"""
        self.reset_recording_ui()
        
        if not (self.auto_process_recordings and AI_PROCESSING_AVAILABLE):
            # Show standard completion message if not auto-processing
            reply = QMessageBox.information(
                self, "Recording Complete", 
                f"Screen recording saved successfully!\n\n"
                f"File: {os.path.basename(file_path)}\n"
                f"Location: {os.path.dirname(file_path)}\n\n"
                f"Would you like to process this recording with AI?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes and AI_PROCESSING_AVAILABLE:
                self.start_video_processing(file_path, None)

    def on_recording_error(self, error_message):
        """Handle recording errors"""
        self.reset_recording_ui()
        QMessageBox.critical(self, "Recording Error", error_message)

    def reset_recording_ui(self):
        """Reset recording UI to initial state"""
        self.is_recording = False
        self.recording_start_time = None
        self.recording_timer.stop()
        self.record_action.setText("🎥 Smart Record")
        self.record_action.setToolTip("Record screen with AI analysis")
        self.record_action.setEnabled(True)
        self.recording_label.setText("")
        if self.screen_recorder:
            self.screen_recorder = None

    # Drawing and file handling methods
    def change_color(self, color_text):
        color_map = {
            "Red": Qt.red, "Blue": Qt.blue, "Green": Qt.green,
            "Yellow": Qt.yellow, "Purple": QColor(128, 0, 128), "Black": Qt.black
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
        size_map = {"Thin": 2.0, "Medium": 5.0, "Thick": 8.0}
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
            self, "Open File", "",
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
            self.content_tabs.setCurrentWidget(self.view)
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
            zoom = 9.0 * self.scale_factor
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
            self.content_tabs.setCurrentWidget(self.view)
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
        self.content_tabs.setCurrentWidget(self.view)

    def open_csv(self, file_path):
        if pd is None:
            QMessageBox.warning(self, "Error", "pandas is not installed. Cannot open CSV.")
            return
        try:
            df = pd.read_csv(file_path)
            if not hasattr(self, 'table_widget') or self.table_widget is None:
                self.table_widget = QTableWidget()
                self.content_tabs.addTab(self.table_widget, "📊 Table Data")
            self.table_widget.setColumnCount(df.shape[1])
            self.table_widget.setRowCount(df.shape[0])
            self.table_widget.setHorizontalHeaderLabels(df.columns)
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    self.table_widget.setItem(row, col, QTableWidgetItem(str(df.iat[row, col])))
            self.page_label.setText("CSV")
            self.status_bar.showMessage(f"Opened: {os.path.basename(file_path)} (CSV Table)")
            self.page_list.clear()
            self.content_tabs.setCurrentWidget(self.table_widget)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open CSV file:\n{str(e)}")

    def open_psd(self, file_path):
        if PSDImage is None:
            QMessageBox.warning(self, "Error", "psd-tools is not installed. Cannot open PSD.")
            return
        try:
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
            self.content_tabs.setCurrentWidget(self.view)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open PSD file:\n{str(e)}")

    def show_placeholder(self):
        try:
            scene = QGraphicsScene()
            self.view.setScene(scene)
            font_size = int(16 * self.scale_factor)
            
            recording_status = "🎥 Smart Recording Available" if RECORDING_AVAILABLE else "🎥 Recording Unavailable"
            ai_status = "🤖 AI Processing Available" if AI_PROCESSING_AVAILABLE else "🤖 AI Processing Unavailable"
            
            placeholder_text = (
                "AI-Enhanced Multi-Format Drawing Application\n\n"
                "✨ NEW FEATURES:\n"
                "🎥 Smart screen recording with AI analysis\n"
                "🤖 Automatic key point extraction\n"
                "📸 Annotation screenshot capture\n"
                "🎙️ Audio transcription\n\n"
                "📂 Click 'Open File' to load a document or image\n"
                "✏️ Use drawing tools to annotate\n"
                f"{recording_status}\n"
                f"{ai_status}\n"
                "💾 Save your work as image or PDF\n\n"
                f"Display Scale: {self.scale_factor:.1f}x\n\n"
                "🔧 SETUP REQUIREMENTS:\n"
                "For full functionality, install:\n"
                "pip install opencv-python pyaudio pyautogui pillow\n"
                "pip install speechrecognition openai transformers torch"
            )
            
            text_item = scene.addText(placeholder_text, QFont("Arial", font_size))
            text_item.setDefaultTextColor(QColor(80, 80, 80))
            rect = text_item.boundingRect()
            text_item.setPos(-rect.width() / 2, -rect.height() / 2)
            scene.setSceneRect(QRectF(-400, -300, 800, 600))
            self.view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
            self.content_tabs.setCurrentWidget(self.view)
        except Exception as e:
            print(f"Error showing placeholder: {e}")

    def closeEvent(self, event):
        # Stop recording if active
        if self.is_recording:
            reply = QMessageBox.question(
                self, "Recording Active",
                "Screen recording is currently active. Do you want to stop recording and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.stop_enhanced_recording()
                QApplication.processEvents()
                time.sleep(1)
            else:
                event.ignore()
                return
        
        # Stop video processing if active
        if self.video_processor and self.video_processor.isRunning():
            reply = QMessageBox.question(
                self, "Processing Active",
                "AI video processing is currently active. Do you want to cancel processing and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.video_processor.terminate()
                self.video_processor.wait()
            else:
                event.ignore()
                return
        
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
    app.setApplicationName("AI-Enhanced Multi-Format Drawing Application")
    app.setApplicationVersion("2.0")
    app.setStyle('Fusion')
    
    # Check dependencies on startup
    print("\n" + "="*60)
    print("AI-Enhanced Drawing Application - Dependency Check")
    print("="*60)
    
    if not RECORDING_AVAILABLE:
        print("⚠️  Screen recording dependencies missing:")
        print("   pip install opencv-python pyaudio pyautogui pillow")
    else:
        print("✅ Screen recording available")
    
    if not AI_PROCESSING_AVAILABLE:
        print("⚠️  AI processing dependencies missing:")
        print("   pip install speechrecognition openai transformers torch")
        print("   Note: OpenAI API key required for advanced features")
    else:
        print("✅ AI processing available")
    
    if RECORDING_AVAILABLE and AI_PROCESSING_AVAILABLE:
        print("🚀 All features available - enjoy the enhanced experience!")
    else:
        
        print("ℹ️  Some features will be limited - install dependencies for full functionality")
    
    print("="*60 + "\n")
    
    try:
        window = EnhancedPDFDrawingApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()