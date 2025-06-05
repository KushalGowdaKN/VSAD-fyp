import sys
import os
import cv2
import numpy as np
import time
import threading
import smtplib
import pygame
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                            QComboBox, QTextEdit, QMessageBox,
                            QProgressBar, QGroupBox,
                            QSplitter, QFrame, QTabWidget
                            )
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from keras.models import load_model

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    prediction_signal = pyqtSignal(str, float)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()
    
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.running = True
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        
        while self.running and current_frame < total_frames:
            ret, frame = cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)
                #pass the frame to the model in the main thread
                current_frame += 1
                progress = int((current_frame / total_frames) * 100)
                self.progress_signal.emit(progress)
                time.sleep(0.03)  # To control frame rate
            else:
                break
                
        cap.release()
        self.finished_signal.emit()
        
    def stop(self):
        self.running = False
        self.wait()

class AnomalyDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.setWindowIcon(QIcon("icon.png"))
        self.model = None
        self.webcam_thread = None
        self.video_thread = None
        self.current_frame = None
        self.last_alert_time = {}  # Store last alert time for each anomaly type
        self.alert_cooldown = 300  # 5 minutes (300 seconds) cooldown between alerts
        self.confidence_threshold = 0.70  # Minimum confidence to trigger alert
        self.detection_threshold = 0.50  # Minimum confidence to display detection
        self.alert_sound_path = "alert.wav"  # Path to your alert sound
        self.anomaly_history = {}  # Store detected anomalies
        
        
        # Initialize pygame for sound playback with error handling
        try:
            pygame.mixer.init()
            self.pygame_initialized = True
        except Exception as e:
            self.pygame_initialized = False
            print(f"Warning: Could not initialize pygame audio: {e}")
            print("Alert sounds will be disabled.")
        
        # Replace with hardcoded values:
        self.email_address = "d6517299@gmail.com"  # Hardcode your email
        self.email_password = "ydnb bybf irvn wqna"  # Hardcode your email password
        self.recipient_email = "kushalgowda6015@gmail.com"  # Hardcode recipient email
        self.dark_mode = False  # Set dark mode default
        self.confidence_threshold = 0.98 # Hardcode alert threshold
        self.detection_threshold = 0.98 # Hardcode detection threshold
        self.alert_cooldown = 300  # Hardcode cooldown
        
        # Setup the UI
        self.init_ui()
        
      
        
        # Try to load the model
        self.load_model()

    def init_ui(self):
        # Set window properties
        self.setWindowTitle("Video Surveillance Anomaly Detection System")
        self.setGeometry(100, 100, 1280, 900)
        
        
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f8f9fa;
                color: #343a40;
            }
            QGroupBox {
                border: 1px solid #4dabf7;
                border-radius: 6px;
                margin-top: 1ex;
                font-weight: bold;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 8px;
                color: #1971c2;
            }
            QPushButton {
                background-color: #339af0;
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #1c7ed6;
            }
            QPushButton:disabled {
                background-color: #ced4da;
            }
            QComboBox, QLineEdit {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 6px;
                background-color: white;
            }
            QSlider::handle {
                background-color: #339af0;
                border-radius: 7px;
                border: none;
            }
            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 4px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #51cf66;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 6px;
                background-color: white;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #e9ecef;
                border: 1px solid #dee2e6;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #339af0;
                border: none;
                width: 18px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #74c0fc;
                border-radius: 4px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with proper spacing
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Create a splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Left panel - Video display and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)
        
        # Video feed group
        video_group = QGroupBox("Video Feed")
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(15, 20, 15, 15)
        
        # Video display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("""
            border: 2px solid #4dabf7;
            border-radius: 8px;
            background-color: #212529;
            padding: 2px;
        """)
        video_layout.addWidget(self.image_label)
        
        left_layout.addWidget(video_group)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(15, 20, 15, 15)
        
        # Source selection
        source_layout = QVBoxLayout()
        source_label = QLabel("Source:")
        source_label.setStyleSheet("font-weight: bold;")
        self.source_combo = QComboBox()
        self.source_combo.setStyleSheet("font-size: 12px;")
        self.source_combo.addItems(["Video"])
        self.source_combo.setMinimumWidth(150)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_combo)
        
        # Buttons
        buttons_layout = QVBoxLayout()
        buttons_widget = QWidget()
        buttons_layout_inner = QHBoxLayout(buttons_widget)
        buttons_layout_inner.setContentsMargins(0, 0, 0, 0)
        
        # Create icons for buttons
        play_icon = QIcon.fromTheme("media-playback-start") or QIcon("icons/play.png")
        stop_icon = QIcon.fromTheme("media-playback-stop") or QIcon("icons/stop.png")
        
        self.start_button = QPushButton("Start Detection")
        self.start_button.setIcon(play_icon)
        self.start_button.setMinimumSize(QSize(160, 40))
        self.start_button.clicked.connect(self.start_detection)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setIcon(stop_icon)
        self.stop_button.setMinimumSize(QSize(160, 40))
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        
        buttons_layout_inner.addWidget(self.start_button)
        buttons_layout_inner.addWidget(self.stop_button)
        buttons_layout.addWidget(buttons_widget)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("height: 20px;")
        buttons_layout.addWidget(self.progress_bar)
        
        controls_layout.addLayout(source_layout)
        controls_layout.addStretch(1)
        controls_layout.addLayout(buttons_layout)
        
        left_layout.addWidget(controls_group)
        
        # Right side (with tabs for better organization)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(15)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Create icons for tabs
        status_icon = QIcon.fromTheme("dialog-information") or QIcon("icons/status.png")
        settings_icon = QIcon.fromTheme("preferences-system") or QIcon("icons/settings.png")
        
        # Status tab
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        status_layout.setSpacing(15)
        
        # Current detection - styled as a card
        detection_frame = QFrame()
        detection_frame.setFrameShape(QFrame.StyledPanel)
        detection_frame.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            padding: 10px;
            margin: 5px;
        """)
        detection_layout = QVBoxLayout(detection_frame)
        
        detection_title = QLabel("Current Detection")
        detection_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        detection_title.setAlignment(Qt.AlignCenter)
        
        self.result_label = QLabel("No anomalies detected")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.result_label.setStyleSheet("""
            color: white;
            background-color: #20c997;
            font-size: 14px;
            padding: 10px;
            border-radius: 6px;
        """)
        self.result_label.setMinimumHeight(50)
        
        detection_layout.addWidget(detection_title)
        detection_layout.addWidget(self.result_label)

        # Log display - styled as a card
        log_frame = QFrame()
        log_frame.setFrameShape(QFrame.StyledPanel)
        log_frame.setStyleSheet("""
            background-color: white;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            padding: 10px;
            margin: 5px;
        """)
        log_layout = QVBoxLayout(log_frame)
        
        log_title = QLabel("Activity Log")
        log_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            border: 1px solid #ced4da;
            border-radius: 4px;
            background-color: #f8f9fa;
            font-family: Consolas, Monaco, monospace;
            font-size: 12px;
            line-height: 1.4;
        """)
        
        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_text)
        
        status_layout.addWidget(detection_frame)
        status_layout.addWidget(log_frame, 1)  # Give log more stretch
        
        
         #Add tabs to the tab widget
        tab_widget.addTab(status_tab, status_icon, "Status")
        
        right_layout.addWidget(tab_widget)
        
        # Add the panels to the splitter
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(right_widget)
        
        # Set the initial sizes of the panels
        self.main_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        # Add log message
        self.log_message("Application started. Please load a model to begin.")
    
    def load_model(self):
        try:
            # Create a loading overlay
            self.loading_overlay = QWidget(self)
            self.loading_overlay.setGeometry(self.rect())
            self.loading_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
            
            loading_label = QLabel("Loading model...", self.loading_overlay)
            loading_label.setStyleSheet("color: white; font-size: 18px; background-color: transparent;")
            loading_label.setAlignment(Qt.AlignCenter)
            
            layout = QVBoxLayout(self.loading_overlay)
            layout.addWidget(loading_label)
            self.loading_overlay.setLayout(layout)
            self.loading_overlay.show()
            
            # Process events to show the overlay
            QApplication.processEvents()
            
            model_path = "Model.h5"
            if not os.path.exists(model_path):
                model_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model Files (*.h5)")
                if not model_path:
                    self.loading_overlay.hide()
                    self.log_message("No model selected. Please load a model to continue.", "warning")
                    return False
            
            self.log_message("Loading model...", "info")
            self.model = load_model(model_path)
            self.categories_labels = {
                'Fighting': 0, 
                'Stealing': 1, 
                'Arson': 2, 
                'Explosion': 3
            }
            self.labels_categories = {v: k for k, v in self.categories_labels.items()}
            
            # Hide the overlay
            self.loading_overlay.hide()
            
            self.log_message("Model loaded successfully!", "success")
            return True
        except Exception as e:
            if hasattr(self, 'loading_overlay'):
                self.loading_overlay.hide()
            self.log_message(f"Error loading model: {str(e)}", "error")
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            return False
    
    def start_detection(self):
        if self.model is None:
            if not self.load_model():
                return
        
        # Disable start button and enable stop button
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        source = self.source_combo.currentText()
        
        
          
        if source == "Video":
            video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", 
                                                     "Video Files (*.mp4 *.avi *.mov *.mkv)")
            if not video_path:
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                return
                
            self.log_message(f"Processing video: {os.path.basename(video_path)}", "info")
            self.progress_bar.setVisible(True)
            
            self.video_thread = VideoThread(video_path)
            self.video_thread.change_pixmap_signal.connect(self.update_image)
            self.video_thread.progress_signal.connect(self.update_progress)
            self.video_thread.finished_signal.connect(self.video_finished)
            self.video_thread.start()
            
            # Start the processing timer
            self.process_timer = QTimer()
            self.process_timer.timeout.connect(self.process_current_frame)
            self.process_timer.start(100)  # Process every 100ms
    
    def stop_detection(self):
        # Stop the webcam thread if it's running
        if self.webcam_thread is not None and self.webcam_thread.isRunning():
            self.webcam_thread.stop()
            self.webcam_thread = None
        
        # Stop the video thread if it's running
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread = None
        
        # Stop the processing timer if it's running
        if hasattr(self, 'process_timer') and self.process_timer.isActive():
            self.process_timer.stop()
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Re-enable controls
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        self.log_message("Detection stopped", "info")
    
    def update_image(self, frame):
        self.current_frame = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.image_label.width(), self.image_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def video_finished(self):
        self.log_message("Video processing completed", "success")
        self.stop_detection()
    
    def process_current_frame(self):
        if self.current_frame is not None:
            self.process_image(self.current_frame)
    
    def process_image(self, frame):
        try:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resize to match what the model expects
            resized_frame = cv2.resize(gray_frame, (50, 50))
            
            # Reshape the image for CNN and RNN inputs
            image_cnn = resized_frame.reshape((1,) + resized_frame.shape + (1,))
            image_rnn = resized_frame.reshape((1, -1, 1))
            
            # Make prediction
            prediction = self.model.predict([image_cnn, image_rnn], verbose=0)
            
            # Get the label with highest probability
            label_idx = np.argmax(prediction)
            confidence = prediction[0][label_idx]
            
            category = self.labels_categories[label_idx]
            confidence_percentage = confidence * 100
            
            # Update UI with the prediction
            self.update_prediction_ui(category, confidence_percentage)
            
            # Check if this is an anomaly that should trigger an alert
            self.check_for_alert(category, confidence_percentage, frame)
            
        except Exception as e:
            self.log_message(f"Error processing image: {str(e)}", "error")
    
    def update_prediction_ui(self, category, confidence):
        # Only update UI if confidence meets the detection threshold
        if confidence / 100 >= self.detection_threshold:
            # Update the result label
            self.result_label.setText(f"Detected: {category} ({confidence:.1f}%)")
            
            # Set color based on confidence
            if confidence / 100 >= self.confidence_threshold:
                # High confidence anomaly
                self.result_label.setStyleSheet("""
                    color: white;
                    background-color: #fa5252;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 10px;
                    border-radius: 6px;
                """)
            elif confidence / 100 >= 0.7:
                # Medium confidence
                self.result_label.setStyleSheet("""
                    color: white;
                    background-color: #ff922b;
                    font-weight: bold;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 6px;
                """)
            else:
                # Low confidence
                self.result_label.setStyleSheet("""
                    color: white;
                    background-color: #20c997;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 6px;
                """)
        else:
            # Below detection threshold - no detection to display
            self.result_label.setText("No anomalies detected")
            self.result_label.setStyleSheet("""
                color: white;
                background-color: #20c997;
                font-size: 14px;
                padding: 10px;
                border-radius: 6px;
            """)
    
    def check_for_alert(self, category, confidence, frame):
        current_time = time.time()
        
        # Skip if confidence is below threshold
        if confidence / 100 < self.confidence_threshold:
            return
            
        # Check cooldown period
        if category in self.last_alert_time:
            time_since_last_alert = current_time - self.last_alert_time[category]
            if time_since_last_alert < self.alert_cooldown:
                # Still in cooldown period
                return
        
        # This is a new alert that passed the cooldown period
        self.log_message(f"ALERT: {category} detected with {confidence:.1f}% confidence!", "alert")
        
        # Update last alert time
        self.last_alert_time[category] = current_time
        
        # Save the frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_filename = f"alert_{category}_{timestamp}.jpg"
        cv2.imwrite(frame_filename, frame)
        
        # Send email alert
        self.send_anomaly_alert(category, confidence, frame_filename)
    
    
    def send_anomaly_alert(self, category, confidence, image_path):
        if not self.email_address or not self.email_password or not self.recipient_email:
            self.log_message("Email settings not configured. Alert not sent.", "warning")
            return
            
        try:
            subject = f"SECURITY ALERT: {category} Detected!"
            
            # Prepare email body
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px;">
                    <h2 style="color: #d63031; text-align: center;">Security Alert: {category} Detected</h2>
                    <p><b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><b>Confidence:</b> {confidence:.1f}%</p>
                    <p style="font-size: 18px; color: #d63031; text-align: center; font-weight: bold; margin-top: 20px;">
                        Please check the security system immediately!
                    </p>
                </div>
            </body>
            </html>
            """
            
            # Send email in a separate thread to not block the UI
            threading.Thread(
                target=self._send_email_thread, 
                args=(subject, body, image_path)
            ).start()
            
        except Exception as e:
            self.log_message(f"Error sending alert email: {str(e)}", "error")
    
    def _send_email_thread(self, subject, body, image_path):
        try:
            self.send_alert_email(subject, body, image_path)
            self.log_message("Alert email sent successfully", "success")
        except Exception as e:
            self.log_message(f"Failed to send alert email: {str(e)}", "error")
    
    def send_alert_email(self, subject, body, image_path=None):
        message = MIMEMultipart()
        message['From'] = self.email_address
        message['To'] = self.recipient_email
        message['Subject'] = subject
        
        message.attach(MIMEText(body, 'html'))
        
        # Attach image if provided
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as file:
                img = MIMEImage(file.read())
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                message.attach(img)
        
        # Send the email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(self.email_address, self.email_password)
        server.send_message(message)
        server.quit()
    
    def log_message(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # HTML styling for different message types
        if level == "error":
            style = "color: #e03131; font-weight: bold;"
        elif level == "warning":
            style = "color: #f76707;"
        elif level == "success":
            style = "color: #2b8a3e; font-weight: bold;"
        elif level == "alert":
            style = "color: #e03131; font-weight: bold; background-color: #ffe3e3; padding: 2px 4px; border-radius: 3px;"
        else:  # info
            style = "color: #1864ab;"
            
        log_entry = f'<span style="color: #868e96;">[{timestamp}]</span> <span style="{style}">{message}</span>'
        self.log_text.append(log_entry)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        print(f"[{timestamp}] {message}")  # Plain text for console
    
    def resizeEvent(self, event):
        # Adjust UI elements when the window is resized
        super().resizeEvent(event)
        
        # Recalculate splitter proportions
        if hasattr(self, 'main_splitter'):
            total_width = self.main_splitter.width()
            self.main_splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])
            
        # Update loading overlay if it exists
        if hasattr(self, 'loading_overlay') and self.loading_overlay.isVisible():
            self.loading_overlay.setGeometry(self.rect())

if __name__ == "__main__":
    # Fix for high DPI screens
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
    app = QApplication(sys.argv)
    window = AnomalyDetectionApp()
    window.show()
    sys.exit(app.exec_())