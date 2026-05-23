import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
import time
import os # Added for path joining
import random
import math
import traceback
from collections import deque
from PIL import Image
from torchvision import transforms

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QVBoxLayout, QHBoxLayout, QGroupBox, QSlider, 
                             QPushButton, QGridLayout, QFileDialog, QStatusBar, QSizePolicy, QFrame)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QPointF, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QBrush

from model import LightweightAgeEstimator
from config import Config, ROOT_DIR # Added ROOT_DIR
from experiment import (
    artifact_path,
    build_training_metadata,
    checkpoint_metadata_mismatches,
    compatible_best_model_paths,
    format_metadata_mismatches,
    load_model_state_package,
)
from utils import DLDLProcessor

# ================= 样式表 =================
STYLESHEET = """
QMainWindow { 
    /* 背景透明 */
}

QGroupBox { 
    color: #00e0ff; 
    font-weight: bold; 
    border: 1px solid #555; 
    border-radius: 10px; 
    font-family: 'Segoe UI', 'Microsoft YaHei';
    font-size: 22px; 
    margin-top: 40px; 
    background-color: rgba(30, 30, 30, 210); 
    padding-top: 25px; 
    padding-bottom: 15px;
    padding-left: 15px;
    padding-right: 15px;
}

QGroupBox::title { 
    subcontrol-origin: margin; 
    subcontrol-position: top left; 
    padding: 0 5px; 
    left: 15px; 
    top: 0px; 
}

QLabel { 
    color: #e0e0e0; 
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    font-size: 16px;
    background: transparent;
}

QPushButton { 
    background-color: rgba(14, 99, 156, 220); 
    color: white; 
    border: 1px solid #0e639c;
    border-radius: 6px; 
    padding: 10px; 
    font-weight: bold; 
    font-size: 16px;
}
QPushButton:hover { 
    background-color: #1177bb; 
    border: 1px solid #4fc1ff; 
}
QPushButton:pressed { background-color: #094770; }
QPushButton:disabled { background-color: rgba(51, 51, 51, 180); color: #888; border: 1px solid #333; }
QPushButton#btn_stop { background-color: rgba(197, 48, 48, 220); border: 1px solid #c53030; }
QPushButton#btn_stop:hover { background-color: #e53e3e; border: 1px solid #ff6b6b; }

QSlider::groove:horizontal { 
    border: 1px solid #555; 
    height: 6px; 
    background: rgba(24, 24, 24, 150); 
    margin: 2px 0; 
    border-radius: 3px; 
}
QSlider::handle:horizontal { 
    background: #00e0ff; 
    border: 1px solid #00e0ff; 
    width: 20px;
    height: 20px; 
    margin: -7px 0; 
    border-radius: 10px; 
}
QFrame[frameShape="4"] { color: #666; }
"""

# ================= 粒子特效系统 =================
class Particle:
    def __init__(self, w, h):
        self.x = random.random() * w
        self.y = random.random() * h
        self.vx = (random.random() - 0.5) * 4.0
        self.vy = (random.random() - 0.5) * 4.0
        self.radius = random.random() * 2 + 1.0
        colors = [QColor(0, 224, 255), QColor(255, 0, 255), QColor(0, 100, 255), QColor(50, 255, 50), QColor(255, 255, 255)]
        self.color = random.choice(colors)
        self.color.setAlpha(150) 

    def update(self, w, h, mx, my):
        dx = self.x - mx
        dy = self.y - my
        dist_sq = dx*dx + dy*dy
        if dist_sq < 40000: 
            dist = math.sqrt(dist_sq)
            if dist > 0:
                force = (200 - dist) / 200 * 2.0
                self.vx += (dx / dist) * force
                self.vy += (dy / dist) * force
        self.x += self.vx
        self.y += self.vy
        if self.x < 0: self.x = w
        if self.x > w: self.x = 0
        if self.y < 0: self.y = h
        if self.y > h: self.y = 0
        self.vx *= 0.99
        self.vy *= 0.99

class ParticleBackground(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.particles = []
        self.mouse_pos = QPoint(-1000, -1000)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30) 
        self.num_particles = 400
        self.initialized = False
        
    def resizeEvent(self, event):
        w = self.width()
        h = self.height()
        if not self.initialized or len(self.particles) < self.num_particles:
            current_len = len(self.particles)
            for _ in range(self.num_particles - current_len):
                self.particles.append(Particle(w, h))
            self.initialized = True
        super().resizeEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        super().mouseMoveEvent(event)

    def update_animation(self):
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0: return
        mx = self.mouse_pos.x()
        my = self.mouse_pos.y()
        for p in self.particles:
            p.update(w, h, mx, my)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 25)) 
        for i in range(min(len(self.particles), 100)):
            p1 = self.particles[i]
            for j in range(i + 1, min(len(self.particles), 100)):
                p2 = self.particles[j]
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                if abs(dx) > 100 or abs(dy) > 100: continue
                dist_sq = dx*dx + dy*dy
                if dist_sq < 10000:
                    opacity = int((1 - dist_sq / 10000) * 100)
                    if opacity > 0:
                        pen = QPen(QColor(100, 200, 255, opacity))
                        pen.setWidth(1)
                        painter.setPen(pen)
                        painter.drawLine(QPointF(p1.x, p1.y), QPointF(p2.x, p2.y))
        painter.setPen(Qt.NoPen)
        for p in self.particles:
            painter.setBrush(QBrush(p.color))
            painter.drawEllipse(QPointF(p.x, p.y), p.radius, p.radius)

# ================= 工具函数 =================
def draw_distribution_chart(prob_dist, width=400, height=200, peak_age=None, exp_age=None, max_age=100):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30) 
    margin_left, margin_right, margin_bottom, margin_top = 40, 15, 30, 15
    graph_w = width - margin_left - margin_right
    graph_h = height - margin_bottom - margin_top
    
    prob_dist = prob_dist.cpu().numpy()
    max_prob = np.max(prob_dist)
    if max_prob > 0:
        scaled_probs = (prob_dist / max_prob) * (graph_h * 0.9)
    else:
        scaled_probs = prob_dist

    base_y, base_x = height - margin_bottom, margin_left
    tick_color, text_color = (80, 80, 80), (200, 200, 200)
    
    cv2.line(canvas, (base_x, base_y), (width - margin_right, base_y), tick_color, 1)
    cv2.line(canvas, (base_x, base_y), (base_x, margin_top), tick_color, 1)

    step = 20 if max_age >= 80 else 10
    for age in range(0, max_age + 1, step):
        x_ratio = age / float(max_age)
        x_pos = base_x + int(x_ratio * graph_w)
        cv2.line(canvas, (x_pos, base_y), (x_pos, base_y + 5), tick_color, 1)
        label = str(age)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        text_x = max(0, min(x_pos - text_w // 2, width - text_w))
        cv2.putText(canvas, label, (text_x, base_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    y_labels = [("0.0", 0.0), ("0.5", 0.5), ("1.0", 1.0)]
    for label, ratio in y_labels:
        y_pos = base_y - int(ratio * (graph_h * 0.9))
        cv2.line(canvas, (base_x, y_pos), (base_x - 5, y_pos), tick_color, 1)
        if ratio > 0:
            cv2.line(canvas, (base_x, y_pos), (width - margin_right, y_pos), (50, 50, 50), 1)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, label, (base_x - text_w - 8, y_pos + text_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    points = []
    for i, prob in enumerate(scaled_probs):
        x = base_x + int((i / float(max_age)) * graph_w)
        y = base_y - int(prob)
        points.append((x, y))
    pts = np.array(points, np.int32)
    p_start = np.array([[base_x, base_y]], np.int32)
    p_end = np.array([[points[-1][0], base_y]], np.int32)
    pts_fill = np.concatenate((p_start, pts, p_end))
    
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_fill], (0, 224, 255))
    cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
    cv2.polylines(canvas, [pts], False, (0, 224, 255), 2, cv2.LINE_AA)

    def draw_indicator(age_val, color, label_text):
        if age_val < 0 or age_val > max_age: return
        x_ratio = age_val / float(max_age)
        x_pos = base_x + int(x_ratio * graph_w)
        cv2.line(canvas, (x_pos, margin_top), (x_pos, base_y), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x_pos + 5, margin_top), (x_pos + 5 + text_w, margin_top + text_h + 5), (40, 40, 40), -1)
        cv2.putText(canvas, label_text, (x_pos + 5, margin_top + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if peak_age is not None: draw_indicator(peak_age, (255, 0, 255), "Peak")
    if exp_age is not None: draw_indicator(exp_age, (0, 255, 255), "Mean")
    
    return canvas

# ================= Multi-Scale TTA (6x) =================
def multi_scale_tta(images, model, base_size=224):
    """
    激进 Multi-Scale TTA: 3个尺度 (0.9, 1.0, 1.1) x 2 (原始 + 翻转) = 6x 平均
    用于 Demo 提升预测精度
    """
    scales = [0.9, 1.0, 1.1]
    all_probs = []

    for scale in scales:
        if scale != 1.0:
            new_size = int(base_size * scale)
            resized = F.interpolate(images, size=new_size, mode='bilinear', align_corners=False)
            if new_size > base_size:
                start = (new_size - base_size) // 2
                resized = resized[:, :, start:start+base_size, start:start+base_size]
            else:
                pad = (base_size - new_size) // 2
                resized = F.pad(resized, (pad, base_size-new_size-pad, pad, base_size-new_size-pad), mode='reflect')
        else:
            resized = images
        
        # 原始
        logits = model(resized)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
        
        # 水平翻转
        flipped = torch.flip(resized, dims=[3])
        logits_flip = model(flipped)
        probs_flip = F.softmax(logits_flip, dim=1)
        all_probs.append(probs_flip)
    
    # 平均 6 个预测
    return torch.stack(all_probs, dim=0).mean(dim=0)


def load_compatible_weights(model, cfg, model_path, device, seed=42):
    state_dict, checkpoint = load_model_state_package(model_path, device)
    expected_metadata = build_training_metadata(cfg, seed)
    mismatches = checkpoint_metadata_mismatches(checkpoint, expected_metadata)
    if mismatches:
        raise RuntimeError(format_metadata_mismatches(mismatches))
    model.load_state_dict(state_dict)
    model.eval()


def default_model_path(cfg, device, seed=42):
    exact_path = artifact_path(ROOT_DIR, "best_model", cfg, seed, ".pth")
    if os.path.exists(exact_path):
        return exact_path
    compatible, incompatible = compatible_best_model_paths(ROOT_DIR, cfg, seed=seed, device=device)
    if compatible:
        print(f"DEBUG: Auto-discovered compatible model: {os.path.basename(compatible[0])}", flush=True)
        return compatible[0]
    if incompatible:
        print("⚠️ Found best_model files, but none match the current Config metadata.", flush=True)
        for path, reason in incompatible[:5]:
            print(f"   rejected {os.path.basename(path)}: {reason}", flush=True)
    return None

# ================= 后台处理线程 =================
class WorkerThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_dist_signal = pyqtSignal(QImage)
    update_age_signal = pyqtSignal(float, str) 
    finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._run_flag = False
        self.mode = 'camera'
        self.source_path = 0
        
        self.crop_scale = 1.5
        self.shift_vertical = 0.30
        self.shift_horizontal = 0.05
        self.calibration_offset = 0.0
        self.smooth_window = 15
        
        # Global Bias Correction (Calibrated)
        # Strategy: Median Bias = -0.25. Use Beta=-0.25.
        self.bias_alpha = 1.0
        self.bias_beta = -0.25
        
        self.cfg = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        print(f"🚀 后端设备: {self.device}", flush=True)
        print("DEBUG: Initializing Model...", flush=True)
        
        try:
            print("DEBUG: Instantiating LightweightAgeEstimator...", flush=True)
            self.model = LightweightAgeEstimator(self.cfg).to(self.device)
            
            model_path = default_model_path(self.cfg, self.device)

            if model_path:
                print(f"⏳ Loading model from: {model_path}", flush=True)
                load_compatible_weights(self.model, self.cfg, model_path, self.device)
                self.model_loaded = True
                print("DEBUG: Model loaded successfully", flush=True)
            else:
                print("⚠️ No compatible model found. Please use 'Load Model' button.", flush=True)
        except Exception as e:
            print(f"❌ Model Initialization/Load failed: {e}", flush=True)
            traceback.print_exc()
            self.model_loaded = False
    

        print("DEBUG: Initializing DLDLProcessor...", flush=True)
        self.dldl_tools = DLDLProcessor(self.cfg)
        print("DEBUG: Initializing Transforms...", flush=True)
        self.transform = transforms.Compose([
            transforms.Resize((self.cfg.img_size, self.cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("DEBUG: Initializing MediaPipe...", flush=True)
        self.mp_face_detection = mp.solutions.face_detection
        self.mean_buffer = deque(maxlen=self.smooth_window)
        self.peak_buffer = deque(maxlen=self.smooth_window)

    def set_source(self, mode, path=0):
        self.mode = mode
        self.source_path = path
        self.mean_buffer.clear()
        self.peak_buffer.clear()

    def run(self):
        self._run_flag = True
        if not self.model_loaded:
            print("❌ Model is not loaded; inference is disabled.", flush=True)
            self._run_flag = False
            self.finished_signal.emit()
            return
        cap = None 
        
        try:
            if self.mode == 'image':
                origin_frame = cv2.imdecode(np.fromfile(self.source_path, dtype=np.uint8), -1)
                if origin_frame is None: 
                    print("❌ 错误：无法读取图片")
                    self._run_flag = False
                cap = None
            else:
                if self.mode == 'camera':
                    # Force DirectShow for Windows to avoid NVIDIA driver hang/lag
                    cap = cv2.VideoCapture(self.source_path, cv2.CAP_DSHOW)
                    # Optimize caching for low latency
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                else: 
                    cap = cv2.VideoCapture(self.source_path)
                
                if not cap.isOpened():
                    print("❌ 错误：无法打开摄像头")
                    self._run_flag = False
            
            face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
            
            while self._run_flag:
                if not self._run_flag: break

                if self.mode == 'image':
                    frame = origin_frame.copy()
                    time.sleep(0.05)
                else:
                    if cap is None or not cap.isOpened(): break
                    ret, frame = cap.read()
                    if not ret:
                        if self.mode == 'video':
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            break
                    if self.mode == 'camera':
                        frame = cv2.flip(frame, 1)

                if frame is None or frame.size == 0: continue

                h, w, ch = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                
                if results.detections:
                    for detection in results.detections:
                        if not self._run_flag: break

                        bboxC = detection.location_data.relative_bounding_box
                        x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                                       int(bboxC.width * w), int(bboxC.height * h)
                        
                        cx, cy = x + bw // 2, y + bh // 2
                        cy -= int(bh * self.shift_vertical)
                        cx -= int(bw * self.shift_horizontal)
                        new_w = int(bw * self.crop_scale)
                        new_h = int(bh * self.crop_scale)
                        x1 = max(0, cx - new_w // 2)
                        y1 = max(0, cy - new_h // 2)
                        x2 = min(w, x1 + new_w)
                        y2 = min(h, y1 + new_h)
                        
                        if x2 <= x1 or y2 <= y1: continue

                        face_roi = frame[y1:y2, x1:x2]
                        
                        if face_roi.size > 0 and new_w > 20 and new_h > 20:
                            pil_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                            input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                # === Multi-Scale TTA (6x: 0.9/1.0/1.1 + flip) ===
                                probs = multi_scale_tta(input_tensor, self.model, base_size=self.cfg.img_size)
                                
                                raw_mean = self.dldl_tools.expectation_regression(probs).item()
                                raw_peak = torch.argmax(probs, dim=1).item()
                            
                            if self.mode == 'image':
                                final_mean = (raw_mean * self.bias_alpha + self.bias_beta) + self.calibration_offset
                            else:
                                self.mean_buffer.append(raw_mean)
                                avg_smooth = sum(self.mean_buffer) / len(self.mean_buffer)
                                final_mean = (avg_smooth * self.bias_alpha + self.bias_beta) + self.calibration_offset
                            
                            # Logic Lock: Clamp to physical limits (1-100)
                            final_mean = max(1, min(final_mean, 100))
                            
                            # --- 计算峰值区间 (概率 >= 90% * Max_Prob) ---
                            probs_np = probs.cpu().numpy()[0]
                            max_prob = np.max(probs_np)
                            threshold = max_prob * 0.90
                            
                            valid_indices = np.where(probs_np >= threshold)[0]
                            
                            if len(valid_indices) > 0:
                                low_idx = np.min(valid_indices)
                                high_idx = np.max(valid_indices)
                            else:
                                low_idx = raw_peak
                                high_idx = raw_peak
                            
                            final_low = low_idx + self.calibration_offset
                            final_high = high_idx + self.calibration_offset
                            
                            interval_str = f"{int(final_low)}-{int(final_high)}"
                            
                            self.update_age_signal.emit(final_mean, interval_str)
                            
                            dist_img = draw_distribution_chart(probs[0], width=400, height=200, peak_age=raw_peak, exp_age=raw_mean, max_age=self.cfg.max_age)
                            h_d, w_d, ch_d = dist_img.shape
                            qt_dist_img = QImage(dist_img.data, w_d, h_d, ch_d * w_d, QImage.Format_RGB888).copy()
                            self.update_dist_signal.emit(qt_dist_img)
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    blank = np.zeros((200, 400, 3), dtype=np.uint8)
                    blank[:] = (30, 30, 30)
                    qt_blank = QImage(blank.data, 400, 200, 400*3, QImage.Format_RGB888).copy()
                    self.update_dist_signal.emit(qt_blank)
                    self.update_age_signal.emit(0.0, "No Face")

                if not self._run_flag: break

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                self.change_pixmap_signal.emit(qt_image)

        except Exception as e:
            print(f"⚠️ 线程异常: {e}")
            traceback.print_exc()

        finally:
            if cap is not None:
                cap.release()
            self.finished_signal.emit()

    def stop(self):
        self._run_flag = False

    def set_params(self, scale, shift_v, shift_h):
        self.crop_scale = scale
        self.shift_vertical = shift_v
        self.shift_horizontal = shift_h

    def set_calibration_offset(self, val):
        self.calibration_offset = val

    def load_new_model(self, model_path):
        print(f"🔄 Loading new model from: {model_path}", flush=True)
        try:
            # Re-instantiate to be safe or just load weights? 
            # Ideally just load weights if arch is same. Assuming same arch for now.
            load_compatible_weights(self.model, self.cfg, model_path, self.device)
            self.model_loaded = True
            print("✅ New model loaded successfully", flush=True)
            return True, "Success"
        except Exception as e:
            self.model_loaded = False
            print(f"❌ Failed to load model: {e}", flush=True)
            return False, str(e)

# ================= 主界面 =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("表观年龄估计系统 v17.0 (Safe & Final)")
        self.setStyleSheet(STYLESHEET)
        self.secret_calibration = 0.0

        screen = QApplication.primaryScreen().geometry()
        initial_w = int(screen.width() * 0.7)
        initial_h = int(screen.height() * 0.7)
        self.resize(initial_w, initial_h)
        self.move((screen.width() - initial_w)//2, (screen.height() - initial_h)//2)

        self.particle_bg = ParticleBackground()
        self.setCentralWidget(self.particle_bg)
        
        main_layout = QHBoxLayout(self.particle_bg)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # 左侧
        video_group = QGroupBox(" 👁‍🗨 实时监测 (Live Monitor) ")
        video_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(10, 30, 10, 10) 
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: #000; border: 2px solid #333; border-radius: 6px;")
        video_layout.addWidget(self.video_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        self.btn_camera = QPushButton("📷 启动摄像头")
        self.btn_camera.clicked.connect(self.start_camera)
        
        self.btn_file = QPushButton("📂 打开本地文件")
        self.btn_file.clicked.connect(self.open_file)
        
        self.btn_load_model = QPushButton("🧩 加载模型")
        self.btn_load_model.clicked.connect(self.load_model_action)
        self.btn_load_model.setStyleSheet("background-color: rgba(60, 60, 60, 220); border: 1px solid #555;")
        
        self.btn_stop = QPushButton("🛑 停止检测")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.stop_thread_action)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_file)
        btn_layout.addWidget(self.btn_load_model)
        btn_layout.addWidget(self.btn_stop)
        video_layout.addLayout(btn_layout)
        video_group.setLayout(video_layout)
        
        # 右侧
        control_panel = QWidget()
        control_panel.setStyleSheet("background: transparent;") 
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(460)
        
        # 1. 预测结果
        res_group = QGroupBox(" 📊 预测结果 (Prediction) ")
        res_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum) 
        res_layout = QVBoxLayout()
        res_layout.setContentsMargins(10, 35, 10, 15) 
        
        dual_res_layout = QHBoxLayout()
        mean_layout = QVBoxLayout()
        self.mean_val = QLabel("READY")
        self.mean_val.setAlignment(Qt.AlignCenter)
        self.mean_val.setMinimumHeight(80)
        self.mean_val.setStyleSheet("font-family: 'Arial'; font-size: 65px; color: #444; font-weight: bold;")
        self.mean_lbl = QLabel("综合评估 (Mean)")
        self.mean_lbl.setAlignment(Qt.AlignCenter)
        self.mean_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        mean_layout.addWidget(self.mean_val)
        mean_layout.addWidget(self.mean_lbl)
        
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #666;")
        
        peak_layout = QVBoxLayout()
        self.peak_val = QLabel("--")
        self.peak_val.setAlignment(Qt.AlignCenter)
        self.peak_val.setMinimumHeight(80)
        self.peak_val.setStyleSheet("font-family: 'Arial'; font-size: 65px; color: #444; font-weight: bold;")
        self.peak_lbl = QLabel("峰值区间 (>90%)")
        self.peak_lbl.setAlignment(Qt.AlignCenter)
        self.peak_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        peak_layout.addWidget(self.peak_val)
        peak_layout.addWidget(self.peak_lbl)
        
        dual_res_layout.addLayout(mean_layout)
        dual_res_layout.addWidget(line)
        dual_res_layout.addLayout(peak_layout)
        res_layout.addLayout(dual_res_layout)
        res_group.setLayout(res_layout)
        
        # 2. 分布图
        dist_group = QGroupBox(" 📈 概率分布 (Uncertainty) ")
        dist_layout = QVBoxLayout()
        dist_layout.setContentsMargins(10, 35, 10, 10)
        self.dist_label = QLabel()
        self.dist_label.setMinimumSize(400, 200)
        self.dist_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.dist_label.setScaledContents(True)
        self.dist_label.setStyleSheet("background-color: #252526; border-radius: 6px;") 
        self.dist_label.setAlignment(Qt.AlignCenter)
        dist_layout.addWidget(self.dist_label)
        dist_group.setLayout(dist_layout)
        
        # 3. 几何参数
        tune_group = QGroupBox(" 🛠️ 几何修正 (Geometry Tuning) ")
        tune_layout = QGridLayout()
        tune_layout.setVerticalSpacing(25)
        tune_layout.setContentsMargins(15, 40, 15, 20)
        
        self.lbl_scale = QLabel("裁剪缩放: +50%")
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setRange(100, 200); self.slider_scale.setValue(150)
        self.slider_scale.valueChanged.connect(self.update_params)
        
        self.lbl_shift_v = QLabel("垂直修正: +30%")
        self.slider_shift_v = QSlider(Qt.Horizontal)
        self.slider_shift_v.setRange(-50, 50); self.slider_shift_v.setValue(30)
        self.slider_shift_v.valueChanged.connect(self.update_params)
        
        self.lbl_shift_h = QLabel("水平修正: +5%")
        self.slider_shift_h = QSlider(Qt.Horizontal)
        self.slider_shift_h.setRange(-50, 50); self.slider_shift_h.setValue(5)
        self.slider_shift_h.valueChanged.connect(self.update_params)
        
        tune_layout.addWidget(self.lbl_scale, 0, 0)
        tune_layout.addWidget(self.slider_scale, 1, 0)
        tune_layout.addWidget(self.lbl_shift_v, 2, 0)
        tune_layout.addWidget(self.slider_shift_v, 3, 0)
        tune_layout.addWidget(self.lbl_shift_h, 4, 0)
        tune_layout.addWidget(self.slider_shift_h, 5, 0)
        tune_group.setLayout(tune_layout)
        
        control_layout.addWidget(res_group, stretch=2)
        control_layout.addWidget(dist_group, stretch=2)
        control_layout.addWidget(tune_group, stretch=1)

        main_layout.addWidget(video_group, stretch=3)
        main_layout.addWidget(control_panel, stretch=1)
        
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet("color: #888; font-size: 11px; padding: 5px; background: transparent;")
        self.setStatusBar(self.status_bar)
        
        device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        self.status_bar.showMessage(f"System Ready | Backend: {device_name} | SysOffset: {self.secret_calibration:+.1f}")

        self.thread = WorkerThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_dist_signal.connect(self.update_dist)
        self.thread.update_age_signal.connect(self.update_age_dual)
        self.thread.finished_signal.connect(self.on_thread_finished)
        
        self.setFocusPolicy(Qt.StrongFocus)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Up: self.secret_calibration += 1.0
        elif key == Qt.Key_Down: self.secret_calibration -= 1.0
        elif key == Qt.Key_Right: self.secret_calibration += 0.1
        elif key == Qt.Key_Left: self.secret_calibration -= 0.1
        self.thread.set_calibration_offset(self.secret_calibration)
        device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        self.status_bar.showMessage(f"Processing | Backend: {device_name} | SysOffset: {self.secret_calibration:+.1f}")
        
    def start_camera(self):
        if self.thread.isRunning(): return
        if not self.thread.model_loaded:
            self.status_bar.showMessage("❌ No compatible model loaded.", 3000)
            return
        self.btn_camera.setEnabled(False)
        self.btn_file.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.thread.set_source('camera', 0)
        self.thread.start()

    def load_model_action(self):
        if self.thread.isRunning():
            self.status_bar.showMessage("⚠️ Please stop detection first!", 3000)
            return
            
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", ".", "PyTorch Model (*.pth)")
        if file_path:
            self.status_bar.showMessage(f"Loading model: {os.path.basename(file_path)}...")
            success, msg = self.thread.load_new_model(file_path)
            if success:
                self.status_bar.showMessage(f"✅ Model Loaded: {os.path.basename(file_path)}")
            else:
                self.status_bar.showMessage(f"❌ Load Failed: {msg}")

    def open_file(self):
        if self.thread.isRunning(): return
        if not self.thread.model_loaded:
            self.status_bar.showMessage("❌ No compatible model loaded.", 3000)
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "媒体文件 (*.jpg *.png *.mp4 *.avi)")
        if file_path:
            ext = file_path.lower().split('.')[-1]
            mode = 'image' if ext in ['jpg', 'png', 'jpeg', 'bmp'] else 'video'
            self.btn_camera.setEnabled(False)
            self.btn_file.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.thread.set_source(mode, file_path)
            self.thread.start()

    def stop_thread_action(self):
        if self.thread.isRunning():
            self.thread.stop()
            self.btn_stop.setText("停止中...")
            self.btn_stop.setEnabled(False)

    def on_thread_finished(self):
        self.btn_camera.setEnabled(True)
        self.btn_file.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("🛑 停止检测")
        self.video_label.clear()
        self.mean_val.setText("READY")
        self.peak_val.setText("--")
        self.mean_val.setStyleSheet("font-family: 'Arial'; font-size: 65px; color: #444; font-weight: bold;")
        self.peak_val.setStyleSheet("font-family: 'Arial'; font-size: 65px; color: #444; font-weight: bold;")

    def update_params(self):
        raw_scale = self.slider_scale.value()
        raw_shift_v = self.slider_shift_v.value()
        raw_shift_h = self.slider_shift_h.value()

        scale = raw_scale / 100.0
        shift_v = raw_shift_v / 100.0
        shift_h = raw_shift_h / 100.0

        scale_pct = raw_scale - 100
        self.lbl_scale.setText(f"裁剪缩放: {scale_pct:+d}%")
        self.lbl_shift_v.setText(f"垂直修正: {raw_shift_v:+d}%")
        self.lbl_shift_h.setText(f"水平修正: {raw_shift_h:+d}%")

        self.thread.set_params(scale, shift_v, shift_h)

    def update_image(self, qt_img):
        scaled_pixmap = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def update_dist(self, qt_img):
        self.dist_label.setPixmap(QPixmap.fromImage(qt_img))

    def update_age_dual(self, mean_age, interval_str):
        if mean_age > 0:
            self.mean_val.setText(f"{mean_age:.1f}")
            self.peak_val.setText(f"{interval_str}")
            self.mean_val.setStyleSheet("font-family: 'Arial'; font-size: 65px; color: #00e0ff; font-weight: bold;")
            self.peak_val.setStyleSheet("font-family: 'Arial'; font-size: 50px; color: #ff00ff; font-weight: bold;")
        else:
            self.mean_val.setText("...")
            self.peak_val.setText("...")

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.thread.stop()
            if not self.thread.wait(2000):
                self.thread.terminate()
        event.accept()

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
