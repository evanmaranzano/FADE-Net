import streamlit as st
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from PIL import Image
from torchvision import transforms
import time
import pandas as pd
from collections import deque

import glob
import os

# Import local modules
from model import LightweightAgeEstimator
from config import Config, ROOT_DIR
from experiment import (
    artifact_path,
    build_training_metadata,
    checkpoint_metadata_mismatches,
    compatible_best_model_paths,
    format_metadata_mismatches,
    load_model_state_package,
    populate_runtime_model_metadata,
)
from utils import DLDLProcessor

# ================= Configuration & Styles =================
st.set_page_config(
    page_title="AgeEstimator Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modern/Light CSS
st.markdown("""
    <style>
    /* Global Background & Font */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f0f2f6 100%);
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }
    
    /* Headers - Clean Dark Blue */
    h1, h2, h3 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    
    /* Metrics / Stat Boxes - White Cards with Shadow */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    div[data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-weight: bold;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 5px 5px 0 0;
        color: #6b7280;
        font-weight: 600;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f3f4f6;
        color: #1f2937;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        color: #2563eb;
        border-bottom: 3px solid #2563eb;
    }
    
    /* Images */
    img {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ================= Helper Functions =================
def get_available_cameras():
    """
    Returns a static list of camera indices to avoid slow probing.
    Probing triggers heavy driver initialization (e.g. NVIDIA Broadcast) which causes lag and log spam.
    """
    # Simply return 0 and 1 as options. 99% of users use index 0.
    camera_map = {
        "Default Camera (0)": 0,
        "Secondary Camera (1)": 1
    }
    return camera_map

# ================= Helper Functions =================
def draw_distribution_chart(prob_dist, width=400, height=200, peak_age=None, exp_age=None, max_age=100):
    """
    Draws the probability distribution chart (Modern Light Style).
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (255, 255, 255)  # White Background
    margin_left, margin_right, margin_bottom, margin_top = 40, 15, 30, 15
    graph_w = width - margin_left - margin_right
    graph_h = height - margin_bottom - margin_top
    
    # Handle tensor or numpy
    if isinstance(prob_dist, torch.Tensor):
        prob_dist = prob_dist.cpu().numpy()
        
    max_prob = np.max(prob_dist)
    if max_prob > 0:
        scaled_probs = (prob_dist / max_prob) * (graph_h * 0.9)
    else:
        scaled_probs = prob_dist

    base_y, base_x = height - margin_bottom, margin_left
    tick_color = (200, 200, 200) # Light Grey
    text_color = (80, 80, 80)    # Dark Grey
    fill_color = (235, 99, 37)   # Blue (BGR: 235, 99, 37 -> #2563eb)
    fill_color_alpha = (250, 200, 180) # Lighter Blue 
    
    # Axes
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

    # Plot Curve
    points = []
    for i, prob in enumerate(scaled_probs):
        x = base_x + int((i / float(max_age)) * graph_w)
        y = base_y - int(prob)
        points.append((x, y))
    pts = np.array(points, np.int32)
    p_start = np.array([[base_x, base_y]], np.int32)
    p_end = np.array([[points[-1][0], base_y]], np.int32)
    pts_fill = np.concatenate((p_start, pts, p_end))
    
    # Clean Fill
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_fill], fill_color) 
    cv2.addWeighted(overlay, 0.2, canvas, 0.8, 0, canvas) # softer transparency
    cv2.polylines(canvas, [pts], False, fill_color, 2, cv2.LINE_AA)

    # Indicators
    def draw_indicator(age_val, color, label_text):
        if age_val < 0 or age_val > max_age: return
        x_ratio = age_val / float(max_age)
        x_pos = base_x + int(x_ratio * graph_w)
        cv2.line(canvas, (x_pos, margin_top), (x_pos, base_y), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # White box background for text
        cv2.rectangle(canvas, (x_pos + 5, margin_top), (x_pos + 5 + text_w, margin_top + text_h + 5), (255, 255, 255), -1)
        cv2.rectangle(canvas, (x_pos + 5, margin_top), (x_pos + 5 + text_w, margin_top + text_h + 5), color, 1)
        cv2.putText(canvas, label_text, (x_pos + 5, margin_top + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if peak_age is not None: draw_indicator(peak_age, (255, 0, 255), "Peak") # Magenta
    if exp_age is not None: draw_indicator(exp_age, (235, 99, 37), "Mean")  # Blue
    
    return canvas

@st.cache_resource
def load_model(model_path=None):
    """Loads and caches the model to avoid reloading."""
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Updated instantiation to match src/model.py
    model = LightweightAgeEstimator(cfg).to(device)
    try:
        if model_path is None:
            exact_path = artifact_path(ROOT_DIR, "best_model", cfg, 42, ".pth")
            if os.path.exists(exact_path):
                model_path = exact_path
            else:
                compatible, _incompatible = compatible_best_model_paths(ROOT_DIR, cfg, seed=42, device=device)
                model_path = compatible[0] if compatible else None
            
        if model_path is None:
            raise RuntimeError("No compatible best_model checkpoint found for current Config.")
        print(f"⏳ Loading model from: {model_path}")
        state_dict, checkpoint = load_model_state_package(model_path, device)
        expected_metadata = build_training_metadata(cfg, 42)
        mismatches = checkpoint_metadata_mismatches(checkpoint, expected_metadata)
        if mismatches:
            raise RuntimeError(format_metadata_mismatches(mismatches))
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None, None, None
        
    dldl_tools = DLDLProcessor(cfg)
    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    return model, dldl_tools, transform, face_detection, device

# ================= Multi-Scale TTA (6x) =================
def multi_scale_tta(images, model):
    """
    激进 Multi-Scale TTA: 3个尺度 (0.9, 1.0, 1.1) x 2 (原始 + 翻转) = 6x 平均
    用于 Demo 提升预测精度
    """
    scales = [0.9, 1.0, 1.1]
    all_probs = []
    
    for scale in scales:
        if scale != 1.0:
            new_size = int(224 * scale)
            resized = F.interpolate(images, size=new_size, mode='bilinear', align_corners=False)
            if new_size > 224:
                start = (new_size - 224) // 2
                resized = resized[:, :, start:start+224, start:start+224]
            else:
                pad = (224 - new_size) // 2
                resized = F.pad(resized, (pad, 224-new_size-pad, pad, 224-new_size-pad), mode='reflect')
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

# ================= Inference Logic =================
def process_single_image(image_np, model, dldl_tools, transform, face_detection, device, cfg, params):
    h, w, c = image_np.shape
    results = face_detection.process(image_np)
    
    detections_data = []
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                           int(bboxC.width * w), int(bboxC.height * h)
            
            # Apply shifts
            cx, cy = x + bw // 2, y + bh // 2
            cy -= int(bh * params['shift_v'])
            cx -= int(bw * params['shift_h'])
            new_w = int(bw * params['scale'])
            new_h = int(bh * params['scale'])
            x1 = max(0, cx - new_w // 2)
            y1 = max(0, cy - new_h // 2)
            x2 = min(w, x1 + new_w)
            y2 = min(h, y1 + new_h)
            
            if x2 <= x1 or y2 <= y1: continue
                
            face_roi = image_np[y1:y2, x1:x2]
            
            if face_roi.size > 0:
                pil_img = Image.fromarray(face_roi)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Multi-Scale TTA (6x: 0.9/1.0/1.1 + flip)
                    probs = multi_scale_tta(input_tensor, model)
                    
                    # Age Calculation
                    raw_mean = dldl_tools.expectation_regression(probs).item()
                    raw_peak = torch.argmax(probs, dim=1).item()
                    
                    # Bias Correction
                    bias_alpha, bias_beta = 1.0, -0.25
                    final_mean = max(1, min((raw_mean * bias_alpha + bias_beta), 100))
                    
                    # Interval
                    probs_np = probs.cpu().numpy()[0]
                    threshold = np.max(probs_np) * 0.90
                    valid_indices = np.where(probs_np >= threshold)[0]
                    interval_str = f"{np.min(valid_indices)}-{np.max(valid_indices)}" if len(valid_indices) > 0 else f"{raw_peak}"
                    
                    # Visualization Assets
                    dist_chart = draw_distribution_chart(probs[0], peak_age=raw_peak, exp_age=raw_mean, max_age=cfg.max_age)
                    
                    detections_data.append({
                        "bbox": (x1,y1,x2,y2),
                        "mean_age": final_mean,
                        "interval": interval_str,
                        "dist_chart": dist_chart
                    })
                    
    return detections_data

# ================= Main App =================
def main():
    st.title("🧬 AgeEstimator Pro")
    st.markdown("### Next-Gen Biological Age Assessment System")

    # Sidebar: Model Selection
    st.sidebar.markdown("## 🧠 Model Selection")
    
    cfg = Config()
    device_for_scan = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        populate_runtime_model_metadata(cfg)
    except Exception as exc:
        st.error(f"Model config initialization failed: {exc}")
        st.info("Check the current Python environment and install project requirements if needed.")
        st.stop()
    compatible_models, incompatible_models = compatible_best_model_paths(ROOT_DIR, cfg, seed=42, device=device_for_scan)
    model_files = [os.path.basename(f) for f in compatible_models]
    
    if not model_files:
        st.error("No metadata-compatible best_model checkpoint found in root directory.")
        if incompatible_models:
            with st.expander("Rejected checkpoints"):
                for path, reason in incompatible_models:
                    st.write(f"{os.path.basename(path)}: {reason}")
        st.stop()
        
    selected_model = st.sidebar.selectbox("Choose Model", model_files, index=0 if len(model_files) > 0 else 0)
    selected_model_path = os.path.join(ROOT_DIR, selected_model)

    # Load Model
    model, dldl_tools, transform, face_detection, device = load_model(selected_model_path)

    if model is None:
        st.stop()

    # Sidebar: Controls
    st.sidebar.markdown("## ⚙️ Control Panel")
    
    with st.sidebar.expander("🛠️ Geometry Tuning", expanded=False):
        crop_scale = st.slider("Crop Scale", 1.0, 2.0, 1.5, 0.05)
        shift_v = st.slider("Vertical Shift", -0.5, 0.5, 0.3, 0.05)
        shift_h = st.slider("Horizontal Shift", -0.5, 0.5, 0.05, 0.05)
    
    params = {'scale': crop_scale, 'shift_v': shift_v, 'shift_h': shift_h}
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Single Image", "📂 Batch Processing", "🎥 Live Video"])
    
    # --- Tab 1: Single Image ---
    with tab1:
        col_in, col_res = st.columns([1, 1.5])
        
        with col_in:
            st.markdown("#### Input Source")
            src_method = st.radio("Method", ["Upload", "Snapshot"], label_visibility="collapsed")
            img_input = None
            if src_method == "Upload":
                f = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
                if f: img_input = np.array(Image.open(f).convert('RGB'))
            else:
                f = st.camera_input("Take Snapshot")
                if f: img_input = np.array(Image.open(f).convert('RGB'))
        
        if img_input is not None:
            with st.spinner("Analyzing biometric features..."):
                results = process_single_image(img_input, model, dldl_tools, transform, face_detection, device, cfg, params)
            
            with col_res:
                if not results:
                    st.warning("No face detected.")
                    st.image(img_input, caption="Input", use_container_width=True)
                else:
                    # Only show first face for Single Mode
                    r = results[0]
                    x1,y1,x2,y2 = r['bbox']
                    # Draw Box
                    vis_img = img_input.copy()
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    st.image(vis_img, caption="Biometric Scan", use_container_width=True)
                    
                    # Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("Predicted Age", f"{r['mean_age']:.1f}", delta="Mean")
                    m2.metric("Confidence Interval", r['interval'], delta=">90% Prob")
                    
                    st.image(r['dist_chart'], caption="Uncertainty Distribution", use_container_width=True)

    # --- Tab 2: Batch Processing ---
    with tab2:
        st.markdown("#### Bulk Analysis")
        files = st.file_uploader("Upload Multiple Images", type=["jpg", "png"], accept_multiple_files=True)
        
        if files:
            if st.button(f"Process {len(files)} Images"):
                progress_bar = st.progress(0)
                batch_results = []
                
                st.markdown("---")
                # Grid Layout for gallery
                cols = st.columns(4)
                
                for i, file in enumerate(files):
                    img = np.array(Image.open(file).convert('RGB'))
                    res = process_single_image(img, model, dldl_tools, transform, face_detection, device, cfg, params)
                    
                    age_val = "N/A"
                    if res:
                        age_val = f"{res[0]['mean_age']:.1f}"
                        # Draw simplified box
                        r = res[0]
                        cv2.rectangle(img, (r['bbox'][0], r['bbox'][1]), (r['bbox'][2], r['bbox'][3]), (0, 255, 0), 5)
                    
                    batch_results.append({"Filename": file.name, "Predicted Age": age_val})
                    
                    # Add to gallery
                    with cols[i % 4]:
                        st.image(img, caption=f"{file.name}\nAge: {age_val}", use_container_width=True)
                    
                    progress_bar.progress((i + 1) / len(files))
                
                st.markdown("### Summary Report")
                df = pd.DataFrame(batch_results)
                st.dataframe(df, use_container_width=True)

    # --- Tab 3: Live Video ---
    with tab3:
        st.markdown("#### Real-time Biometric Monitor")
        
        # Camera Selection
        cam_options = get_available_cameras()
        selected_cam_name = st.sidebar.selectbox("Select Camera", options=list(cam_options.keys()))
        c_idx = cam_options[selected_cam_name]
        
        run_video = st.toggle("Start Feed", value=False)
        
        place_video = st.empty()
        place_stats = st.empty()
        
        if run_video:
            # Force DirectShow for Windows to ensure instant startup
            cap = cv2.VideoCapture(c_idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce latency
            
            if not cap.isOpened():
                st.error(f"Cannot open camera {c_idx}")
            else:
                while run_video:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    start_t = time.time()
                    
                    # Inference
                    res_list = process_single_image(frame_rgb, model, dldl_tools, transform, face_detection, device, cfg, params)
                    
                    # Draw Overlay
                    for r in res_list:
                        x1,y1,x2,y2 = r['bbox']
                        # Cyberpunk Box
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        # Label
                        label = f"Age: {r['mean_age']:.1f}"
                        cv2.putText(frame_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    fps = 1.0 / (time.time() - start_t)
                    
                    # Update UI
                    place_video.image(frame_rgb, channels="RGB", use_container_width=True)
                    place_stats.markdown(f"**System Status**: 🟢 ONLINE | **FPS**: {fps:.1f} | **Faces**: {len(res_list)}")
                    
                    # Streamlit rerun trick is not needed for loop, but we need to check toggle
                    # Since streamlit script reruns top-down on interaction, the 'while' loop blocks interaction.
                    # Proper way in streamlit is somewhat hacky or using 'streamlit-webrtc'.
                    # For simple local demo, we just rely on the loop. To stop, user must toggle OFF (which triggers rerun).
                    # Actually, inside the loop, we can't detect UI changes easily without experimental_rerun or session state checks.
                    # We will just run for 10000 frames or until error for this basic demo.
                    
                    # BETTER: Use a placeholder button to stop?
                    # Streamlit's st.toggle state won't update *during* this loop unless we interrupt.
                    # We'll just run. User can click "Stop" which sets session state on next run? No.
                    # User interprets "Toggle Off" -> Streamlit kills the script and restarts?
                    # Yes, Streamlit "Stop" button or unchecking toggle usually triggers a re-run/halt.
                    time.sleep(0.01)
                    
            cap.release()

if __name__ == "__main__":
    main()
