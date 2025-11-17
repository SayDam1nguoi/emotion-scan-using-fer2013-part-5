# -*- coding: utf-8 -*-
"""
Emotion Detection
Main detection logic cho face và emotion detection
"""
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import time

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Import from refactored modules
from .config import (EMOTIONS, NEGATIVE_EMOTIONS, CONFIDENCE_THRESHOLDS, 
                     DETECTION_CONFIG, TRAINING_CONFIG, PATHS)
from .calibration import calibrate_predictions, apply_confidence_filter
from .lighting import analyze_lighting, get_lighting_summary
from .preprocessing import preprocess_face_for_detection, load_dataset_from_fer2013
from .model import (build_mobilenet_model, train_model, save_model, 
                   load_model, create_callbacks)
from .affectiva_scoring import AffectivaScorer
from .dress_analysis import analyze_dress_complete, get_dress_summary

# Global pose dress detector (singleton)
_pose_dress_detector = None

def get_pose_dress_detector():
    """Get or create pose dress detector (singleton)"""
    global _pose_dress_detector
    if _pose_dress_detector is None:
        try:
            from .dress_detector_pose import PoseDressDetector
            _pose_dress_detector = PoseDressDetector()
        except Exception as e:
            print(f"Warning: Could not load pose dress detector: {e}")
            _pose_dress_detector = False  # Mark as failed
    return _pose_dress_detector if _pose_dress_detector is not False else None

def analyze_dress_improved(face_box, frame, use_pose=True):
    """
    Improved dress analysis using pose detection
    Falls back to color-based if pose fails
    """
    if use_pose:
        detector = get_pose_dress_detector()
        if detector:
            try:
                result = detector.detect_clothing(frame, face_box)
                if result and result.get('confidence', 0) > 0.5:
                    # Convert to compatible format
                    return {
                        'color_name': result['color_name'],
                        'dress_type': result.get('coverage', 'unknown'),
                        'combined_score': result['formality_score'],
                        'score': result['formality_score'],
                        'confidence': result['confidence'],
                        'method': 'pose'
                    }
            except Exception as e:
                print(f"Pose detection failed: {e}")
    
    # Fallback to original method
    result = analyze_dress_complete(face_box, frame, use_yolo=True)
    if result:
        result['method'] = 'color'
    return result
from .background_analysis import analyze_background_cleanliness
from utils.visualization import plot_emotion_chart
from utils.suggestions import get_detailed_suggestions
from utils.mode_suggestions import get_mode_specific_suggestions
from utils.file_utils import save_results
from ui.training_window import show_loading_window


def get_emotion_status(emotion, confidence):
    """
    Lay mo ta trang thai cam xuc
    
    Args:
        emotion: ten cam xuc
        confidence: do tin cay
    
    Returns:
        status string
    """
    status_map = {
        'Happy': 'Tuoi cuoi',
        'Sad': 'U sau',
        'Angry': 'Buc minh',
        'Neutral': 'Nghiem tuc'
    }
    
    return status_map.get(emotion, emotion)


def calculate_dress_score(dress_samples):
    """
    Tinh diem trang phuc tu dress samples (color + type)
    
    Args:
        dress_samples: list of dress analysis results
    
    Returns:
        dress_score (0-100)
    """
    if not dress_samples:
        return 70.0  # Default score
    
    # Get average combined score (color + type)
    scores = [sample.get('combined_score', sample.get('score', 70)) for sample in dress_samples]
    avg_score = np.mean(scores)
    
    # Get most common color and type
    colors = [sample.get('color_name', 'Unknown') for sample in dress_samples]
    types = [sample.get('dress_type', 'Unknown') for sample in dress_samples]
    
    from collections import Counter
    most_common_color = Counter(colors).most_common(1)[0][0] if colors else 'Unknown'
    most_common_type = Counter(types).most_common(1)[0][0] if types else 'Unknown'
    
    return avg_score


def calculate_lighting_score(lighting_samples):
    """
    Tính điểm ánh sáng từ lighting samples
    
    Args:
        lighting_samples: list of brightness values
    
    Returns:
        lighting_score (0-100)
    """
    if not lighting_samples:
        return 70.0
    
    avg_brightness = np.mean(lighting_samples)
    
    # Optimal range: 100-160
    if 100 <= avg_brightness <= 160:
        score = 100.0
    elif 80 <= avg_brightness < 100:
        # Too dark
        score = 50 + (avg_brightness - 80) * 2.5
    elif 160 < avg_brightness <= 180:
        # Slightly bright
        score = 100 - (avg_brightness - 160) * 2
    elif avg_brightness < 80:
        # Very dark
        score = max(20, avg_brightness * 0.625)
    else:
        # Too bright
        score = max(20, 100 - (avg_brightness - 180) * 3)
    
    return min(100, max(0, score))


def start_detection(csv_path, video_path=None, camera_id=0, analysis_mode='candidate'):
    """
    Main detection function - quét cảm xúc từ camera hoặc video
    
    Args:
        csv_path: path to dataset CSV
        video_path: path to video file (None for camera)
        camera_id: camera ID (default 0)
        analysis_mode: 'recruiter' or 'candidate' (default 'candidate')
    """
    try:
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Lỗi", "Chọn file dataset (.csv) trước!")
            return

        # Detect mode: camera (live) or video (CV)
        detection_mode = 'live' if video_path is None else 'video'
        
        # Detect dataset type
        dataset_name = "CK+ Extended" if "ck" in os.path.basename(csv_path).lower() else "FER2013"
        messagebox.showinfo("Thông báo", f"Tải dữ liệu từ {dataset_name} và huấn luyện model...")
        
        # Load dataset
        images, labels = load_dataset_from_fer2013(csv_path)
        
        # Filter out invalid labels (must be 0-6 for 7 emotions)
        valid_mask = labels < len(EMOTIONS)
        images = images[valid_mask]
        labels = labels[valid_mask]
        
        if len(images) == 0:
            messagebox.showerror("Lỗi", "Dataset không có dữ liệu hợp lệ!")
            return
        
        # Prepare data
        images = np.stack([images]*3, axis=-1)
        labels_cat = to_categorical(labels, num_classes=len(EMOTIONS))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_cat, 
            test_size=TRAINING_CONFIG['test_size'], 
            random_state=TRAINING_CONFIG['random_state'], 
            stratify=labels
        )

        # Load or train model
        if os.path.exists(PATHS['model']):
            model = load_model()
        else:
            model = build_mobilenet_model()
            
            # Show training window
            root, win, progress, epoch_label, status_label, time_label = show_loading_window(
                TRAINING_CONFIG['epochs']
            )
            
            # Create callbacks
            callbacks = create_callbacks(
                progress, epoch_label, status_label, time_label, 
                TRAINING_CONFIG['epochs'], root
            )
            
            # Train model
            train_model(model, X_train, y_train, X_test, y_test, callbacks)
            
            # Save model
            save_model(model)
            
            win.destroy()
            root.update()

        # Show loading window after model is loaded
        import tkinter as tk
        loading_root = tk.Tk()
        loading_root.withdraw()
        
        loading_win = tk.Toplevel(loading_root)
        loading_win.title("Đang tải...")
        loading_win.geometry("400x150")
        loading_win.resizable(False, False)
        loading_win.configure(bg="#ffffff")
        
        # Center window
        loading_win.update_idletasks()
        x = (loading_win.winfo_screenwidth() // 2) - 200
        y = (loading_win.winfo_screenheight() // 2) - 75
        loading_win.geometry(f"400x150+{x}+{y}")
        
        tk.Label(loading_win, text="⏳", font=("Segoe UI Emoji", 32),
                bg="#ffffff").pack(pady=(20, 10))
        message = "Đang khởi động camera..." if video_path is None else "Đang tải video..."
        tk.Label(loading_win, text=message,
                font=("Segoe UI", 12, "bold"), bg="#ffffff").pack()
        tk.Label(loading_win, text="Vui lòng đợi trong giây lát",
                font=("Segoe UI", 10), bg="#ffffff", fg="#7f8c8d").pack(pady=5)
        
        loading_win.update()
        
        # Start emotion detection
        cap = cv2.VideoCapture(camera_id if video_path is None else video_path)
        
        # MediaPipe face detector (faster than MTCNN)
        from core.face_detector_mediapipe import MediaPipeFaceDetector
        from core.config import FACE_DETECTION_CONFIG, VIDEO_DETECTION_CONFIG
        
        # SỬ DỤNG CONFIG KHÁC NHAU CHO VIDEO VÀ CAMERA
        if video_path is not None:
            # VIDEO: Dễ dàng hơn để quét đầy đủ
            detector = MediaPipeFaceDetector(
                min_detection_confidence=VIDEO_DETECTION_CONFIG['min_detection_confidence'],
                model_selection=FACE_DETECTION_CONFIG['model_selection'],
                min_tracking_confidence=VIDEO_DETECTION_CONFIG['min_tracking_confidence']
            )
            # Cập nhật min_consecutive_frames cho video
            detector.min_consecutive_frames = VIDEO_DETECTION_CONFIG['min_consecutive_frames']
            detect_every_n_frames = VIDEO_DETECTION_CONFIG['detect_every_n_frames']
            min_face_size_override = VIDEO_DETECTION_CONFIG['min_face_size']
            face_threshold_override = VIDEO_DETECTION_CONFIG['face_threshold']
            history_size = VIDEO_DETECTION_CONFIG['history_size']
            edge_margin = VIDEO_DETECTION_CONFIG['edge_margin']
        else:
            # CAMERA: Nghiêm ngặt hơn
            detector = MediaPipeFaceDetector(
                min_detection_confidence=FACE_DETECTION_CONFIG['min_detection_confidence'],
                model_selection=FACE_DETECTION_CONFIG['model_selection'],
                min_tracking_confidence=FACE_DETECTION_CONFIG.get('min_tracking_confidence', 0.5)
            )
            detect_every_n_frames = DETECTION_CONFIG['detect_every_n_frames']
            min_face_size_override = DETECTION_CONFIG['min_face_size']
            face_threshold_override = 0.7
            history_size = DETECTION_CONFIG['history_size']
            edge_margin = 0.1
        
        # Close loading window
        loading_win.destroy()
        loading_root.destroy()
        emotion_counts = [0]*len(EMOTIONS)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Video playback speed
        video_speed = DETECTION_CONFIG.get('video_playback_speed', 1.0)
        if video_path is not None:
            delay = int(1000 / (fps * video_speed))
        else:
            delay = int(1000 / fps)
        
        # FPS optimization
        frame_count = 0
        detection_scale = DETECTION_CONFIG['detection_scale']
        lighting_interval = DETECTION_CONFIG['lighting_check_interval']
        last_faces = []  # Cache faces từ lần detect trước
        
        # Temporal smoothing (đã set ở trên dựa vào video/camera)
        emotion_history = []
        # history_size đã được set ở trên
        
        # LOẠI BỎ FALSE SCAN "NHẤP NHÁY" - Track thời gian xuất hiện
        face_first_seen = None  # Thời điểm face xuất hiện lần đầu
        face_stable = False     # Face đã ổn định chưa (xuất hiện đủ lâu)
        min_stable_duration = VIDEO_DETECTION_CONFIG.get('min_stable_duration', 1.5) if video_path else 0.5
        
        # Lighting analysis
        lighting_samples = []
        
        # Dress color analysis
        dress_color_samples = []
        
        # Background analysis
        background_samples = []
        environment_samples = []
        object_samples = []
        lighting_quality_samples = []  # Task 2.4
        current_face_boxes = []
        
        # Behavior analysis (gesture & focus)
        prev_gray = None
        movement_samples = []
        posture_samples = []
        eye_contact_samples = []
        
        # Attention tracking (only for live mode)
        attention_tracker = None
        if detection_mode == 'live':
            from .attention_tracking import AttentionTracker
            attention_tracker = AttentionTracker()
        
        # FPS tracking
        import time
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        # Create window once (prevent multiple windows)
        window_name = 'Emotion Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            display_frame = frame.copy()
            h_frame, w_frame = display_frame.shape[:2]  # Define early for use in attention tracking
            
            # Analyze lighting (mỗi N frames)
            if frame_count % lighting_interval == 0:
                brightness, light_status, light_color = analyze_lighting(frame)
                lighting_samples.append(brightness)
            
            # Display lighting info
            if lighting_samples:
                cv2.putText(display_frame, f"Anh sang: {light_status} ({brightness:.0f})", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)
            
            # Display FPS và mode
            mode_text = "VIDEO MODE (STRICT)" if video_path is not None else "CAMERA MODE"
            fps_text = f"FPS: {current_fps:.1f} | {mode_text}"
            if video_path is not None:
                fps_text += f" | Conf: {face_threshold_override:.2f} | Min: {min_face_size_override}px"
            cv2.putText(display_frame, fps_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Detect faces (chỉ mỗi N frames để tăng FPS)
            if frame_count % detect_every_n_frames == 0:
                # Resize frame cho detection
                small_frame = cv2.resize(frame, (0, 0), fx=detection_scale, fy=detection_scale)
                
                # Enhance cho low light nếu cần
                if DETECTION_CONFIG.get('enhance_low_light', False):
                    # Check brightness
                    gray_check = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    avg_brightness = np.mean(gray_check)
                    
                    # Nếu ánh sáng thấp (< 100), enhance
                    if avg_brightness < 100:
                        # CLAHE enhancement
                        lab = cv2.cvtColor(small_frame, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                        l = clahe.apply(l)
                        small_frame = cv2.merge([l, a, b])
                        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_LAB2BGR)
                        
                        # Thêm brightness boost nhẹ
                        small_frame = cv2.convertScaleAbs(small_frame, alpha=1.2, beta=10)
                
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_small)
                
                # Scale lại tọa độ về kích thước gốc
                scale_factor = 1.0 / detection_scale
                for f in faces:
                    f['box'] = [int(coord * scale_factor) for coord in f['box']]
                
                last_faces = faces
            else:
                # Sử dụng faces từ lần detect trước
                faces = last_faces
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Collect face boxes for background analysis
            current_face_boxes = [f['box'] for f in faces if f['confidence'] >= CONFIDENCE_THRESHOLDS['face_detection']]
            
            # Analyze background (moi 60 frames)
            if frame_count % 60 == 0:
                bg_result = analyze_background_cleanliness(frame, current_face_boxes)
                if bg_result['score'] > 0:
                    background_samples.append(bg_result)
            
            # Analyze environment (moi 120 frames - it nhan hon)
            if frame_count % 120 == 0:
                from .background_analysis import (
                    analyze_environment, 
                    analyze_inappropriate_objects,
                    analyze_background_lighting
                )
                env_result = analyze_environment(frame, use_deep_learning=True)
                environment_samples.append(env_result)
                
                # Detect inappropriate objects (cung luc voi environment)
                obj_result = analyze_inappropriate_objects(frame)
                object_samples.append(obj_result)
                
                # Analyze background lighting quality (Task 2.4)
                face_boxes_for_lighting = [(f['box'][0], f['box'][1], f['box'][2], f['box'][3]) for f in faces]
                lighting_quality_result = analyze_background_lighting(frame, face_boxes_for_lighting)
                lighting_quality_samples.append(lighting_quality_result)
            
            # Show face detection status
            if len(faces) == 0:
                # RESET face tracking khi không có face
                face_first_seen = None
                face_stable = False
                
                cv2.putText(display_frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, f"Detected {len(faces)} face(s)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            for f in faces:
                x, y, w, h = f['box']
                confidence_face = f['confidence']
                
                # Draw all detected faces with different colors
                # SỬ DỤNG THRESHOLD KHÁC NHAU CHO VIDEO VÀ CAMERA
                # face_threshold_override đã được set ở trên
                
                if confidence_face < face_threshold_override:
                    # Low confidence face - KHÔNG XỬ LÝ (bỏ qua false positives)
                    cv2.rectangle(display_frame, (x,y), (x+w,y+h), (128,128,128), 1)
                    cv2.putText(display_frame, f"Low: {confidence_face:.2f}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
                    continue
                
                # FACE QUALITY CHECK - loại bỏ false positives
                # 1. Kiểm tra aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.7 or aspect_ratio > 1.5:
                    cv2.putText(display_frame, "Bad ratio", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    continue  # Face không đúng tỷ lệ
                
                # 2. Kiểm tra kích thước tối thiểu (sử dụng override cho video)
                if w < min_face_size_override or h < min_face_size_override:
                    cv2.putText(display_frame, "Too small", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    continue  # Face quá nhỏ
                
                # 3. Kiểm tra vị trí (face phải ở trung tâm, không ở rìa)
                # Sử dụng edge_margin khác nhau cho video/camera
                frame_h, frame_w = frame.shape[:2]
                center_x = x + w // 2
                center_y = y + h // 2
                # Bỏ qua faces ở rìa frame (có thể là false positives)
                # Video: edge_margin = 0.05 (dễ dàng hơn), Camera: 0.1 (nghiêm ngặt hơn)
                if center_x < frame_w * edge_margin or center_x > frame_w * (1 - edge_margin):
                    cv2.putText(display_frame, "Edge face", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    continue
                if center_y < frame_h * (edge_margin / 2) or center_y > frame_h * (1 - edge_margin / 2):
                    cv2.putText(display_frame, "Edge face", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                    continue
                
                # Expand face ROI
                margin = int(DETECTION_CONFIG['face_margin'] * max(w, h))
                x1, y1 = max(0, x-margin), max(0, y-margin)
                x2, y2 = min(rgb.shape[1], x+w+margin), min(rgb.shape[0], y+h+margin)
                
                # Validate ROI
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Analyze dress (color + type) moi 30 frames - IMPROVED with pose
                if frame_count % 30 == 0:
                    dress_result = analyze_dress_improved((x, y, w, h), frame, use_pose=True)
                    if dress_result and dress_result.get('combined_score', 0) > 0:
                        dress_color_samples.append(dress_result)
                
                # Preprocess face
                face_roi = rgb[y1:y2, x1:x2]
                
                # Check if face_roi is valid
                if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                    continue
                
                face_roi = preprocess_face_for_detection(face_roi)
                face_img = cv2.resize(face_roi, (48,48))/255.0
                face_img = face_img.reshape(1,48,48,3)
                
                # Predict
                pred = model.predict(face_img, verbose=0)[0]
                
                # Apply probability calibration
                pred = calibrate_predictions(pred)
                
                # Temporal smoothing
                emotion_history.append(pred)
                if len(emotion_history) > history_size:
                    emotion_history.pop(0)
                
                # Average predictions
                avg_pred = np.mean(emotion_history, axis=0)
                
                # Apply calibration again on averaged predictions
                avg_pred = calibrate_predictions(avg_pred)
                
                # KIỂM TRA THỜI GIAN XUẤT HIỆN - Loại bỏ "nhấp nháy"
                current_time = time.time()
                
                if face_first_seen is None:
                    # Face mới xuất hiện
                    face_first_seen = current_time
                    face_stable = False
                else:
                    # Kiểm tra face đã xuất hiện đủ lâu chưa
                    duration = current_time - face_first_seen
                    if duration >= min_stable_duration:
                        face_stable = True
                
                # CHỈ TÍNH CẢM XÚC NẾU FACE ĐÃ ỔN ĐỊNH (xuất hiện đủ lâu)
                if face_stable:
                    # THAY ĐỔI: Tính tổng hợp TẤT CẢ emotions theo probability
                    for i in range(len(EMOTIONS)):
                        emotion_counts[i] += avg_pred[i]  # Cộng probability thay vì count
                
                # Lấy emotion có probability cao nhất để hiển thị
                pred_idx = np.argmax(avg_pred)
                emo = EMOTIONS[pred_idx]
                confidence = avg_pred[pred_idx]
                
                # Luôn valid (không filter)
                is_valid = True
                
                # Get emotion status description
                emotion_status = get_emotion_status(emo, avg_pred[pred_idx])
                
                # Display with different colors
                if face_stable:
                    # Face đã ổn định - hiển thị cảm xúc
                    if is_valid:
                        color = (0,0,255) if emo in NEGATIVE_EMOTIONS else (0,255,0)
                        thickness = 3
                    else:
                        color = (0,165,255)
                        thickness = 2
                    
                    cv2.rectangle(display_frame,(x,y),(x+w,y+h),color,thickness)
                else:
                    # Face chưa ổn định - đang chờ
                    duration = current_time - face_first_seen if face_first_seen else 0
                    remaining = min_stable_duration - duration
                    
                    # Hiển thị màu vàng và thông báo "đang chờ"
                    cv2.rectangle(display_frame,(x,y),(x+w,y+h),(0,255,255),2)
                    cv2.putText(display_frame, f"Dang cho on dinh... {remaining:.1f}s", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
                # Display emotion status (Vietnamese description) - CHI hien thi text
                if is_valid and face_stable:
                    status_y = y + h + 30
                    cv2.putText(display_frame, emotion_status, (x, status_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Track posture and eye contact
                from .behavior_analysis import analyze_posture_simple, estimate_eye_contact
                posture_result = analyze_posture_simple((x, y, w, h), frame.shape)
                posture_samples.append(posture_result)
                
                eye_contact_result = estimate_eye_contact((x, y, w, h), frame.shape)
                eye_contact_samples.append(eye_contact_result)
                
                # Track attention/focus (for live interview mode)
                if attention_tracker is not None:
                    attention_result = attention_tracker.analyze_attention(
                        (x, y, w, h), frame.shape
                    )
                    # Display attention status on frame
                    attention_text = f"Tap trung: {attention_result['attention_score']:.0f}%"
                    cv2.putText(display_frame, attention_text, (10, h_frame - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Track gestures (movement detection)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and len(faces) > 0:
                from .behavior_analysis import detect_hand_movement
                face_box = faces[0]['box'] if faces else None
                movement_result = detect_hand_movement(prev_gray, curr_gray, face_box)
                movement_samples.append(movement_result)
            prev_gray = curr_gray.copy()
            
            # Add instructions at bottom
            instructions = "Press 'q' to quit | 'n' to add note (recruiter)"
            if analysis_mode == 'recruiter':
                instructions += " | Green=Valid | Yellow=Waiting"
            cv2.putText(display_frame, instructions,
                       (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n') and analysis_mode == 'recruiter':
                # Mở cửa sổ ghi nhận xét (chỉ cho recruiter)
                if 'notes_manager' not in locals():
                    from .recruiter_notes import RecruiterNotes, create_notes_window
                    candidate_name = "Candidate_" + datetime.now().strftime('%Y%m%d_%H%M%S')
                    notes_manager = RecruiterNotes(candidate_name, 'video' if video_path else 'camera')
                    notes_window = create_notes_window(notes_manager)
                else:
                    # Đưa cửa sổ lên trên
                    try:
                        notes_window.lift()
                        notes_window.focus_force()
                    except:
                        # Cửa sổ đã đóng, tạo lại
                        notes_window = create_notes_window(notes_manager)

        cap.release()
        cv2.destroyAllWindows()
        
        # Lưu nhận xét của recruiter (nếu có)
        if 'notes_manager' in locals() and len(notes_manager.get_all_notes()) > 0:
            notes_file = notes_manager.save_to_file()
            print(f"\n✅ Đã lưu {len(notes_manager.get_all_notes())} nhận xét vào: {notes_file}")
            
            # Hiển thị tóm tắt
            summary = notes_manager.generate_summary()
            print("\n" + summary)

        # Show results
        if sum(emotion_counts)==0:
            # Nếu là mode recruiter đánh giá ứng viên
            if analysis_mode == 'recruiter':
                # Kiểm tra xem có phải do ánh sáng không
                lighting_issue = False
                if lighting_samples:
                    avg_brightness = np.mean(lighting_samples)
                    # Ánh sáng quá thấp (< 70) hoặc quá cao (> 190)
                    if avg_brightness < 70 or avg_brightness > 190:
                        lighting_issue = True
                
                if lighting_issue:
                    # Nếu do ánh sáng, cho phép bỏ qua
                    msg = ("⚠️ Không phát hiện khuôn mặt trong video\n\n"
                           "Nguyên nhân: Ánh sáng quá thấp/cao\n"
                           f"Độ sáng trung bình: {avg_brightness:.0f}/255\n\n"
                           "Đây có thể là vấn đề kỹ thuật, không phải lỗi của ứng viên.\n\n"
                           "Bạn có muốn:\n"
                           "- BỎ QUA video này (khuyến nghị)\n"
                           "- Xem báo cáo kỹ thuật chi tiết")
                    result = messagebox.askyesnocancel(
                        "Vấn đề ánh sáng", 
                        msg,
                        icon='warning'
                    )
                    if result is None:  # Cancel = Bỏ qua
                        messagebox.showinfo("Đã bỏ qua", 
                                          "Video này đã được bỏ qua do vấn đề ánh sáng.\n\n"
                                          "Khuyến nghị: Yêu cầu ứng viên gửi lại video với ánh sáng tốt hơn.")
                        return
                    elif not result:  # No = Không xem báo cáo
                        return
                    # Yes = Tiếp tục xem báo cáo kỹ thuật
                else:
                    # Không phải do ánh sáng - có thể là vấn đề khác
                    msg = ("CẢNH BÁO: Không phát hiện khuôn mặt trong video!\n\n"
                           "Có thể do:\n"
                           "- Khuôn mặt không rõ ràng\n"
                           "- Video chất lượng kém\n"
                           "- Camera không ổn định\n\n"
                           "⚠️ ĐÂY CÓ THỂ LÀ DẤU HIỆU TIÊU CỰC:\n"
                           "Ứng viên có thể:\n"
                           "- Không chuẩn bị kỹ\n"
                           "- Thiếu chuyên nghiệp\n"
                           "- Video không đạt yêu cầu\n\n"
                           "Bạn có muốn xem báo cáo kỹ thuật không?")
                    result = messagebox.askyesno("Không phát hiện khuôn mặt", msg)
                    if not result:
                        return
                    # Tiếp tục với emotion_counts = 0 để tạo báo cáo kỹ thuật
            else:
                messagebox.showwarning("Ket qua","Khong phat hien khuan mat hop le trong video.\n\nVui long:\n- Dam bao khuan mat ro rang\n- Anh sang du\n- Camera on dinh")
                return
        
        # Normalize emotion_counts (giờ là tổng probabilities)
        # Chuyển về percentages
        total_prob = sum(emotion_counts)
        if total_prob > 0:
            emotion_counts = [count / total_prob * 100 for count in emotion_counts]
        
        # Calculate dress score from samples
        dress_score = calculate_dress_score(dress_color_samples)
        
        # Calculate background score (cleanliness + environment + objects + lighting)
        from .background_analysis import (
            get_cleanliness_summary, 
            get_environment_summary, 
            get_objects_summary,
            get_lighting_quality_summary
        )
        bg_summary = get_cleanliness_summary(background_samples)
        env_summary = get_environment_summary(environment_samples)
        obj_summary = get_objects_summary(object_samples)
        lighting_quality_summary = get_lighting_quality_summary(lighting_quality_samples)
        
        # Combined: 40% cleanliness + 30% environment + 15% objects + 15% lighting quality
        base_score = (bg_summary['avg_score'] * 0.4 + 
                     env_summary['avg_score'] * 0.3 + 
                     lighting_quality_summary['avg_quality_score'] * 0.15)
        
        # Apply object adjustments (penalties and bonuses)
        object_adjustment = obj_summary['avg_penalty'] + obj_summary['avg_bonus']
        background_score = max(0, min(100, base_score + object_adjustment))
        
        # Calculate Affectiva-style scores
        scorer = AffectivaScorer()
        
        # 1. Emotion score (from detection)
        emotion_score = scorer.calculate_emotion_score(emotion_counts)
        
        # 2. Appearance score
        lighting_score = calculate_lighting_score(lighting_samples)
        appearance_score = scorer.calculate_appearance_score(
            dress_score=dress_score,  # From dress color analysis
            background_score=background_score,  # From background analysis
            lighting_score=lighting_score
        )
        
        # 3. Behavior score (with gesture and focus tracking)
        from .behavior_analysis import (
            get_posture_summary,
            calculate_eye_contact_percentage,
            analyze_gesture_frequency,
            detect_fidgeting
        )
        
        posture_summary = get_posture_summary(posture_samples)
        eye_contact_summary = calculate_eye_contact_percentage(eye_contact_samples)
        gesture_summary = analyze_gesture_frequency(movement_samples)
        fidgeting_result = detect_fidgeting(movement_samples) if len(movement_samples) >= 30 else {
            'is_fidgeting': False,
            'fidgeting_score': 90.0,
            'confidence_level': 'Unknown'
        }
        
        # Calculate behavior score with real data
        behavior_score = scorer.calculate_behavior_score(
            eye_contact_score=eye_contact_summary['score'],
            posture_score=posture_summary['avg_score'],
            gesture_score=gesture_summary['gesture_score']
        )
        
        # Create behavior summary for mode suggestions
        behavior_summary = {
            'posture': posture_summary,
            'eye_contact': eye_contact_summary,
            'gestures': gesture_summary,
            'fidgeting': fidgeting_result
        }
        
        # 4. Technical score
        technical_score = scorer.calculate_technical_score()
        
        # Generate simple report - only total score
        affectiva_report = scorer.generate_simple_report()
        
        # Get all suggestions based on mode
        final_emo = EMOTIONS[np.argmax(emotion_counts)]
        
        # Mode-specific suggestions (Live Interview vs CV Video)
        from utils.interview_suggestions import get_suggestions_by_mode
        mode_suggestions = get_suggestions_by_mode(
            detection_mode, 
            emotion_counts, 
            lighting_samples, 
            dress_color_samples
        )
        
        # Basic emotion suggestions
        emotion_suggestions = get_detailed_suggestions(emotion_counts)
        lighting_summary = get_lighting_summary(lighting_samples)
        dress_summary = get_dress_summary(dress_color_samples)
        
        # Get attention summary (only for live mode)
        attention_report = ""
        if attention_tracker is not None:
            from .attention_tracking import format_attention_report
            attention_summary = attention_tracker.get_attention_summary()
            attention_report = f"\n\n{'='*60}\n\n{format_attention_report(attention_summary)}"
        
        # Mode-specific suggestions (Recruiter vs Candidate)
        mode_specific_report = get_mode_specific_suggestions(
            analysis_mode,
            emotion_counts,
            lighting_samples,
            dress_color_samples,
            behavior_summary
        )
        
        # Combine all reports for chart
        mode_title = "PHỎNG VẤN ONLINE" if detection_mode == 'live' else "VIDEO CV"
        full_report = f"{affectiva_report}\n\n{'='*60}\n\n{mode_title}:\n{mode_suggestions}{attention_report}{mode_specific_report}"
        
        # Save results
        save_results(emotion_counts, final_emo)
        
        # Save JSON report
        import json
        json_report = scorer.get_json_report()
        with open('results/affectiva_report.json', 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        # Show completion message FIRST
        messagebox.showinfo("Hoan thanh", "Da hoan thanh viec quet cam xuc!\n\nBieu do va loi khuyen se hien thi ngay.")
        
        # Then plot chart with full report (includes Affectiva + lighting)
        plot_emotion_chart(emotion_counts, full_report)

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error details: {error_msg}")
        print(traceback.format_exc())
        messagebox.showerror("Error", f"An error occurred: {error_msg}")



def start_detection_screen(csv_path, capture_region, analysis_mode='recruiter'):
    """
    Detection function cho screen capture (video call)
    
    Args:
        csv_path: path to dataset CSV
        capture_region: (x, y, width, height) vùng capture
        analysis_mode: 'recruiter' or 'candidate' (default 'recruiter')
    """
    try:
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Lỗi", "Chọn file dataset (.csv) trước!")
            return

        # Always live mode for screen capture
        detection_mode = 'live'
        
        # Detect dataset type
        dataset_name = "CK+ Extended" if "ck" in os.path.basename(csv_path).lower() else "FER2013"
        messagebox.showinfo("Thông báo", f"Tải dữ liệu từ {dataset_name} và huấn luyện model...")
        
        # Load dataset
        images, labels = load_dataset_from_fer2013(csv_path)
        
        # Filter out invalid labels
        valid_mask = labels < len(EMOTIONS)
        images = images[valid_mask]
        labels = labels[valid_mask]
        
        if len(images) == 0:
            messagebox.showerror("Lỗi", "Dataset không có dữ liệu hợp lệ!")
            return
        
        # Prepare data
        images = np.stack([images]*3, axis=-1)
        labels_cat = to_categorical(labels, num_classes=len(EMOTIONS))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_cat, 
            test_size=TRAINING_CONFIG['test_size'], 
            random_state=TRAINING_CONFIG['random_state'], 
            stratify=labels
        )

        # Load or train model
        if os.path.exists(PATHS['model']):
            model = load_model()
        else:
            model = build_mobilenet_model()
            
            # Show training window
            root, win, progress, epoch_label, status_label, time_label = show_loading_window(
                TRAINING_CONFIG['epochs']
            )
            
            # Create callbacks
            callbacks = create_callbacks(
                progress, epoch_label, status_label, time_label, 
                TRAINING_CONFIG['epochs'], root
            )
            
            # Train model
            train_model(model, X_train, y_train, X_test, y_test, callbacks)
            
            # Save model
            save_model(model)
            
            win.destroy()
            root.update()

        # Initialize screen capturer
        from .screen_capture import ScreenCapturer
        capturer = ScreenCapturer()
        capturer.set_roi(*capture_region)
        
        # Show info
        messagebox.showinfo("Bắt đầu", 
                          "Bắt đầu quét video call!\n\n"
                          "- Nhấn 'q' để dừng\n"
                          "- Nhấn 's' để chụp ảnh\n"
                          "- Đảm bảo khuôn mặt người đối diện rõ ràng")

        # Initialize MediaPipe face detector
        from core.face_detector_mediapipe import MediaPipeFaceDetector
        from core.config import FACE_DETECTION_CONFIG
        detector = MediaPipeFaceDetector(
            min_detection_confidence=FACE_DETECTION_CONFIG['min_detection_confidence'],
            model_selection=FACE_DETECTION_CONFIG['model_selection'],
            min_tracking_confidence=FACE_DETECTION_CONFIG.get('min_tracking_confidence', 0.5)
        )
        
        # Tracking variables
        emotion_counts = [0] * len(EMOTIONS)
        emotion_history = []
        history_size = DETECTION_CONFIG['history_size']
        
        # Lighting analysis
        lighting_samples = []
        
        # Dress color analysis
        dress_color_samples = []
        
        # Background analysis
        background_samples = []
        environment_samples = []
        object_samples = []
        lighting_quality_samples = []
        current_face_boxes = []
        
        # Behavior analysis
        prev_gray = None
        movement_samples = []
        posture_samples = []
        eye_contact_samples = []
        
        # Attention tracking
        from .attention_tracking import AttentionTracker
        attention_tracker = AttentionTracker()
        
        # FPS tracking
        import time
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        frame_count = 0
        
        # Create window once (prevent multiple windows)
        window_name = "Screen Capture - Video Call Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # FPS limiter (target 30 FPS for smooth video)
        target_fps = 30
        frame_delay = 1.0 / target_fps
        last_frame_time = time.time()
        
        # Cache for face detection optimization
        last_faces = []
        
        # Main loop
        while True:
            # FPS limiting - maintain consistent frame rate
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
            last_frame_time = time.time()
            
            # Capture frame from screen
            frame = capturer.capture_frame()
            if frame is None:
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            display_frame = frame.copy()
            h_frame, w_frame = display_frame.shape[:2]
            
            # Analyze lighting
            if frame_count % 10 == 0:
                lighting_result = analyze_lighting(frame)
                # analyze_lighting returns tuple (brightness, status, color)
                brightness, status, color = lighting_result
                lighting_samples.append(brightness)
            
            # Detect faces (optimize: detect every 3 frames for better FPS)
            if frame_count % 3 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb)
                last_faces = faces  # Cache for next frames
            else:
                faces = last_faces  # Use cached faces
            
            # Background analysis
            if frame_count % 30 == 0:
                current_face_boxes = [f['box'] for f in faces]
                bg_result = analyze_background_cleanliness(frame, current_face_boxes)
                # bg_result returns dict with 'score', not 'cleanliness_score'
                background_samples.append(bg_result['score'])
                # Note: analyze_background_cleanliness doesn't return these keys
                # environment_samples.append(bg_result.get('category', 'Unknown'))
                # For now, skip these as they're from different analysis
            
            # Process each face
            for face in faces:
                x, y, w, h = face['box']
                confidence_face = face['confidence']
                
                # Skip low confidence faces
                if confidence_face < CONFIDENCE_THRESHOLDS['face_detection']:
                    cv2.rectangle(display_frame, (x,y), (x+w,y+h), (0,255,255), 1)
                    cv2.putText(display_frame, f"Low conf: {confidence_face:.2f}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    continue
                
                # Expand face ROI
                margin = int(DETECTION_CONFIG['face_margin'] * max(w, h))
                x1, y1 = max(0, x-margin), max(0, y-margin)
                x2, y2 = min(rgb.shape[1], x+w+margin), min(rgb.shape[0], y+h+margin)
                
                # Validate ROI
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Analyze dress every 30 frames - IMPROVED with pose
                if frame_count % 30 == 0:
                    dress_result = analyze_dress_improved((x, y, w, h), frame, use_pose=True)
                    if dress_result and dress_result.get('combined_score', 0) > 0:
                        dress_color_samples.append(dress_result)
                
                # Preprocess face
                face_roi = rgb[y1:y2, x1:x2]
                
                if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                    continue
                
                face_roi = preprocess_face_for_detection(face_roi)
                face_img = cv2.resize(face_roi, (48,48))/255.0
                face_img = face_img.reshape(1,48,48,3)
                
                # Predict
                pred = model.predict(face_img, verbose=0)[0]
                
                # Apply calibration
                pred = calibrate_predictions(pred)
                
                # Temporal smoothing
                emotion_history.append(pred)
                if len(emotion_history) > history_size:
                    emotion_history.pop(0)
                
                smoothed_pred = np.mean(emotion_history, axis=0)
                emotion_idx = np.argmax(smoothed_pred)
                confidence = smoothed_pred[emotion_idx]
                
                # Apply confidence filter
                is_valid, _ = apply_confidence_filter(smoothed_pred, emotion_idx)
                
                # Cộng probability cho tất cả emotions (nhất quán với start_detection)
                for i in range(len(EMOTIONS)):
                    emotion_counts[i] += smoothed_pred[i]
                
                # Lấy emotion dominant để hiển thị
                emotion = EMOTIONS[emotion_idx]
                
                if is_valid:
                    # Color based on emotion
                    if emotion in NEGATIVE_EMOTIONS:
                        color = (0, 0, 255)
                    elif emotion == 'Happy':
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    
                    # Draw
                    cv2.rectangle(display_frame, (x,y), (x+w,y+h), color, 2)
                    label = f"{emotion}: {confidence*100:.1f}%"
                    cv2.putText(display_frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Track posture and eye contact
                    from .behavior_analysis import analyze_posture_simple, estimate_eye_contact
                    posture_result = analyze_posture_simple((x, y, w, h), frame.shape)
                    posture_samples.append(posture_result)
                    
                    eye_contact_result = estimate_eye_contact((x, y, w, h), frame.shape)
                    eye_contact_samples.append(eye_contact_result)
                    
                    # Track attention
                    attention_result = attention_tracker.analyze_attention(
                        (x, y, w, h), frame.shape
                    )
                    attention_text = f"Tap trung: {attention_result['attention_score']:.0f}%"
                    cv2.putText(display_frame, attention_text, (10, h_frame - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Track gestures
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and len(faces) > 0:
                from .behavior_analysis import detect_hand_movement
                face_box = faces[0]['box'] if faces else None
                movement_result = detect_hand_movement(prev_gray, curr_gray, face_box)
                movement_samples.append(movement_result)
            prev_gray = curr_gray.copy()
            
            # Add instructions
            cv2.putText(display_frame, "QUET VIDEO CALL - Nhan 'q' de dung", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, h_frame - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show (use existing window)
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                cv2.imwrite(f"results/screenshot_{frame_count}.jpg", display_frame)
                print(f"Saved screenshot_{frame_count}.jpg")
        
        # Cleanup
        capturer.close()
        cv2.destroyAllWindows()
        
        # Generate report (same as normal detection)
        if sum(emotion_counts) == 0:
            messagebox.showwarning("Cảnh báo", "Không phát hiện được cảm xúc nào!")
            return
        
        # Calculate scores
        from .dress_analysis import generate_dress_report
        dress_report = generate_dress_report(dress_color_samples)
        dress_score = dress_report['score']
        
        from .background_analysis import generate_background_report
        bg_report = generate_background_report(
            background_samples, environment_samples, 
            object_samples, lighting_quality_samples
        )
        base_score = bg_report['cleanliness_score']
        obj_summary = bg_report['object_summary']
        object_adjustment = obj_summary['avg_penalty'] + obj_summary['avg_bonus']
        background_score = max(0, min(100, base_score + object_adjustment))
        
        # Calculate Affectiva scores
        scorer = AffectivaScorer()
        emotion_score = scorer.calculate_emotion_score(emotion_counts)
        
        lighting_score = calculate_lighting_score(lighting_samples)
        appearance_score = scorer.calculate_appearance_score(
            dress_score=dress_score,
            background_score=background_score,
            lighting_score=lighting_score
        )
        
        from .behavior_analysis import (
            get_posture_summary,
            calculate_eye_contact_percentage,
            analyze_gesture_frequency,
            detect_fidgeting
        )
        
        posture_summary = get_posture_summary(posture_samples)
        eye_contact_summary = calculate_eye_contact_percentage(eye_contact_samples)
        gesture_summary = analyze_gesture_frequency(movement_samples)
        fidgeting_result = detect_fidgeting(movement_samples) if len(movement_samples) >= 30 else {
            'is_fidgeting': False,
            'fidgeting_score': 90.0,
            'confidence_level': 'Unknown'
        }
        
        behavior_score = scorer.calculate_behavior_score(
            eye_contact_score=eye_contact_summary['score'],
            posture_score=posture_summary['avg_score'],
            gesture_score=gesture_summary['gesture_score']
        )
        
        technical_score = scorer.calculate_technical_score()
        affectiva_report = scorer.generate_simple_report()
        
        # Get suggestions
        final_emo = EMOTIONS[np.argmax(emotion_counts)]
        from utils.interview_suggestions import get_suggestions_by_mode
        mode_suggestions = get_suggestions_by_mode(
            'live',  # Always live for screen capture
            emotion_counts, 
            lighting_samples, 
            dress_color_samples
        )
        
        emotion_suggestions = get_detailed_suggestions(emotion_counts)
        lighting_summary = get_lighting_summary(lighting_samples)
        dress_summary = get_dress_summary(dress_color_samples)
        
        # Get attention summary
        from .attention_tracking import format_attention_report
        attention_summary = attention_tracker.get_attention_summary()
        attention_report = f"\n\n{'='*60}\n\n{format_attention_report(attention_summary)}"
        
        # Combine reports
        mode_title = "PHỎNG VẤN QUA VIDEO CALL"
        full_report = f"{affectiva_report}\n\n{'='*60}\n\n{mode_title}:\n{mode_suggestions}{attention_report}"
        
        # Save results
        save_results(emotion_counts, final_emo)
        
        # Save JSON report
        import json
        json_report = scorer.get_json_report()
        with open('results/affectiva_report.json', 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        # Show completion
        messagebox.showinfo("Hoàn thành", 
                          "Đã hoàn thành việc quét cảm xúc từ video call!\n\n"
                          "Biểu đồ và lời khuyên sẽ hiển thị ngay.")
        
        # Plot chart
        plot_emotion_chart(emotion_counts, full_report)

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error details: {error_msg}")
        print(traceback.format_exc())
        messagebox.showerror("Error", f"An error occurred: {error_msg}")



def start_detection_dual(csv_path, capture_region):
    """
    Detection function cho DUAL mode - quét cả 2 người
    Camera: Bạn (người gọi)
    Screen: Người đối diện (người nhận)
    
    Args:
        csv_path: path to dataset CSV
        capture_region: (x, y, width, height) vùng capture màn hình
    """
    try:
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Lỗi", "Chọn file dataset (.csv) trước!")
            return

        # Load and train model (same as before)
        dataset_name = "CK+ Extended" if "ck" in os.path.basename(csv_path).lower() else "FER2013"
        messagebox.showinfo("Thông báo", f"Tải dữ liệu từ {dataset_name} và huấn luyện model...")
        
        images, labels = load_dataset_from_fer2013(csv_path)
        valid_mask = labels < len(EMOTIONS)
        images = images[valid_mask]
        labels = labels[valid_mask]
        
        if len(images) == 0:
            messagebox.showerror("Lỗi", "Dataset không có dữ liệu hợp lệ!")
            return
        
        images = np.stack([images]*3, axis=-1)
        labels_cat = to_categorical(labels, num_classes=len(EMOTIONS))
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_cat, 
            test_size=TRAINING_CONFIG['test_size'], 
            random_state=TRAINING_CONFIG['random_state'], 
            stratify=labels
        )

        if os.path.exists(PATHS['model']):
            model = load_model()
        else:
            model = build_mobilenet_model()
            root, win, progress, epoch_label, status_label, time_label = show_loading_window(
                TRAINING_CONFIG['epochs']
            )
            callbacks = create_callbacks(
                progress, epoch_label, status_label, time_label, 
                TRAINING_CONFIG['epochs'], root
            )
            train_model(model, X_train, y_train, X_test, y_test, callbacks)
            save_model(model)
            win.destroy()
            root.update()

        # Initialize dual analyzer
        from .dual_detection import DualAnalyzer, format_dual_report
        from .screen_capture import ScreenCapturer
        
        dual_analyzer = DualAnalyzer()
        
        # Initialize screen capturer
        screen_capturer = ScreenCapturer()
        screen_capturer.set_roi(*capture_region)
        
        # Initialize MediaPipe face detector
        from core.face_detector_mediapipe import MediaPipeFaceDetector
        from core.config import FACE_DETECTION_CONFIG
        detector = MediaPipeFaceDetector(
            min_detection_confidence=FACE_DETECTION_CONFIG['min_detection_confidence'],
            model_selection=FACE_DETECTION_CONFIG['model_selection'],
            min_tracking_confidence=FACE_DETECTION_CONFIG.get('min_tracking_confidence', 0.5)
        )
        
        # Start capture threads
        dual_analyzer.start()
        dual_analyzer.start_camera_capture(camera_id=0)
        dual_analyzer.start_screen_capture(screen_capturer)
        
        messagebox.showinfo("Bắt đầu", 
                          "Bắt đầu quét CẢ 2 NGƯỜI!\n\n"
                          "- Trái: Bạn (Camera)\n"
                          "- Phải: Người đối diện (Screen)\n"
                          "- Nhấn 'q' để dừng\n"
                          "- Nhấn 's' để chụp ảnh")
        
        # Tracking variables
        history_size = DETECTION_CONFIG['history_size']
        
        # Person 1 (Camera) tracking
        p1_emotion_history = []
        from .attention_tracking import AttentionTracker
        p1_attention_tracker = AttentionTracker()
        
        # Person 2 (Screen) tracking
        p2_emotion_history = []
        p2_attention_tracker = AttentionTracker()
        
        # FPS tracking
        import time
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        frame_count = 0
        
        # Create window once (prevent multiple windows)
        window_name = "DUAL ANALYSIS - Quet ca 2 nguoi"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # FPS limiter (target 30 FPS for smooth video)
        target_fps = 30
        frame_delay = 1.0 / target_fps
        last_frame_time = time.time()
        
        # Main loop
        while True:
            # FPS limiting - maintain consistent frame rate
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
            last_frame_time = time.time()
            
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Get frames from both sources
            camera_frame = dual_analyzer.get_camera_frame()
            screen_frame = dual_analyzer.get_screen_frame()
            
            # Create display (side by side)
            if camera_frame is not None and screen_frame is not None:
                # Resize to same height
                h_target = 480
                camera_resized = cv2.resize(camera_frame, (int(camera_frame.shape[1] * h_target / camera_frame.shape[0]), h_target))
                screen_resized = cv2.resize(screen_frame, (int(screen_frame.shape[1] * h_target / screen_frame.shape[0]), h_target))
                
                # Combine side by side
                display_frame = np.hstack([camera_resized, screen_resized])
                
                # Process Person 1 (Camera)
                if camera_frame is not None:
                    rgb1 = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    faces1 = detector.detect_faces(rgb1)
                    
                    for face in faces1:
                        x, y, w, h = face['box']
                        confidence_face = face['confidence']
                        
                        if confidence_face < CONFIDENCE_THRESHOLDS['face_detection']:
                            continue
                        
                        # Expand and validate ROI
                        margin = int(DETECTION_CONFIG['face_margin'] * max(w, h))
                        x1, y1 = max(0, x-margin), max(0, y-margin)
                        x2, y2 = min(rgb1.shape[1], x+w+margin), min(rgb1.shape[0], y+h+margin)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Preprocess and predict
                        face_roi = rgb1[y1:y2, x1:x2]
                        if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                            continue
                        
                        face_roi = preprocess_face_for_detection(face_roi)
                        face_img = cv2.resize(face_roi, (48,48))/255.0
                        face_img = face_img.reshape(1,48,48,3)
                        
                        pred = model.predict(face_img, verbose=0)[0]
                        pred = calibrate_predictions(pred)
                        
                        p1_emotion_history.append(pred)
                        if len(p1_emotion_history) > history_size:
                            p1_emotion_history.pop(0)
                        
                        smoothed_pred = np.mean(p1_emotion_history, axis=0)
                        emotion_idx = np.argmax(smoothed_pred)
                        confidence = smoothed_pred[emotion_idx]
                        
                        # CẬP NHẬT THEO PROBABILITY (nhất quán với start_detection)
                        for i in range(len(EMOTIONS)):
                            dual_analyzer.update_person1_emotion_prob(i, smoothed_pred[i])
                        
                        is_valid, _ = apply_confidence_filter(smoothed_pred, emotion_idx)
                        
                        if is_valid:
                            emotion = EMOTIONS[emotion_idx]
                            
                            # Draw on camera part
                            color = (0, 0, 255) if emotion in NEGATIVE_EMOTIONS else (0, 255, 0) if emotion == 'Happy' else (255, 0, 0)
                            cv2.rectangle(camera_resized, (x,y), (x+w,y+h), color, 2)
                            label = f"{emotion}: {confidence*100:.1f}%"
                            cv2.putText(camera_resized, label, (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Track attention
                            attention_result = p1_attention_tracker.analyze_attention((x, y, w, h), camera_frame.shape)
                            dual_analyzer.update_person1_attention(attention_result['attention_score'])
                
                # Process Person 2 (Screen)
                if screen_frame is not None:
                    rgb2 = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2RGB)
                    faces2 = detector.detect_faces(rgb2)
                    
                    for face in faces2:
                        x, y, w, h = face['box']
                        confidence_face = face['confidence']
                        
                        if confidence_face < CONFIDENCE_THRESHOLDS['face_detection']:
                            continue
                        
                        margin = int(DETECTION_CONFIG['face_margin'] * max(w, h))
                        x1, y1 = max(0, x-margin), max(0, y-margin)
                        x2, y2 = min(rgb2.shape[1], x+w+margin), min(rgb2.shape[0], y+h+margin)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        face_roi = rgb2[y1:y2, x1:x2]
                        if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                            continue
                        
                        face_roi = preprocess_face_for_detection(face_roi)
                        face_img = cv2.resize(face_roi, (48,48))/255.0
                        face_img = face_img.reshape(1,48,48,3)
                        
                        pred = model.predict(face_img, verbose=0)[0]
                        pred = calibrate_predictions(pred)
                        
                        p2_emotion_history.append(pred)
                        if len(p2_emotion_history) > history_size:
                            p2_emotion_history.pop(0)
                        
                        smoothed_pred = np.mean(p2_emotion_history, axis=0)
                        emotion_idx = np.argmax(smoothed_pred)
                        confidence = smoothed_pred[emotion_idx]
                        
                        # CẬP NHẬT THEO PROBABILITY (nhất quán với start_detection)
                        for i in range(len(EMOTIONS)):
                            dual_analyzer.update_person2_emotion_prob(i, smoothed_pred[i])
                        
                        is_valid, _ = apply_confidence_filter(smoothed_pred, emotion_idx)
                        
                        if is_valid:
                            emotion = EMOTIONS[emotion_idx]
                            
                            # Draw on screen part (offset by camera width)
                            offset_x = camera_resized.shape[1]
                            color = (0, 0, 255) if emotion in NEGATIVE_EMOTIONS else (0, 255, 0) if emotion == 'Happy' else (255, 0, 0)
                            cv2.rectangle(screen_resized, (x,y), (x+w,y+h), color, 2)
                            label = f"{emotion}: {confidence*100:.1f}%"
                            cv2.putText(screen_resized, label, (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
                            # Track attention
                            attention_result = p2_attention_tracker.analyze_attention((x, y, w, h), screen_frame.shape)
                            dual_analyzer.update_person2_attention(attention_result['attention_score'])
                
                # Update display
                display_frame = np.hstack([camera_resized, screen_resized])
                
                # Add labels
                cv2.putText(display_frame, "BAN (Camera)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, "NGUOI DOI DIEN (Screen)", (camera_resized.shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"FPS: {current_fps:.1f} | Nhan 'q' de dung", (10, display_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show (use existing window)
                cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if camera_frame is not None and screen_frame is not None:
                    cv2.imwrite(f"results/dual_screenshot_{frame_count}.jpg", display_frame)
                    print(f"Saved dual_screenshot_{frame_count}.jpg")
        
        # Cleanup
        dual_analyzer.stop()
        screen_capturer.close()
        cv2.destroyAllWindows()
        
        # Generate comparison report
        comparison_data = dual_analyzer.get_comparison_report()
        
        if comparison_data is None:
            messagebox.showwarning("Cảnh báo", "Không đủ dữ liệu để so sánh!")
            return
        
        # Format report
        dual_report = format_dual_report(comparison_data)
        
        # Show completion
        messagebox.showinfo("Hoàn thành", 
                          "Đã hoàn thành việc quét CẢ 2 NGƯỜI!\n\n"
                          "Báo cáo so sánh sẽ hiển thị ngay.")
        
        # Show comparison in scrollable window
        from utils.visualization import show_scrollable_report
        show_scrollable_report(dual_report)
        
        # Also plot individual charts
        from utils.visualization import plot_emotion_chart
        
        # Person 1 chart
        plot_emotion_chart(
            comparison_data['person1']['emotion_counts'],
            f"BẠN (Camera)\n\nĐiểm tích cực: {comparison_data['person1']['positive_score']:.1f}%\n"
            f"Sự tập trung: {comparison_data['person1']['avg_attention']:.1f}/100"
        )
        
        # Person 2 chart
        plot_emotion_chart(
            comparison_data['person2']['emotion_counts'],
            f"NGƯỜI ĐỐI DIỆN (Screen)\n\nĐiểm tích cực: {comparison_data['person2']['positive_score']:.1f}%\n"
            f"Sự tập trung: {comparison_data['person2']['avg_attention']:.1f}/100"
        )

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error details: {error_msg}")
        print(traceback.format_exc())
        messagebox.showerror("Error", f"An error occurred: {error_msg}")



def start_detection_camera_roi(csv_path, roi, camera_id=0):
    """
    Detection function cho camera với ROI (Region of Interest)
    Chỉ quét vùng cụ thể trong khung hình
    
    Args:
        csv_path: path to dataset CSV
        roi: (x, y, width, height) vùng cần quét - MUST be tuple of 4 integers
    """
    try:
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("Lỗi", "Chọn file dataset (.csv) trước!")
            return
        
        # Validate and normalize ROI
        print(f"DEBUG: ROI type = {type(roi)}, value = {roi}")
        
        # Handle different ROI formats
        try:
            if roi is None:
                messagebox.showerror("Lỗi", "ROI là None!")
                return
            elif isinstance(roi, dict):
                # Convert dict to tuple
                print("DEBUG: Converting dict to tuple")
                roi = (int(roi['x']), int(roi['y']), int(roi['width']), int(roi['height']))
            elif isinstance(roi, (list, tuple)):
                # Convert to tuple of ints
                if len(roi) != 4:
                    messagebox.showerror("Lỗi", f"ROI phải có 4 phần tử, nhận được {len(roi)}")
                    return
                roi = tuple(int(x) for x in roi)
            else:
                messagebox.showerror("Lỗi", f"ROI format không hợp lệ: {type(roi)}")
                return
            
            print(f"DEBUG: ROI after normalization = {roi}")
            
            # Final validation
            if len(roi) != 4:
                messagebox.showerror("Lỗi", f"ROI phải có 4 phần tử: (x, y, w, h)")
                return
            
            # Extract values for clarity
            roi_x, roi_y, roi_w, roi_h = roi
            print(f"DEBUG: ROI extracted: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
            
        except (ValueError, TypeError, KeyError) as e:
            messagebox.showerror("Lỗi", f"Không thể xử lý ROI: {roi}\nError: {e}")
            return

        # Load and train model (same as before)
        dataset_name = "CK+ Extended" if "ck" in os.path.basename(csv_path).lower() else "FER2013"
        messagebox.showinfo("Thông báo", f"Tải dữ liệu từ {dataset_name} và huấn luyện model...")
        
        images, labels = load_dataset_from_fer2013(csv_path)
        valid_mask = labels < len(EMOTIONS)
        images = images[valid_mask]
        labels = labels[valid_mask]
        
        if len(images) == 0:
            messagebox.showerror("Lỗi", "Dataset không có dữ liệu hợp lệ!")
            return
        
        images = np.stack([images]*3, axis=-1)
        labels_cat = to_categorical(labels, num_classes=len(EMOTIONS))
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_cat, 
            test_size=TRAINING_CONFIG['test_size'], 
            random_state=TRAINING_CONFIG['random_state'], 
            stratify=labels
        )

        if os.path.exists(PATHS['model']):
            model = load_model()
        else:
            model = build_mobilenet_model()
            root, win, progress, epoch_label, status_label, time_label = show_loading_window(
                TRAINING_CONFIG['epochs']
            )
            callbacks = create_callbacks(
                progress, epoch_label, status_label, time_label, 
                TRAINING_CONFIG['epochs'], root
            )
            train_model(model, X_train, y_train, X_test, y_test, callbacks)
            save_model(model)
            win.destroy()
            root.update()

        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        messagebox.showinfo("Bắt đầu", 
                          f"Bắt đầu quét vùng ROI: {roi_w}x{roi_h}\n"
                          f"Vị trí: ({roi_x}, {roi_y})\n\n"
                          "- Nhấn 'q' để dừng\n"
                          "- Nhấn 's' để chụp ảnh\n"
                          "- Vùng màu xanh là vùng đang quét")
        
        # Initialize MediaPipe face detector
        from core.face_detector_mediapipe import MediaPipeFaceDetector
        from core.config import FACE_DETECTION_CONFIG
        detector = MediaPipeFaceDetector(
            min_detection_confidence=FACE_DETECTION_CONFIG['min_detection_confidence'],
            model_selection=FACE_DETECTION_CONFIG['model_selection']
        )
        
        # Tracking variables (same as normal detection)
        detection_mode = 'live'
        emotion_counts = [0] * len(EMOTIONS)
        emotion_history = []
        history_size = DETECTION_CONFIG['history_size']
        
        lighting_samples = []
        dress_color_samples = []
        background_samples = []
        environment_samples = []
        object_samples = []
        lighting_quality_samples = []
        current_face_boxes = []
        
        prev_gray = None
        movement_samples = []
        posture_samples = []
        eye_contact_samples = []
        
        # Import and initialize attention tracker
        from .attention_tracking import AttentionTracker
        attention_tracker = AttentionTracker()
        
        import time
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        frame_count = 0
        
        # Create window once (prevent multiple windows)
        window_name = "Camera ROI Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        # Main loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            # Calculate FPS
            if fps_frame_count >= 30:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Extract ROI from frame
            roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            # Create display (show full frame with ROI highlighted)
            display_frame = frame.copy()
            
            # Draw ROI rectangle on full frame
            cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 3)
            cv2.putText(display_frame, "VUNG QUET", (roi_x, roi_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            h_frame, w_frame = display_frame.shape[:2]
            
            # Analyze lighting on ROI
            if frame_count % 10 == 0:
                lighting_result = analyze_lighting(roi_frame)
                # analyze_lighting returns tuple (brightness, status, color)
                brightness, status, color = lighting_result
                lighting_samples.append(brightness)
            
            # Detect faces in ROI
            rgb_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb_roi)
            
            # Background analysis on ROI
            if frame_count % 30 == 0:
                current_face_boxes = [f['box'] for f in faces]
                bg_result = analyze_background_cleanliness(roi_frame, current_face_boxes)
                # bg_result returns dict with 'score', not 'cleanliness_score'
                background_samples.append(bg_result['score'])
                # Note: analyze_background_cleanliness doesn't return these keys
                # For now, skip these as they're from different analysis
            
            # Process faces
            for face in faces:
                x, y, w, h = face['box']
                confidence_face = face['confidence']
                
                # Adjust coordinates to full frame
                x_full = roi_x + x
                y_full = roi_y + y
                
                if confidence_face < CONFIDENCE_THRESHOLDS['face_detection']:
                    cv2.rectangle(display_frame, (x_full, y_full), (x_full+w, y_full+h), (0,255,255), 1)
                    cv2.putText(display_frame, f"Low conf: {confidence_face:.2f}", (x_full, y_full-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
                    continue
                
                # Expand face ROI
                margin = int(DETECTION_CONFIG['face_margin'] * max(w, h))
                x1, y1 = max(0, x-margin), max(0, y-margin)
                x2, y2 = min(rgb_roi.shape[1], x+w+margin), min(rgb_roi.shape[0], y+h+margin)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Analyze dress - IMPROVED with pose
                if frame_count % 30 == 0:
                    dress_result = analyze_dress_improved((x, y, w, h), roi_frame, use_pose=True)
                    if dress_result and dress_result.get('combined_score', 0) > 0:
                        dress_color_samples.append(dress_result)
                
                # Preprocess face
                face_roi = rgb_roi[y1:y2, x1:x2]
                
                if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                    continue
                
                face_roi = preprocess_face_for_detection(face_roi)
                face_img = cv2.resize(face_roi, (48,48))/255.0
                face_img = face_img.reshape(1,48,48,3)
                
                # Predict
                pred = model.predict(face_img, verbose=0)[0]
                pred = calibrate_predictions(pred)
                
                emotion_history.append(pred)
                if len(emotion_history) > history_size:
                    emotion_history.pop(0)
                
                smoothed_pred = np.mean(emotion_history, axis=0)
                emotion_idx = np.argmax(smoothed_pred)
                confidence = smoothed_pred[emotion_idx]
                
                is_valid, _ = apply_confidence_filter(smoothed_pred, emotion_idx)
                
                # Cộng probability cho tất cả emotions (nhất quán với start_detection)
                for i in range(len(EMOTIONS)):
                    emotion_counts[i] += smoothed_pred[i]
                
                # Lấy emotion dominant để hiển thị
                emotion = EMOTIONS[emotion_idx]
                
                if is_valid:
                    color = (0, 0, 255) if emotion in NEGATIVE_EMOTIONS else (0, 255, 0) if emotion == 'Happy' else (255, 0, 0)
                    
                    # Draw on full frame
                    cv2.rectangle(display_frame, (x_full, y_full), (x_full+w, y_full+h), color, 2)
                    label = f"{emotion}: {confidence*100:.1f}%"
                    cv2.putText(display_frame, label, (x_full, y_full-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Track behavior (use ROI coordinates)
                    from .behavior_analysis import analyze_posture_simple, estimate_eye_contact
                    posture_result = analyze_posture_simple((x, y, w, h), roi_frame.shape)
                    posture_samples.append(posture_result)
                    
                    eye_contact_result = estimate_eye_contact((x, y, w, h), roi_frame.shape)
                    eye_contact_samples.append(eye_contact_result)
                    
                    # Track attention
                    attention_result = attention_tracker.analyze_attention((x, y, w, h), roi_frame.shape)
                    attention_text = f"Tap trung: {attention_result['attention_score']:.0f}%"
                    cv2.putText(display_frame, attention_text, (10, h_frame - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Track gestures
            curr_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None and len(faces) > 0:
                from .behavior_analysis import detect_hand_movement
                face_box = faces[0]['box'] if faces else None
                movement_result = detect_hand_movement(prev_gray, curr_gray, face_box)
                movement_samples.append(movement_result)
            prev_gray = curr_gray.copy()
            
            # Add instructions
            cv2.putText(display_frame, "QUET VUNG ROI - Nhan 'q' de dung", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, h_frame - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show (use existing window)
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"results/roi_screenshot_{frame_count}.jpg", display_frame)
                print(f"Saved roi_screenshot_{frame_count}.jpg")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate report (same as normal detection)
        if sum(emotion_counts) == 0:
            messagebox.showwarning("Cảnh báo", "Không phát hiện được cảm xúc nào!")
            return
        
        # Calculate scores (same as before)
        from .dress_analysis import generate_dress_report
        dress_report = generate_dress_report(dress_color_samples)
        dress_score = dress_report['score']
        
        from .background_analysis import generate_background_report
        bg_report = generate_background_report(
            background_samples, environment_samples, 
            object_samples, lighting_quality_samples
        )
        base_score = bg_report['cleanliness_score']
        obj_summary = bg_report['object_summary']
        object_adjustment = obj_summary['avg_penalty'] + obj_summary['avg_bonus']
        background_score = max(0, min(100, base_score + object_adjustment))
        
        scorer = AffectivaScorer()
        emotion_score = scorer.calculate_emotion_score(emotion_counts)
        
        lighting_score = calculate_lighting_score(lighting_samples)
        appearance_score = scorer.calculate_appearance_score(
            dress_score=dress_score,
            background_score=background_score,
            lighting_score=lighting_score
        )
        
        from .behavior_analysis import (
            get_posture_summary,
            calculate_eye_contact_percentage,
            analyze_gesture_frequency,
            detect_fidgeting
        )
        
        posture_summary = get_posture_summary(posture_samples)
        eye_contact_summary = calculate_eye_contact_percentage(eye_contact_samples)
        gesture_summary = analyze_gesture_frequency(movement_samples)
        fidgeting_result = detect_fidgeting(movement_samples) if len(movement_samples) >= 30 else {
            'is_fidgeting': False,
            'fidgeting_score': 90.0,
            'confidence_level': 'Unknown'
        }
        
        behavior_score = scorer.calculate_behavior_score(
            eye_contact_score=eye_contact_summary['score'],
            posture_score=posture_summary['avg_score'],
            gesture_score=gesture_summary['gesture_score']
        )
        
        technical_score = scorer.calculate_technical_score()
        affectiva_report = scorer.generate_simple_report()
        
        final_emo = EMOTIONS[np.argmax(emotion_counts)]
        from utils.interview_suggestions import get_suggestions_by_mode
        mode_suggestions = get_suggestions_by_mode(
            'live',
            emotion_counts, 
            lighting_samples, 
            dress_color_samples
        )
        
        emotion_suggestions = get_detailed_suggestions(emotion_counts)
        lighting_summary = get_lighting_summary(lighting_samples)
        dress_summary = get_dress_summary(dress_color_samples)
        
        from .attention_tracking import format_attention_report
        attention_summary = attention_tracker.get_attention_summary()
        attention_report = f"\n\n{'='*60}\n\n{format_attention_report(attention_summary)}"
        
        mode_title = "PHỎNG VẤN ONLINE (VÙNG ROI)"
        full_report = f"{affectiva_report}\n\n{'='*60}\n\n{mode_title}:\n{mode_suggestions}{attention_report}"
        
        save_results(emotion_counts, final_emo)
        
        import json
        json_report = scorer.get_json_report()
        with open('results/affectiva_report.json', 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False)
        
        messagebox.showinfo("Hoàn thành", 
                          "Đã hoàn thành việc quét vùng ROI!\n\n"
                          "Biểu đồ và lời khuyên sẽ hiển thị ngay.")
        
        plot_emotion_chart(emotion_counts, full_report)

    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"Error details: {error_msg}")
        print(traceback.format_exc())
        messagebox.showerror("Error", f"An error occurred: {error_msg}")
