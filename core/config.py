# -*- coding: utf-8 -*-
"""
Configuration and Constants
Chứa tất cả constants và configurations cho emotion detection
"""
import numpy as np

# ===== Danh sách cảm xúc =====
# Chỉ sử dụng 4 cảm xúc chính
EMOTIONS = ['Angry', 'Happy', 'Sad', 'Neutral']
NEGATIVE_EMOTIONS = ['Angry', 'Sad']

# Mapping từ FER2013 (7 emotions) sang 4 emotions
# FER2013: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
EMOTION_MAPPING = {
    0: 0,  # Angry -> Angry
    1: 0,  # Disgust -> Angry (gộp vào Angry)
    2: 2,  # Fear -> Sad (gộp vào Sad)
    3: 1,  # Happy -> Happy
    4: 2,  # Sad -> Sad
    5: 3,  # Surprise -> Neutral (CHANGED: hop ly hon cho interview)
    6: 3   # Neutral -> Neutral
}

# ===== Confidence thresholds =====
CONFIDENCE_THRESHOLDS = {
    'default': 0.30,  # Balanced default
    'Happy': 0.46,    # TĂNG NHẸ để giảm Happy xuống 22-26% (0.45 → 0.46)
    'Angry': 0.45,    # giữ nguyên
    'Sad': 0.55,      # giữ nguyên
    'Neutral': 0.15,  # giữ nguyên - dễ detect
    'face_detection': 0.80  # Default threshold
}

# ===== Sensitivity levels =====
SENSITIVITY_LEVELS = {
    'low': 0.45,      # Strict, fewer false positives
    'medium': 0.30,   # Balanced (default)
    'high': 0.20      # Permissive, maximum detections
}

# ===== Calibration weights =====
CALIBRATION_WEIGHTS = np.array([
    0.75,  # Angry - giữ nguyên
    0.90,  # Happy - GIẢM NHẸ để xuống 22-26% (0.92 → 0.90)
    0.60,  # Sad - giữ nguyên
    1.76   # Neutral - TĂNG NHẸ để cân bằng (1.75 → 1.76)
])

# ===== Training configuration =====
TRAINING_CONFIG = {
    'epochs': 35,
    'batch_size': 64,
    'test_size': 0.2,
    'random_state': 42,
    'early_stop_patience': 5,
    'reduce_lr_factor': 0.5,
    'reduce_lr_patience': 3,
    'min_lr': 0.00001
}

# ===== Data augmentation configuration =====
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'brightness_range': [0.85, 1.15],
    'fill_mode': 'nearest'
}

# ===== Model configuration =====
MODEL_CONFIG = {
    'input_shape': (48, 48, 3),
    'base_model': 'MobileNetV2',
    'weights': 'imagenet',
    'dropout_rates': [0.5, 0.4, 0.3],
    'dense_units': [256, 128],
    'l2_regularization': 0.001,
    'learning_rate': 0.0001,
    'unfreeze_layers': -40
}

# ===== Face Detection configuration =====
FACE_DETECTION_CONFIG = {
    'detector_type': 'mediapipe',  # 'mtcnn' or 'mediapipe'
    'min_detection_confidence': 0.75,  # TĂNG CAO để loại bỏ false positives (0.5 → 0.75)
    'model_selection': 0,  # MediaPipe: 0=short-range (2m), 1=full-range (5m)
    'min_tracking_confidence': 0.7,  # TĂNG để chỉ track faces ổn định
}

# ===== Detection configuration - CAMERA (Live) =====
DETECTION_CONFIG = {
    'history_size': 10,  # TĂNG MẠNH temporal smoothing (7 → 10) để ổn định hơn
    'face_margin': 0.15,  # Face ROI margin
    'detect_every_n_frames': 2,  # Detect mỗi 2 frames
    'detection_scale': 1.0,  # MediaPipe nhanh, không cần scale down
    'lighting_check_interval': 5,  # Kiểm tra ánh sáng mỗi N frames
    'enhance_low_light': True,  # Bật enhancement cho low light
    'min_face_size': 100,  # TĂNG để loại bỏ faces nhỏ (false positives) (60 → 100)
    'max_faces': 1,  # CHỈ LẤY 1 FACE CHÍNH (loại bỏ false positives)
    'video_playback_speed': 1.0,  # TỐC ĐỘ BÌNH THƯỜNG (1.0 = 100%)
    'face_aspect_ratio_range': (0.7, 1.5),  # Tỷ lệ width/height hợp lý cho face
    'min_face_area_ratio': 0.02,  # Face phải chiếm ít nhất 2% frame
    'bilateral_filter': {
        'd': 5,
        'sigmaColor': 50,
        'sigmaSpace': 50
    }
}

# ===== Detection configuration - VIDEO (Recorded) =====
# NGHIÊM NGẶT để loại bỏ false positives, nhưng detect thường xuyên hơn
VIDEO_DETECTION_CONFIG = {
    'min_detection_confidence': 0.75,  # TĂNG LẠI - giống camera (0.6 → 0.75)
    'min_tracking_confidence': 0.7,   # TĂNG LẠI để chỉ track faces ổn định
    'history_size': 8,                # Vừa phải (7 → 8) - cân bằng giữa rõ và ổn định
    'detect_every_n_frames': 1,       # DETECT MỖI FRAME - quan trọng cho video
    'min_face_size': 100,             # TĂNG LẠI - chỉ faces lớn (80 → 100)
    'max_faces': 1,                   # Chỉ 1 face chính
    'face_threshold': 0.75,           # TĂNG CAO - loại bỏ false positives (0.6 → 0.75)
    'min_consecutive_frames': 4,      # TĂNG - phải xuất hiện 4 frames (2 → 4)
    'skip_quality_checks': False,     # KHÔNG BỎ QUA quality checks
    'edge_margin': 0.15,              # TĂNG - xa rìa hơn (0.05 → 0.15)
    'min_face_area_ratio': 0.03,     # Face phải chiếm ít nhất 3% frame (tăng từ 2%)
    'max_face_area_ratio': 0.7,      # Face không quá 70% frame
    # LOẠI BỎ FALSE SCAN "NHẤP NHÁY"
    'min_stable_duration': 1.5,       # Face phải xuất hiện ít nhất 1.5 giây (MỚI)
    'ignore_short_detections': True,  # Bỏ qua detections < min_stable_duration (MỚI)
}

# ===== Lighting thresholds =====
LIGHTING_THRESHOLDS = {
    'too_dark': 60,
    'dark': 80,
    'bright': 180,
    'too_bright': 200
}

# ===== File paths =====
PATHS = {
    'model': 'emotion_model_mobilenet.h5',
    'results_dir': 'results',
    'results_csv': 'results/emotion_results.csv',
    'chart_output': 'results/emotion_chart.png'
}
