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
    'Happy': 0.35,    # GIẢM để dễ detect Happy hơn
    'Angry': 0.40,    # TĂNG để khó detect Angry hơn
    'Sad': 0.50,      # TĂNG để khó detect Sad hơn
    'Neutral': 0.28,  # GIẢM để dễ detect Neutral hơn
    'face_detection': 0.80  # Giam xuong de detect face de hon trong low light
}

# ===== Sensitivity levels =====
SENSITIVITY_LEVELS = {
    'low': 0.45,      # Strict, fewer false positives
    'medium': 0.30,   # Balanced (default) - very low for detection
    'high': 0.20      # Permissive, maximum detections
}

# ===== Calibration weights =====
CALIBRATION_WEIGHTS = np.array([
    0.85,  # Angry - giảm xuống để tăng Happy/Neutral
    1.25,  # Happy - TĂNG MẠNH (target 40-45%)
    0.70,  # Sad - giảm xuống để tăng Happy/Neutral
    1.30   # Neutral - TĂNG MẠNH (target 45-50%)
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

# ===== Detection configuration =====
DETECTION_CONFIG = {
    'history_size': 5,  # Temporal smoothing window
    'face_margin': 0.15,  # Face ROI margin
    'detect_every_n_frames': 2,  # Detect thường xuyên hơn (mỗi 2 frames)
    'detection_scale': 0.75,  # Tăng resolution để detect tốt hơn
    'lighting_check_interval': 5,  # Kiểm tra ánh sáng mỗi N frames
    'enhance_low_light': True,  # Bật enhancement cho low light
    'bilateral_filter': {
        'd': 5,
        'sigmaColor': 50,
        'sigmaSpace': 50
    }
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
