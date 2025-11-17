# -*- coding: utf-8 -*-
"""
Image Preprocessing
Xử lý tiền xử lý ảnh và dataset
"""
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import resample
from .config import DETECTION_CONFIG, EMOTION_MAPPING


def preprocess_image(img):
    """
    Preprocess single image
    
    Args:
        img: grayscale image (48x48)
    
    Returns:
        preprocessed image
    """
    img = cv2.equalizeHist((img * 255).astype(np.uint8))
    return img / 255.0


def preprocess_face_for_detection(face_rgb):
    """
    Tiền xử lý khuôn mặt để tăng độ chính xác
    Enhanced cho low light conditions
    
    Args:
        face_rgb: RGB face image
    
    Returns:
        preprocessed face image
    """
    # Check brightness
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    avg_brightness = np.mean(gray)
    
    # Nếu ánh sáng thấp, enhance mạnh hơn
    if avg_brightness < 80:
        # CLAHE cho low light
        lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        face_rgb = cv2.merge([l, a, b])
        face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_LAB2RGB)
        
        # Brightness boost
        face_rgb = cv2.convertScaleAbs(face_rgb, alpha=1.3, beta=15)
    else:
        # Cân bằng histogram bình thường
        face_yuv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YUV)
        face_yuv[:,:,0] = cv2.equalizeHist(face_yuv[:,:,0])
        face_rgb = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2RGB)
    
    # Giảm nhiễu nhẹ
    config = DETECTION_CONFIG['bilateral_filter']
    face_rgb = cv2.bilateralFilter(face_rgb, config['d'], 
                                   config['sigmaColor'], config['sigmaSpace'])
    
    return face_rgb


def load_dataset_from_fer2013(csv_path, usage="Training"):
    """
    Load dataset from FER2013 or CK+ Extended format
    Supports both formats automatically
    
    Args:
        csv_path: path to CSV file
        usage: "Training", "PublicTest", or "PrivateTest"
    
    Returns:
        (images, labels) as numpy arrays
    """
    data = pd.read_csv(csv_path)
    
    # Filter by usage if column exists
    if "Usage" in data.columns:
        data = data[data["Usage"] == usage]

    images, labels = [], []
    for idx, row in data.iterrows():
        try:
            # Handle both string and array formats
            if isinstance(row['pixels'], str):
                img = np.fromstring(row['pixels'], dtype=int, sep=' ').reshape(48, 48)
            else:
                img = np.array(row['pixels']).reshape(48, 48)
            
            images.append(preprocess_image(img))
            labels.append(row['emotion'])
        except Exception as e:
            continue

    images = np.array(images)
    labels = np.array(labels)
    
    # Map từ 7 emotions sang 4 emotions
    mapped_labels = np.array([EMOTION_MAPPING[label] for label in labels])

    # Shuffle để tránh bias
    shuffle_idx = np.random.permutation(len(images))
    X = images[shuffle_idx]
    y = mapped_labels[shuffle_idx]
    
    return X, y
