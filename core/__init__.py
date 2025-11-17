# -*- coding: utf-8 -*-
"""
Core Package
Emotion detection core functionality
"""
from .detector import start_detection
from .config import EMOTIONS, NEGATIVE_EMOTIONS
from .calibration import calibrate_predictions, apply_confidence_filter
from .lighting import analyze_lighting, get_lighting_summary
from .preprocessing import preprocess_image, preprocess_face_for_detection, load_dataset_from_fer2013
from .model import build_mobilenet_model, train_model, save_model, load_model

__all__ = [
    'start_detection',
    'EMOTIONS',
    'NEGATIVE_EMOTIONS',
    'calibrate_predictions',
    'apply_confidence_filter',
    'analyze_lighting',
    'get_lighting_summary',
    'preprocess_image',
    'preprocess_face_for_detection',
    'load_dataset_from_fer2013',
    'build_mobilenet_model',
    'train_model',
    'save_model',
    'load_model'
]
