# -*- coding: utf-8 -*-
"""
Probability Calibration
Xử lý calibration và confidence filtering cho emotion predictions
"""
import numpy as np
from .config import CALIBRATION_WEIGHTS, CONFIDENCE_THRESHOLDS, SENSITIVITY_LEVELS, EMOTIONS


def calibrate_predictions(predictions, calibration_weights=None):
    """
    Calibrate emotion predictions to reduce bias
    
    Args:
        predictions: raw model output probabilities (numpy array)
        calibration_weights: optional weights for each emotion (numpy array)
    
    Returns:
        calibrated probabilities (numpy array)
    """
    if calibration_weights is None:
        calibration_weights = CALIBRATION_WEIGHTS
    
    # Apply calibration
    calibrated = predictions * calibration_weights
    
    # Re-normalize to sum to 1
    calibrated = calibrated / np.sum(calibrated)
    
    return calibrated


def apply_confidence_filter(predictions, emotion_idx, sensitivity='medium'):
    """
    Filter predictions based on confidence thresholds
    
    Args:
        predictions: numpy array of emotion probabilities
        emotion_idx: index of predicted emotion
        sensitivity: 'low', 'medium', or 'high'
    
    Returns:
        (is_valid, confidence_score)
    """
    base_threshold = SENSITIVITY_LEVELS[sensitivity]
    emotion_name = EMOTIONS[emotion_idx]
    
    # Apply emotion-specific threshold
    if emotion_name == 'Happy':
        threshold = max(base_threshold, CONFIDENCE_THRESHOLDS['Happy'])
    elif emotion_name == 'Angry':
        threshold = max(base_threshold, CONFIDENCE_THRESHOLDS['Angry'])
    elif emotion_name == 'Sad':
        threshold = max(base_threshold, CONFIDENCE_THRESHOLDS['Sad'])
    else:
        threshold = base_threshold
    
    confidence = predictions[emotion_idx]
    return (confidence >= threshold, confidence)
