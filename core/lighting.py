# -*- coding: utf-8 -*-
"""
Lighting Analysis
Phân tích chất lượng ánh sáng trong video/camera
"""
import cv2
import numpy as np
from .config import LIGHTING_THRESHOLDS


def analyze_lighting(frame):
    """
    Phân tích chất lượng ánh sáng trong frame
    
    Args:
        frame: BGR image frame
    
    Returns:
        (brightness_level, status, color)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    
    # Determine lighting status
    if avg_brightness < LIGHTING_THRESHOLDS['too_dark']:
        return avg_brightness, "Qua toi", (0, 0, 255)  # Red
    elif avg_brightness < LIGHTING_THRESHOLDS['dark']:
        return avg_brightness, "Hoi toi", (0, 165, 255)  # Orange
    elif avg_brightness > LIGHTING_THRESHOLDS['too_bright']:
        return avg_brightness, "Qua sang", (0, 0, 255)  # Red
    elif avg_brightness > LIGHTING_THRESHOLDS['bright']:
        return avg_brightness, "Hoi sang", (0, 165, 255)  # Orange
    else:
        return avg_brightness, "Tot", (0, 255, 0)  # Green


def get_lighting_summary(lighting_samples):
    """
    Tổng kết chất lượng ánh sáng
    
    Args:
        lighting_samples: list of brightness values
    
    Returns:
        summary string
    """
    if not lighting_samples:
        return "Khong co du lieu anh sang"
    
    avg_brightness = np.mean(lighting_samples)
    
    # Count issues
    too_dark = sum(1 for b in lighting_samples if b < LIGHTING_THRESHOLDS['too_dark'])
    dark = sum(1 for b in lighting_samples 
               if LIGHTING_THRESHOLDS['too_dark'] <= b < LIGHTING_THRESHOLDS['dark'])
    too_bright = sum(1 for b in lighting_samples if b > LIGHTING_THRESHOLDS['too_bright'])
    bright = sum(1 for b in lighting_samples 
                 if LIGHTING_THRESHOLDS['bright'] < b <= LIGHTING_THRESHOLDS['too_bright'])
    good = sum(1 for b in lighting_samples 
               if LIGHTING_THRESHOLDS['dark'] <= b <= LIGHTING_THRESHOLDS['bright'])
    
    total = len(lighting_samples)
    
    # Summary
    summary = f"PHAN TICH ANH SANG:\n\n"
    summary += f"Do sang trung binh: {avg_brightness:.1f}/255\n\n"
    
    if avg_brightness < 70:
        summary += "ANH SANG QUA TOI\n"
        summary += f"- {too_dark + dark} frames ({(too_dark + dark)/total*100:.1f}%) thieu sang\n"
        summary += "Khuyen nghi:\n"
        summary += "  - Bat them den\n"
        summary += "  - Di chuyen gan cua so\n"
        summary += "  - Su dung den ban\n"
    elif avg_brightness > 190:
        summary += "ANH SANG QUA SANG\n"
        summary += f"- {too_bright + bright} frames ({(too_bright + bright)/total*100:.1f}%) qua sang\n"
        summary += "Khuyen nghi:\n"
        summary += "  - Tranh anh sang truc tiep\n"
        summary += "  - Dong rem cua\n"
        summary += "  - Thay doi goc camera\n"
    else:
        summary += "ANH SANG TOT\n"
        summary += f"- {good} frames ({good/total*100:.1f}%) anh sang tot\n"
        if too_dark + dark > 0:
            summary += f"- {too_dark + dark} frames ({(too_dark + dark)/total*100:.1f}%) hoi toi\n"
        if too_bright + bright > 0:
            summary += f"- {too_bright + bright} frames ({(too_bright + bright)/total*100:.1f}%) hoi sang\n"
    
    return summary
