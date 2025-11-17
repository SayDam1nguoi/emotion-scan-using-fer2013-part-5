# -*- coding: utf-8 -*-
"""
Behavior Analysis
Phân tích hành vi: posture, eye contact, gestures, fidgeting
"""
import cv2
import numpy as np


# ===== Task 3.1: Posture Analysis =====

def analyze_posture_simple(face_box, frame_shape):
    """
    Phân tích tư thế đơn giản dựa trên vị trí khuôn mặt
    (Simplified version - không dùng MediaPipe Pose)
    
    Args:
        face_box: (x, y, w, h) face bounding box
        frame_shape: (height, width) của frame
    
    Returns:
        dict with posture analysis
    """
    if not face_box:
        return {
            'posture_type': 'Unknown',
            'posture_score': 70.0,
            'spine_angle': 0.0,
            'is_centered': False
        }
    
    x, y, w, h = face_box
    frame_h, frame_w = frame_shape[:2]
    
    # Calculate face center
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Frame center
    frame_center_x = frame_w // 2
    frame_center_y = frame_h // 2
    
    # Check if face is centered horizontally
    horizontal_offset = abs(face_center_x - frame_center_x) / frame_w
    is_centered = horizontal_offset < 0.15  # Within 15% of center
    
    # Check vertical position (should be in upper-middle area)
    vertical_position = face_center_y / frame_h
    
    # Estimate posture based on position
    if vertical_position < 0.3:
        # Too high - might be standing or leaning forward
        posture_type = "Leaning Forward"
        posture_score = 75.0
    elif vertical_position > 0.6:
        # Too low - slouching
        posture_type = "Slouching"
        posture_score = 50.0
    elif is_centered and 0.3 <= vertical_position <= 0.5:
        # Good position
        posture_type = "Good Posture"
        posture_score = 95.0
    else:
        # Tilted or off-center
        posture_type = "Tilted"
        posture_score = 65.0
    
    # Estimate spine angle (simplified)
    spine_angle = horizontal_offset * 30  # Max 30 degrees tilt
    
    return {
        'posture_type': posture_type,
        'posture_score': posture_score,
        'spine_angle': spine_angle,
        'is_centered': is_centered,
        'vertical_position': vertical_position
    }


def get_posture_summary(posture_samples):
    """
    Tổng hợp posture từ nhiều samples
    
    Args:
        posture_samples: list of posture analysis results
    
    Returns:
        summary dict
    """
    if not posture_samples:
        return {
            'avg_score': 70.0,
            'dominant_posture': 'Unknown',
            'recommendation': 'Khong co du lieu'
        }
    
    # Calculate average score
    scores = [s['posture_score'] for s in posture_samples]
    avg_score = np.mean(scores)
    
    # Get most common posture
    from collections import Counter
    postures = [s['posture_type'] for s in posture_samples]
    dominant_posture = Counter(postures).most_common(1)[0][0]
    
    # Generate recommendation
    if dominant_posture == "Good Posture":
        recommendation = "Tư thế tốt! Tiếp tục duy trì."
    elif dominant_posture == "Slouching":
        recommendation = "Đang gù lưng. Ngồi thẳng lưng và nhìn thẳng vào camera."
    elif dominant_posture == "Leaning Forward":
        recommendation = "Đang cúi người về phía trước. Ngồi lùi một chút."
    elif dominant_posture == "Tilted":
        recommendation = "Đang nghiêng. Cần chỉnh lại vị trí cho thẳng."
    else:
        recommendation = "Kiểm tra lại tư thế ngồi."
    
    return {
        'avg_score': avg_score,
        'dominant_posture': dominant_posture,
        'recommendation': recommendation
    }


# ===== Task 3.2: Eye Contact Tracking =====

def estimate_eye_contact(face_box, frame_shape, face_landmarks=None):
    """
    Ước tính eye contact dựa trên vị trí khuôn mặt
    (Simplified - không dùng MediaPipe Face Mesh)
    
    Args:
        face_box: (x, y, w, h)
        frame_shape: (height, width)
        face_landmarks: optional landmarks (not used in simple version)
    
    Returns:
        dict with eye contact estimation
    """
    if not face_box:
        return {
            'looking_at_camera': False,
            'confidence': 0.0,
            'gaze_direction': 'Unknown'
        }
    
    x, y, w, h = face_box
    frame_h, frame_w = frame_shape[:2]
    
    # Face center
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # Frame center
    frame_center_x = frame_w // 2
    frame_center_y = frame_h // 2
    
    # Calculate offset
    offset_x = abs(face_center_x - frame_center_x) / frame_w
    offset_y = abs(face_center_y - frame_center_y) / frame_h
    
    # If face is centered, assume looking at camera
    looking_at_camera = offset_x < 0.2 and offset_y < 0.2
    
    # Confidence based on how centered the face is
    confidence = 1.0 - (offset_x + offset_y) / 2
    confidence = max(0.0, min(1.0, confidence))
    
    # Estimate gaze direction
    if looking_at_camera:
        gaze_direction = "At Camera"
    elif offset_x > 0.3:
        gaze_direction = "To Side"
    elif offset_y > 0.3:
        gaze_direction = "Up/Down"
    else:
        gaze_direction = "Slightly Off"
    
    return {
        'looking_at_camera': looking_at_camera,
        'confidence': confidence,
        'gaze_direction': gaze_direction
    }


def calculate_eye_contact_percentage(eye_contact_samples):
    """
    Tính % thời gian nhìn camera
    
    Args:
        eye_contact_samples: list of eye contact results
    
    Returns:
        dict with percentage and score
    """
    if not eye_contact_samples:
        return {
            'percentage': 0.0,
            'score': 0.0,
            'recommendation': 'Khong co du lieu'
        }
    
    # Count frames looking at camera
    looking_count = sum(1 for s in eye_contact_samples if s['looking_at_camera'])
    total = len(eye_contact_samples)
    
    percentage = (looking_count / total) * 100 if total > 0 else 0.0
    
    # Score based on percentage
    if percentage >= 70:
        score = 100.0
        recommendation = "Eye contact tốt! (>70%)"
    elif percentage >= 50:
        score = 80.0
        recommendation = "Eye contact khá ổn. Nên tăng lên >70%."
    elif percentage >= 30:
        score = 60.0
        recommendation = "Eye contact thấp. Nên nhìn camera nhiều hơn."
    else:
        score = 40.0
        recommendation = "Eye contact rất thấp! Phải nhìn camera nhiều hơn."
    
    return {
        'percentage': percentage,
        'score': score,
        'recommendation': recommendation
    }


# ===== Task 3.3: Gesture Analysis =====

def detect_hand_movement(prev_frame, curr_frame, face_box):
    """
    Detect hand movement đơn giản bằng motion detection
    (Simplified - không dùng MediaPipe Hands)
    
    Args:
        prev_frame: previous frame (grayscale)
        curr_frame: current frame (grayscale)
        face_box: (x, y, w, h) to exclude face region
    
    Returns:
        dict with movement detection
    """
    if prev_frame is None or curr_frame is None:
        return {
            'has_movement': False,
            'movement_intensity': 0.0
        }
    
    # Calculate frame difference
    diff = cv2.absdiff(prev_frame, curr_frame)
    
    # Threshold
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Exclude face region (to focus on hands/body)
    if face_box:
        x, y, w, h = face_box
        # Expand face region
        margin = int(max(w, h) * 0.3)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(thresh.shape[1], x + w + margin)
        y2 = min(thresh.shape[0], y + h + margin)
        thresh[y1:y2, x1:x2] = 0
    
    # Calculate movement intensity
    movement_pixels = np.sum(thresh > 0)
    total_pixels = thresh.size
    movement_intensity = movement_pixels / total_pixels
    
    has_movement = movement_intensity > 0.01  # 1% threshold
    
    return {
        'has_movement': has_movement,
        'movement_intensity': movement_intensity
    }


def analyze_gesture_frequency(movement_samples):
    """
    Phân tích tần suất cử chỉ
    
    Args:
        movement_samples: list of movement detection results
    
    Returns:
        dict with gesture analysis
    """
    if not movement_samples:
        return {
            'gesture_frequency': 'Unknown',
            'gesture_score': 70.0,
            'movement_count': 0,
            'recommendation': 'Khong co du lieu'
        }
    
    # Count movements
    movement_count = sum(1 for s in movement_samples if s['has_movement'])
    total = len(movement_samples)
    movement_rate = movement_count / total if total > 0 else 0.0
    
    # Classify frequency
    if movement_rate > 0.5:
        gesture_frequency = "Too Much"
        gesture_score = 60.0
        recommendation = "Cử chỉ quá nhiều. Nên giữ bình tĩnh hơn."
    elif movement_rate > 0.2:
        gesture_frequency = "Moderate"
        gesture_score = 90.0
        recommendation = "Cử chỉ vừa phải. Tốt!"
    elif movement_rate > 0.05:
        gesture_frequency = "Minimal"
        gesture_score = 75.0
        recommendation = "Cử chỉ ít. Có thể thêm một chút để tự nhiên hơn."
    else:
        gesture_frequency = "Very Little"
        gesture_score = 70.0
        recommendation = "Rất ít cử chỉ. Nên tự nhiên hơn."
    
    return {
        'gesture_frequency': gesture_frequency,
        'gesture_score': gesture_score,
        'movement_count': movement_count,
        'movement_rate': movement_rate,
        'recommendation': recommendation
    }


# ===== Task 3.4: Fidgeting Detection =====

def detect_fidgeting(movement_samples, window_size=30):
    """
    Phát hiện fidgeting (cử động bồn chồn)
    
    Args:
        movement_samples: list of movement detection results
        window_size: số frames để phân tích
    
    Returns:
        dict with fidgeting analysis
    """
    if len(movement_samples) < window_size:
        return {
            'is_fidgeting': False,
            'fidgeting_score': 0.0,
            'confidence_level': 'Unknown'
        }
    
    # Analyze recent movements
    recent = movement_samples[-window_size:]
    
    # Count frequent small movements (fidgeting pattern)
    movement_count = sum(1 for s in recent if s['has_movement'])
    movement_rate = movement_count / window_size
    
    # Calculate average intensity
    intensities = [s['movement_intensity'] for s in recent]
    avg_intensity = np.mean(intensities)
    
    # Fidgeting = frequent but small movements
    is_fidgeting = movement_rate > 0.4 and avg_intensity < 0.05
    
    # Calculate fidgeting score (lower = more fidgeting)
    if is_fidgeting:
        fidgeting_score = max(0, 100 - movement_rate * 100)
        confidence_level = "Nervous"
    elif movement_rate > 0.3:
        fidgeting_score = 70.0
        confidence_level = "Slightly Nervous"
    else:
        fidgeting_score = 90.0
        confidence_level = "Confident"
    
    return {
        'is_fidgeting': is_fidgeting,
        'fidgeting_score': fidgeting_score,
        'confidence_level': confidence_level,
        'movement_rate': movement_rate
    }


def get_behavior_summary(posture_summary, eye_contact_summary, gesture_summary, fidgeting_result):
    """
    Tổng hợp behavior analysis
    
    Args:
        posture_summary: posture analysis summary
        eye_contact_summary: eye contact summary
        gesture_summary: gesture analysis summary
        fidgeting_result: fidgeting detection result
    
    Returns:
        comprehensive behavior summary
    """
    # Calculate overall behavior score
    behavior_score = (
        posture_summary['avg_score'] * 0.35 +
        eye_contact_summary['score'] * 0.35 +
        gesture_summary['gesture_score'] * 0.15 +
        fidgeting_result['fidgeting_score'] * 0.15
    )
    
    # Generate recommendations
    recommendations = []
    
    if posture_summary['avg_score'] < 70:
        recommendations.append(posture_summary['recommendation'])
    
    if eye_contact_summary['score'] < 70:
        recommendations.append(eye_contact_summary['recommendation'])
    
    if gesture_summary['gesture_score'] < 70:
        recommendations.append(gesture_summary['recommendation'])
    
    if fidgeting_result['is_fidgeting']:
        recommendations.append("Phát hiện fidgeting. Nên bình tĩnh và giữ yên.")
    
    if not recommendations:
        recommendations.append("Hành vi tốt! Tiếp tục duy trì.")
    
    return {
        'behavior_score': behavior_score,
        'posture': posture_summary,
        'eye_contact': eye_contact_summary,
        'gestures': gesture_summary,
        'fidgeting': fidgeting_result,
        'recommendations': recommendations
    }
