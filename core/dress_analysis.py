# -*- coding: utf-8 -*-
"""
Dress Code Analysis
Phan tich trang phuc cua ung vien
"""
import cv2
import numpy as np


def detect_upper_body_region(face_box, frame_shape):
    """
    Detect vung upper body tu face bounding box
    
    Args:
        face_box: (x, y, w, h) cua face
        frame_shape: (height, width, channels) cua frame
    
    Returns:
        (x1, y1, x2, y2) cua upper body region
    """
    x, y, w, h = face_box
    frame_h, frame_w = frame_shape[:2]
    
    # Upper body thuong nam duoi face
    # Estimate: width = 2.5 * face_width, height = 3 * face_height
    body_w = int(w * 2.5)
    body_h = int(h * 3)
    
    # Center horizontally around face
    body_x1 = max(0, x - int((body_w - w) / 2))
    body_x2 = min(frame_w, body_x1 + body_w)
    
    # Start from below face
    body_y1 = y + h
    body_y2 = min(frame_h, body_y1 + body_h)
    
    return (body_x1, body_y1, body_x2, body_y2)


def analyze_dominant_color(image_region):
    """
    Phan tich mau sac chu dao trong vung anh (HSV color space)
    
    Args:
        image_region: BGR image region
    
    Returns:
        (dominant_color_name, hsv_values, confidence)
    """
    if image_region.size == 0:
        return ("Unknown", (0, 0, 0), 0.0)
    
    # Convert to HSV
    hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
    
    # Flatten to get all pixels
    pixels = hsv.reshape(-1, 3)
    
    # Calculate histogram for Hue channel
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    # Get dominant hue
    dominant_h = np.argmax(hist_h)
    dominant_s = np.argmax(hist_s)
    dominant_v = np.argmax(hist_v)
    
    # Calculate confidence (how dominant is this color)
    total_pixels = pixels.shape[0]
    confidence = float(hist_h[dominant_h]) / total_pixels
    
    # Classify color based on HSV
    color_name = classify_color_hsv(dominant_h, dominant_s, dominant_v)
    
    return (color_name, (dominant_h, dominant_s, dominant_v), confidence)


def classify_color_hsv(h, s, v):
    """
    Phan loai mau sac tu HSV values
    
    Args:
        h: Hue (0-179)
        s: Saturation (0-255)
        v: Value (0-255)
    
    Returns:
        color name string
    """
    # Low saturation = grayscale colors
    if s < 30:
        if v < 50:
            return "Black"
        elif v < 150:
            return "Gray"
        else:
            return "White"
    
    # High saturation = vivid colors
    # Hue ranges (OpenCV uses 0-179 for Hue)
    if h < 10 or h > 170:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 130:
        return "Blue"
    elif 130 <= h < 170:
        return "Purple"
    else:
        return "Unknown"


def classify_formality(color_name):
    """
    Phan loai do formal cua mau sac
    
    Args:
        color_name: ten mau sac
    
    Returns:
        (formality_level, score)
        formality_level: "Formal", "Business Casual", "Casual"
        score: 0-100
    """
    formal_colors = {
        "Black": ("Formal", 100),
        "White": ("Formal", 95),
        "Blue": ("Formal", 90),  # Navy blue
        "Gray": ("Formal", 85),
    }
    
    business_casual_colors = {
        "Purple": ("Business Casual", 70),
        "Green": ("Business Casual", 65),
    }
    
    casual_colors = {
        "Red": ("Casual", 50),
        "Orange": ("Casual", 40),
        "Yellow": ("Casual", 35),
    }
    
    if color_name in formal_colors:
        return formal_colors[color_name]
    elif color_name in business_casual_colors:
        return business_casual_colors[color_name]
    elif color_name in casual_colors:
        return casual_colors[color_name]
    else:
        return ("Unknown", 50)


def calculate_dress_color_score(color_name, confidence):
    """
    Tinh diem tong the cho mau sac trang phuc
    
    Args:
        color_name: ten mau sac
        confidence: do tin cay (0-1)
    
    Returns:
        dress_color_score (0-100)
    """
    formality_level, base_score = classify_formality(color_name)
    
    # Adjust score based on confidence
    # High confidence = more reliable score
    adjusted_score = base_score * (0.5 + 0.5 * confidence)
    
    return min(100, max(0, adjusted_score))


def analyze_dress_color(face_box, frame):
    """
    Phan tich mau sac trang phuc tu frame
    
    Args:
        face_box: (x, y, w, h) cua face
        frame: BGR image frame
    
    Returns:
        dict with analysis results
    """
    # Detect upper body region
    body_x1, body_y1, body_x2, body_y2 = detect_upper_body_region(
        face_box, frame.shape
    )
    
    # Extract upper body region
    body_region = frame[body_y1:body_y2, body_x1:body_x2]
    
    if body_region.size == 0:
        return {
            'color_name': 'Unknown',
            'formality_level': 'Unknown',
            'score': 50.0,
            'confidence': 0.0,
            'region': None
        }
    
    # Analyze dominant color
    color_name, hsv_values, confidence = analyze_dominant_color(body_region)
    
    # Classify formality
    formality_level, _ = classify_formality(color_name)
    
    # Calculate score
    score = calculate_dress_color_score(color_name, confidence)
    
    return {
        'color_name': color_name,
        'formality_level': formality_level,
        'score': score,
        'confidence': confidence,
        'hsv_values': hsv_values,
        'region': (body_x1, body_y1, body_x2, body_y2),
        'body_region': body_region  # For dress type detection
    }


def analyze_dress_complete(face_box, frame, use_yolo=True):
    """
    Phan tich day du: mau sac + loai trang phuc
    
    Args:
        face_box: (x, y, w, h) cua face
        frame: BGR image frame
        use_yolo: su dung YOLO de detect loai trang phuc
    
    Returns:
        dict with complete analysis
    """
    # Analyze color first
    color_result = analyze_dress_color(face_box, frame)
    
    if color_result['body_region'] is None:
        return color_result
    
    # Analyze dress type with YOLO
    try:
        from .dress_detection_yolo import analyze_dress_type
        type_result = analyze_dress_type(color_result['body_region'], use_yolo=use_yolo)
        
        # Combine scores: 60% color + 40% type
        combined_score = (color_result['score'] * 0.6 + type_result['score'] * 0.4)
        
        # Update result
        color_result.update({
            'dress_type': type_result['dress_type'],
            'type_score': type_result['score'],
            'type_confidence': type_result['confidence'],
            'detection_method': type_result['method'],
            'combined_score': combined_score
        })
    except Exception as e:
        # Fallback if YOLO fails
        color_result['dress_type'] = 'Unknown'
        color_result['combined_score'] = color_result['score']
    
    return color_result



def generate_dress_report(dress_samples):
    """
    Tao bao cao tong the ve trang phuc
    
    Args:
        dress_samples: list of dress analysis results
    
    Returns:
        dict with overall assessment and suggestions
    """
    if not dress_samples:
        return {
            'overall_category': 'Unknown',
            'score': 70.0,
            'report': 'Khong co du lieu trang phuc',
            'suggestions': []
        }
    
    # Get statistics
    from collections import Counter
    
    colors = [s.get('color_name', 'Unknown') for s in dress_samples]
    types = [s.get('dress_type', 'Unknown') for s in dress_samples]
    formality_levels = [s.get('formality_level', 'Unknown') for s in dress_samples]
    
    most_common_color = Counter(colors).most_common(1)[0][0]
    most_common_type = Counter(types).most_common(1)[0][0]
    most_common_formality = Counter(formality_levels).most_common(1)[0][0]
    
    # Calculate average score
    scores = [s.get('combined_score', s.get('score', 70)) for s in dress_samples]
    avg_score = np.mean(scores)
    
    # Determine overall category
    overall_category = determine_overall_category(
        most_common_color, 
        most_common_type, 
        most_common_formality,
        avg_score
    )
    
    # Generate report text
    report = generate_report_text(
        overall_category,
        most_common_color,
        most_common_type,
        avg_score
    )
    
    # Generate suggestions
    suggestions = generate_dress_suggestions(
        overall_category,
        most_common_color,
        most_common_type,
        avg_score
    )
    
    return {
        'overall_category': overall_category,
        'score': avg_score,
        'dominant_color': most_common_color,
        'dominant_type': most_common_type,
        'report': report,
        'suggestions': suggestions
    }


def determine_overall_category(color, dress_type, formality_level, score):
    """
    Xac dinh category tong the
    
    Args:
        color: dominant color
        dress_type: dominant dress type
        formality_level: dominant formality level
        score: average score
    
    Returns:
        overall category string
    """
    # Score-based classification
    if score >= 85:
        return "Formal"
    elif score >= 70:
        return "Business Casual"
    elif score >= 50:
        return "Casual"
    else:
        return "Inappropriate"


def generate_report_text(category, color, dress_type, score):
    """
    Tao text bao cao
    
    Args:
        category: overall category
        color: dominant color
        dress_type: dominant dress type
        score: average score
    
    Returns:
        report text string
    """
    report = f"DANH GIA TRANG PHUC\n"
    report += f"="*40 + "\n\n"
    
    report += f"Phan loai: {category}\n"
    report += f"Diem so: {score:.1f}/100\n\n"
    
    report += f"Mau sac chu dao: {color}\n"
    report += f"Loai trang phuc: {dress_type}\n\n"
    
    # Category-specific comments
    if category == "Formal":
        report += "Danh gia: Rat chuyen nghiep!\n"
        report += "Trang phuc phu hop cho phong van.\n"
    elif category == "Business Casual":
        report += "Danh gia: Kha tot.\n"
        report += "Trang phuc chap nhan duoc cho phong van.\n"
    elif category == "Casual":
        report += "Danh gia: Hoi thoai mai.\n"
        report += "Nen mac trang phuc chinh thong hon.\n"
    else:
        report += "Danh gia: Khong phu hop.\n"
        report += "Can thay doi trang phuc.\n"
    
    return report


def generate_dress_suggestions(category, color, dress_type, score):
    """
    Tao goi y cai thien
    
    Args:
        category: overall category
        color: dominant color
        dress_type: dominant dress type
        score: average score
    
    Returns:
        list of suggestions
    """
    suggestions = []
    
    # Category-based suggestions
    if category == "Inappropriate":
        suggestions.append("Mac ao so mi hoac vest")
        suggestions.append("Chon mau toi (den, xanh navy, trang)")
        suggestions.append("Tranh trang phuc qua thoai mai")
    
    elif category == "Casual":
        suggestions.append("Nen mac ao so mi thay vi ao thun")
        suggestions.append("Chon mau trang phuc chinh thong hon")
    
    elif category == "Business Casual":
        suggestions.append("Co the them vest de tang tinh chuyen nghiep")
        suggestions.append("Duy tri phong cach hien tai")
    
    else:  # Formal
        suggestions.append("Trang phuc rat tot!")
        suggestions.append("Tiep tuc duy tri")
    
    # Color-specific suggestions
    if color in ["Red", "Orange", "Yellow"]:
        suggestions.append("Mau sac qua noi bat - nen chon mau trung tinh")
    
    # Type-specific suggestions
    if dress_type in ["t-shirt", "hoodie"]:
        suggestions.append(f"Thay {dress_type} bang ao so mi")
    
    return suggestions


def get_dress_summary(dress_samples):
    """
    Lay tom tat ngan gon ve trang phuc
    
    Args:
        dress_samples: list of dress analysis results
    
    Returns:
        summary string for display
    """
    if not dress_samples:
        return "Khong co du lieu trang phuc"
    
    report_data = generate_dress_report(dress_samples)
    
    summary = f"{report_data['overall_category']} ({report_data['score']:.0f}/100)\n"
    summary += f"Mau: {report_data['dominant_color']}, "
    summary += f"Loai: {report_data['dominant_type']}\n"
    
    if report_data['suggestions']:
        summary += f"Goi y: {report_data['suggestions'][0]}"
    
    return summary
