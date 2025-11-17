# -*- coding: utf-8 -*-
"""
Background Analysis
Phan tich background trong video phong van
"""
import cv2
import numpy as np


def get_background_region(frame, face_boxes):
    """
    Lay vung background (loai tru faces)
    
    Args:
        frame: BGR image
        face_boxes: list of (x, y, w, h) face boxes
    
    Returns:
        background mask
    """
    h, w = frame.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Mask out faces
    for (x, y, fw, fh) in face_boxes:
        # Expand face region a bit
        margin = int(max(fw, fh) * 0.5)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + fw + margin)
        y2 = min(h, y + fh + margin)
        
        mask[y1:y2, x1:x2] = 0
    
    return mask


def analyze_edge_density(frame, mask=None):
    """
    Phan tich edge density (nhieu canh = lon xon)
    
    Args:
        frame: BGR image
        mask: optional mask to focus on specific region
    
    Returns:
        edge_density (0-1)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Apply mask if provided
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        total_pixels = np.sum(mask > 0)
    else:
        total_pixels = edges.size
    
    # Calculate density
    edge_pixels = np.sum(edges > 0)
    density = edge_pixels / total_pixels if total_pixels > 0 else 0
    
    return density


def analyze_texture_complexity(frame, mask=None):
    """
    Phan tich texture complexity
    
    Args:
        frame: BGR image
        mask: optional mask
    
    Returns:
        complexity score (0-1)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply mask
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Calculate standard deviation (high = complex texture)
    std_dev = np.std(gray)
    
    # Normalize to 0-1 (assume max std_dev = 80)
    complexity = min(std_dev / 80.0, 1.0)
    
    return complexity


def calculate_cleanliness_score(edge_density, texture_complexity):
    """
    Tinh diem do sach/gon gang
    
    Args:
        edge_density: edge density (0-1)
        texture_complexity: texture complexity (0-1)
    
    Returns:
        cleanliness_score (0-100)
    """
    # Low edge density + low complexity = clean
    # High edge density + high complexity = messy
    
    messiness = (edge_density * 0.6 + texture_complexity * 0.4)
    cleanliness = 1.0 - messiness
    
    # Convert to 0-100 scale
    score = cleanliness * 100
    
    return score


def classify_cleanliness(score):
    """
    Phan loai do sach
    
    Args:
        score: cleanliness score (0-100)
    
    Returns:
        (category, color)
    """
    if score > 80:
        return ("Clean", (0, 255, 0))  # Green
    elif score > 50:
        return ("Moderate", (0, 165, 255))  # Orange
    else:
        return ("Messy", (0, 0, 255))  # Red


def analyze_background_cleanliness(frame, face_boxes=None):
    """
    Phan tich tong the do sach cua background
    
    Args:
        frame: BGR image
        face_boxes: list of face boxes to exclude
    
    Returns:
        dict with analysis results
    """
    if frame.size == 0:
        return {
            'score': 50.0,
            'category': 'Unknown',
            'edge_density': 0.0,
            'texture_complexity': 0.0
        }
    
    # Get background mask (exclude faces)
    mask = None
    if face_boxes and len(face_boxes) > 0:
        mask = get_background_region(frame, face_boxes)
    
    # Analyze
    edge_density = analyze_edge_density(frame, mask)
    texture_complexity = analyze_texture_complexity(frame, mask)
    
    # Calculate score
    score = calculate_cleanliness_score(edge_density, texture_complexity)
    category, color = classify_cleanliness(score)
    
    return {
        'score': score,
        'category': category,
        'edge_density': edge_density,
        'texture_complexity': texture_complexity,
        'color': color
    }


def get_cleanliness_summary(cleanliness_samples):
    """
    Tao summary tu nhieu samples
    
    Args:
        cleanliness_samples: list of cleanliness analysis results
    
    Returns:
        summary dict
    """
    if not cleanliness_samples:
        return {
            'avg_score': 70.0,
            'category': 'Moderate',
            'recommendation': 'Khong co du lieu'
        }
    
    # Calculate average
    scores = [s['score'] for s in cleanliness_samples]
    avg_score = np.mean(scores)
    
    category, _ = classify_cleanliness(avg_score)
    
    # Generate recommendation
    if avg_score > 80:
        recommendation = "Background sach se, gon gang. Tot!"
    elif avg_score > 50:
        recommendation = "Background kha on. Co the don dep them."
    else:
        recommendation = "Background lon xon! Nen don dep hoac doi vi tri."
    
    return {
        'avg_score': avg_score,
        'category': category,
        'recommendation': recommendation
    }



# ===== Task 2.2: Environment Classification =====

# Environment categories and their scores
ENVIRONMENT_SCORES = {
    'Office': 100,
    'Home': 80,
    'Living Room': 75,
    'Unknown': 70,
    'Bedroom': 50,
    'Kitchen': 45,
    'Outdoor': 30,
    'Restaurant': 25,
}

# ImageNet classes mapping to environments
IMAGENET_TO_ENVIRONMENT = {
    # Office-related
    'desk': 'Office',
    'monitor': 'Office',
    'laptop': 'Office',
    'computer': 'Office',
    'office': 'Office',
    'bookcase': 'Office',
    
    # Home-related
    'couch': 'Living Room',
    'sofa': 'Living Room',
    'television': 'Living Room',
    'table': 'Home',
    'chair': 'Home',
    'lamp': 'Home',
    'wall': 'Home',
    
    # Bedroom
    'bed': 'Bedroom',
    'pillow': 'Bedroom',
    'wardrobe': 'Bedroom',
    
    # Kitchen
    'refrigerator': 'Kitchen',
    'oven': 'Kitchen',
    'microwave': 'Kitchen',
    
    # Outdoor
    'tree': 'Outdoor',
    'sky': 'Outdoor',
    'grass': 'Outdoor',
}


def classify_environment_simple(frame):
    """
    Phan loai moi truong don gian (khong dung deep learning)
    Dua tren color va texture analysis
    
    Args:
        frame: BGR image
    
    Returns:
        (environment_type, confidence)
    """
    # Analyze colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate color statistics
    h_mean = np.mean(hsv[:,:,0])
    s_mean = np.mean(hsv[:,:,1])
    v_mean = np.mean(hsv[:,:,2])
    
    # Analyze edges (furniture = many edges)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Simple heuristics
    if edge_density > 0.15:
        # Many edges = indoor with furniture
        if v_mean < 100:
            return ('Bedroom', 0.6)  # Dark = bedroom
        else:
            return ('Office', 0.6)  # Bright with furniture = office
    elif s_mean < 50 and v_mean > 150:
        # Low saturation, bright = plain wall
        return ('Home', 0.7)
    elif h_mean > 30 and h_mean < 90:
        # Green hues = outdoor
        return ('Outdoor', 0.6)
    else:
        return ('Unknown', 0.5)


def classify_environment_mobilenet(frame):
    """
    Phan loai moi truong bang MobileNetV2 (ImageNet)
    
    Args:
        frame: BGR image
    
    Returns:
        (environment_type, confidence)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
        
        # Load model (singleton pattern)
        if not hasattr(classify_environment_mobilenet, 'model'):
            classify_environment_mobilenet.model = MobileNetV2(weights='imagenet')
        
        model = classify_environment_mobilenet.model
        
        # Preprocess
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # Predict
        preds = model.predict(img, verbose=0)
        decoded = decode_predictions(preds, top=5)[0]
        
        # Map to environment
        for (imagenet_id, class_name, score) in decoded:
            class_lower = class_name.lower()
            
            # Check mapping
            for key, env in IMAGENET_TO_ENVIRONMENT.items():
                if key in class_lower:
                    return (env, float(score))
        
        # No match found
        return ('Unknown', 0.5)
    
    except Exception as e:
        print(f"MobileNet classification failed: {e}")
        # Fallback to simple method
        return classify_environment_simple(frame)


def analyze_environment(frame, use_deep_learning=True):
    """
    Phan tich moi truong
    
    Args:
        frame: BGR image
        use_deep_learning: su dung MobileNet hay khong
    
    Returns:
        dict with environment analysis
    """
    if frame.size == 0:
        return {
            'environment_type': 'Unknown',
            'confidence': 0.0,
            'score': 70.0,
            'suitable': True
        }
    
    # Classify environment
    if use_deep_learning:
        env_type, confidence = classify_environment_mobilenet(frame)
    else:
        env_type, confidence = classify_environment_simple(frame)
    
    # Get score
    score = ENVIRONMENT_SCORES.get(env_type, 70.0)
    
    # Check suitability
    suitable = score >= 70
    
    return {
        'environment_type': env_type,
        'confidence': confidence,
        'score': score,
        'suitable': suitable
    }


def get_environment_summary(environment_samples):
    """
    Tao summary tu nhieu samples
    
    Args:
        environment_samples: list of environment analysis results
    
    Returns:
        summary dict
    """
    if not environment_samples:
        return {
            'dominant_environment': 'Unknown',
            'avg_score': 70.0,
            'suitable': True,
            'recommendation': 'Khong co du lieu'
        }
    
    # Get most common environment
    from collections import Counter
    env_types = [s['environment_type'] for s in environment_samples]
    dominant_env = Counter(env_types).most_common(1)[0][0]
    
    # Calculate average score
    scores = [s['score'] for s in environment_samples]
    avg_score = np.mean(scores)
    
    suitable = avg_score >= 70
    
    # Generate recommendation
    if dominant_env == 'Office':
        recommendation = "Moi truong van phong - Ly tuong!"
    elif dominant_env in ['Home', 'Living Room']:
        recommendation = "Moi truong nha rieng - Chap nhan duoc"
    elif dominant_env == 'Bedroom':
        recommendation = "Phong ngu - Khong phu hop! Nen doi sang phong khac"
    elif dominant_env == 'Outdoor':
        recommendation = "Ngoai troi - Khong phu hop cho phong van!"
    else:
        recommendation = "Moi truong khong xac dinh - Nen chon vi tri ro rang hon"
    
    return {
        'dominant_environment': dominant_env,
        'avg_score': avg_score,
        'suitable': suitable,
        'recommendation': recommendation
    }



# ===== Task 2.3: Inappropriate Objects Detection =====

# Inappropriate objects for interview
INAPPROPRIATE_OBJECTS = {
    # Very inappropriate (high penalty)
    'bed': -30,
    'toilet': -40,
    'bottle': -15,
    'wine glass': -20,
    'beer': -25,
    
    # Moderately inappropriate
    'teddy bear': -10,
    'sports ball': -10,
    'skateboard': -10,
    'surfboard': -10,
    'frisbee': -10,
    
    # Slightly inappropriate
    'backpack': -5,
    'handbag': -5,
    'suitcase': -5,
    'umbrella': -5,
}

# Professional objects (bonus points)
PROFESSIONAL_OBJECTS = {
    'laptop': 10,
    'keyboard': 8,
    'mouse': 5,
    'book': 8,
    'clock': 5,
    'potted plant': 5,
}


def detect_objects_yolo(frame):
    """
    Detect objects bang YOLO
    
    Args:
        frame: BGR image
    
    Returns:
        list of detected objects
    """
    try:
        from ultralytics import YOLO
        
        # Load model (singleton)
        if not hasattr(detect_objects_yolo, 'model'):
            detect_objects_yolo.model = YOLO('yolov8n.pt')
        
        model = detect_objects_yolo.model
        
        # Run detection
        results = model(frame, verbose=False)
        
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = result.names[cls_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                detected_objects.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return detected_objects
    
    except Exception as e:
        print(f"YOLO object detection failed: {e}")
        return []


def analyze_inappropriate_objects(frame):
    """
    Phan tich cac vat the khong phu hop
    
    Args:
        frame: BGR image
    
    Returns:
        dict with analysis results
    """
    if frame.size == 0:
        return {
            'inappropriate_objects': [],
            'professional_objects': [],
            'penalty_score': 0,
            'bonus_score': 0,
            'total_adjustment': 0
        }
    
    # Detect all objects
    detected_objects = detect_objects_yolo(frame)
    
    inappropriate = []
    professional = []
    penalty_score = 0
    bonus_score = 0
    
    for obj in detected_objects:
        class_name = obj['class']
        confidence = obj['confidence']
        
        # Check if inappropriate
        if class_name in INAPPROPRIATE_OBJECTS:
            penalty = INAPPROPRIATE_OBJECTS[class_name] * confidence
            penalty_score += penalty
            inappropriate.append({
                'object': class_name,
                'confidence': confidence,
                'penalty': penalty
            })
        
        # Check if professional
        elif class_name in PROFESSIONAL_OBJECTS:
            bonus = PROFESSIONAL_OBJECTS[class_name] * confidence
            bonus_score += bonus
            professional.append({
                'object': class_name,
                'confidence': confidence,
                'bonus': bonus
            })
    
    total_adjustment = penalty_score + bonus_score
    
    return {
        'inappropriate_objects': inappropriate,
        'professional_objects': professional,
        'penalty_score': penalty_score,
        'bonus_score': bonus_score,
        'total_adjustment': total_adjustment
    }


def get_objects_summary(object_samples):
    """
    Tao summary tu nhieu samples
    
    Args:
        object_samples: list of object analysis results
    
    Returns:
        summary dict
    """
    if not object_samples:
        return {
            'has_inappropriate': False,
            'inappropriate_count': 0,
            'professional_count': 0,
            'avg_penalty': 0,
            'avg_bonus': 0,
            'recommendation': 'Khong co du lieu'
        }
    
    # Aggregate all objects
    all_inappropriate = []
    all_professional = []
    total_penalty = 0
    total_bonus = 0
    
    for sample in object_samples:
        all_inappropriate.extend(sample['inappropriate_objects'])
        all_professional.extend(sample['professional_objects'])
        total_penalty += sample['penalty_score']
        total_bonus += sample['bonus_score']
    
    # Average penalties/bonuses
    avg_penalty = total_penalty / len(object_samples)
    avg_bonus = total_bonus / len(object_samples)
    
    has_inappropriate = len(all_inappropriate) > 0
    
    # Generate recommendation
    if has_inappropriate:
        from collections import Counter
        most_common = Counter([obj['object'] for obj in all_inappropriate]).most_common(1)[0][0]
        recommendation = f"Phat hien: {most_common}! Loai bo hoac che kin."
    elif len(all_professional) > 0:
        recommendation = "Background chuyen nghiep. Tot!"
    else:
        recommendation = "Background trung tinh."
    
    return {
        'has_inappropriate': has_inappropriate,
        'inappropriate_count': len(all_inappropriate),
        'professional_count': len(all_professional),
        'avg_penalty': avg_penalty,
        'avg_bonus': avg_bonus,
        'recommendation': recommendation
    }



# ===== Task 2.4: Background Lighting Quality =====

def detect_backlight(frame, face_boxes=None):
    """
    Phat hien backlight (anh sang tu phia sau)
    
    Args:
        frame: BGR image
        face_boxes: list of face boxes
    
    Returns:
        dict with backlight analysis
    """
    if frame.size == 0:
        return {
            'has_backlight': False,
            'backlight_severity': 0.0,
            'face_brightness': 128.0,
            'background_brightness': 128.0
        }
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get face region brightness
    face_brightness = 128.0
    if face_boxes and len(face_boxes) > 0:
        face_mask = np.zeros((h, w), dtype=np.uint8)
        for (x, y, fw, fh) in face_boxes:
            face_mask[y:y+fh, x:x+fw] = 255
        
        face_pixels = gray[face_mask > 0]
        if len(face_pixels) > 0:
            face_brightness = np.mean(face_pixels)
    
    # Get background brightness (exclude faces)
    bg_mask = np.ones((h, w), dtype=np.uint8) * 255
    if face_boxes and len(face_boxes) > 0:
        for (x, y, fw, fh) in face_boxes:
            margin = int(max(fw, fh) * 0.5)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + fw + margin)
            y2 = min(h, y + fh + margin)
            bg_mask[y1:y2, x1:x2] = 0
    
    bg_pixels = gray[bg_mask > 0]
    background_brightness = np.mean(bg_pixels) if len(bg_pixels) > 0 else 128.0
    
    # Detect backlight: background much brighter than face
    brightness_diff = background_brightness - face_brightness
    
    # Backlight if background is 30+ brighter than face
    has_backlight = brightness_diff > 30
    backlight_severity = max(0, min(1, brightness_diff / 100.0))
    
    return {
        'has_backlight': has_backlight,
        'backlight_severity': backlight_severity,
        'face_brightness': face_brightness,
        'background_brightness': background_brightness,
        'brightness_diff': brightness_diff
    }


def analyze_face_background_contrast(frame, face_boxes=None):
    """
    Phan tich do tuong phan giua khuon mat va background
    
    Args:
        frame: BGR image
        face_boxes: list of face boxes
    
    Returns:
        dict with contrast analysis
    """
    if frame.size == 0 or not face_boxes or len(face_boxes) == 0:
        return {
            'contrast_ratio': 1.0,
            'contrast_quality': 'Unknown',
            'score': 70.0
        }
    
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get face brightness
    face_mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y, fw, fh) in face_boxes:
        face_mask[y:y+fh, x:x+fw] = 255
    
    face_pixels = gray[face_mask > 0]
    face_brightness = np.mean(face_pixels) if len(face_pixels) > 0 else 128.0
    
    # Get background brightness
    bg_mask = np.ones((h, w), dtype=np.uint8) * 255
    for (x, y, fw, fh) in face_boxes:
        margin = int(max(fw, fh) * 0.5)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + fw + margin)
        y2 = min(h, y + fh + margin)
        bg_mask[y1:y2, x1:x2] = 0
    
    bg_pixels = gray[bg_mask > 0]
    bg_brightness = np.mean(bg_pixels) if len(bg_pixels) > 0 else 128.0
    
    # Calculate contrast ratio
    # Good contrast: face slightly brighter than background
    # Bad contrast: face much darker (backlight) or same brightness
    
    if bg_brightness > 0:
        contrast_ratio = face_brightness / bg_brightness
    else:
        contrast_ratio = 1.0
    
    # Ideal contrast: 1.1 to 1.5 (face slightly brighter)
    # Score based on how close to ideal
    if 1.1 <= contrast_ratio <= 1.5:
        contrast_quality = 'Excellent'
        score = 100.0
    elif 1.0 <= contrast_ratio < 1.1:
        contrast_quality = 'Good'
        score = 85.0
    elif 0.9 <= contrast_ratio < 1.0:
        contrast_quality = 'Fair'
        score = 70.0
    elif 0.7 <= contrast_ratio < 0.9:
        contrast_quality = 'Poor (Backlight)'
        score = 50.0
    else:
        contrast_quality = 'Very Poor (Strong Backlight)'
        score = 30.0
    
    return {
        'contrast_ratio': contrast_ratio,
        'contrast_quality': contrast_quality,
        'score': score,
        'face_brightness': face_brightness,
        'background_brightness': bg_brightness
    }


def suggest_camera_position(backlight_result, contrast_result):
    """
    Goi y vi tri camera tot hon
    
    Args:
        backlight_result: result from detect_backlight()
        contrast_result: result from analyze_face_background_contrast()
    
    Returns:
        list of suggestions
    """
    suggestions = []
    
    # Check backlight
    if backlight_result['has_backlight']:
        severity = backlight_result['backlight_severity']
        if severity > 0.7:
            suggestions.append("⚠️ BACKLIGHT NGHIEM TRONG! Quay lung lai voi cua so/den")
        elif severity > 0.4:
            suggestions.append("⚠️ Co backlight. Nen thay doi vi tri camera")
        else:
            suggestions.append("💡 Backlight nhe. Co the cai thien bang cach dieu chinh goc")
    
    # Check contrast
    contrast_score = contrast_result['score']
    if contrast_score < 50:
        suggestions.append("⚠️ Tuong phan kem! Khuon mat qua toi so voi background")
        suggestions.append("💡 Goi y: Dat camera doi dien voi nguon sang (cua so/den)")
    elif contrast_score < 70:
        suggestions.append("💡 Tuong phan kha on. Co the cai thien them")
    else:
        suggestions.append("✅ Tuong phan tot!")
    
    # Specific brightness suggestions
    face_brightness = backlight_result['face_brightness']
    bg_brightness = backlight_result['background_brightness']
    
    if face_brightness < 80:
        suggestions.append("💡 Khuon mat qua toi. Bat them den hoac mo rem")
    elif face_brightness > 200:
        suggestions.append("💡 Khuon mat qua sang. Giam anh sang truc tiep")
    
    if bg_brightness > 200:
        suggestions.append("💡 Background qua sang. Tranh nguon sang phia sau")
    
    # Position suggestions
    if backlight_result['has_backlight'] or contrast_score < 70:
        suggestions.append("\n📍 VI TRI CAMERA LY TUONG:")
        suggestions.append("  1. Camera doi dien voi cua so (nguon sang tu phia truoc)")
        suggestions.append("  2. Nguon sang ben canh (45 do)")
        suggestions.append("  3. Tranh nguon sang truc tiep phia sau")
    
    return suggestions


def calculate_lighting_quality_score(backlight_result, contrast_result, overall_brightness):
    """
    Tinh diem chat luong anh sang tong the
    
    Args:
        backlight_result: result from detect_backlight()
        contrast_result: result from analyze_face_background_contrast()
        overall_brightness: overall frame brightness (0-255)
    
    Returns:
        lighting_quality_score (0-100)
    """
    # Component scores
    
    # 1. Backlight penalty (40% weight)
    backlight_penalty = backlight_result['backlight_severity'] * 40
    backlight_score = 40 - backlight_penalty
    
    # 2. Contrast score (40% weight)
    contrast_score = contrast_result['score'] * 0.4
    
    # 3. Overall brightness score (20% weight)
    # Ideal: 100-180
    if 100 <= overall_brightness <= 180:
        brightness_score = 20.0
    elif 80 <= overall_brightness < 100 or 180 < overall_brightness <= 200:
        brightness_score = 15.0
    elif 60 <= overall_brightness < 80 or 200 < overall_brightness <= 220:
        brightness_score = 10.0
    else:
        brightness_score = 5.0
    
    # Total score
    total_score = backlight_score + contrast_score + brightness_score
    
    return max(0, min(100, total_score))


def analyze_background_lighting(frame, face_boxes=None):
    """
    Phan tich tong the anh sang background (Task 2.4)
    
    Args:
        frame: BGR image
        face_boxes: list of face boxes
    
    Returns:
        dict with complete lighting analysis
    """
    if frame.size == 0:
        return {
            'backlight': {'has_backlight': False, 'backlight_severity': 0.0},
            'contrast': {'contrast_ratio': 1.0, 'score': 70.0},
            'lighting_quality_score': 70.0,
            'suggestions': ['Khong co du lieu']
        }
    
    # Overall brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overall_brightness = np.mean(gray)
    
    # Detect backlight
    backlight_result = detect_backlight(frame, face_boxes)
    
    # Analyze contrast
    contrast_result = analyze_face_background_contrast(frame, face_boxes)
    
    # Calculate quality score
    quality_score = calculate_lighting_quality_score(
        backlight_result, contrast_result, overall_brightness
    )
    
    # Generate suggestions
    suggestions = suggest_camera_position(backlight_result, contrast_result)
    
    return {
        'backlight': backlight_result,
        'contrast': contrast_result,
        'overall_brightness': overall_brightness,
        'lighting_quality_score': quality_score,
        'suggestions': suggestions
    }


def get_lighting_quality_summary(lighting_samples):
    """
    Tao summary tu nhieu samples
    
    Args:
        lighting_samples: list of lighting analysis results
    
    Returns:
        summary dict
    """
    if not lighting_samples:
        return {
            'avg_quality_score': 70.0,
            'has_backlight_issues': False,
            'avg_contrast_score': 70.0,
            'recommendation': 'Khong co du lieu'
        }
    
    # Calculate averages
    quality_scores = [s['lighting_quality_score'] for s in lighting_samples]
    avg_quality = np.mean(quality_scores)
    
    contrast_scores = [s['contrast']['score'] for s in lighting_samples]
    avg_contrast = np.mean(contrast_scores)
    
    # Check backlight issues
    backlight_count = sum(1 for s in lighting_samples if s['backlight']['has_backlight'])
    has_backlight_issues = backlight_count > len(lighting_samples) * 0.3  # >30% frames
    
    # Generate recommendation
    if avg_quality >= 80:
        recommendation = "Chat luong anh sang tot! Vi tri camera ly tuong."
    elif avg_quality >= 60:
        if has_backlight_issues:
            recommendation = "Co van de backlight. Nen thay doi vi tri camera."
        else:
            recommendation = "Anh sang kha on. Co the cai thien them."
    else:
        if has_backlight_issues:
            recommendation = "BACKLIGHT NGHIEM TRONG! Phai thay doi vi tri camera ngay."
        else:
            recommendation = "Chat luong anh sang kem. Nen dieu chinh nguon sang."
    
    return {
        'avg_quality_score': avg_quality,
        'has_backlight_issues': has_backlight_issues,
        'backlight_percentage': (backlight_count / len(lighting_samples)) * 100,
        'avg_contrast_score': avg_contrast,
        'recommendation': recommendation
    }



def generate_background_report(background_samples, environment_samples=None, 
                               object_samples=None, lighting_quality_samples=None):
    """
    Tạo báo cáo background đơn giản
    
    Args:
        background_samples: list of background scores
        environment_samples: optional (not used)
        object_samples: optional (not used)
        lighting_quality_samples: optional (not used)
    
    Returns:
        dict with background report
    """
    if not background_samples:
        return {
            'cleanliness_score': 70.0,
            'object_summary': {
                'avg_penalty': 0.0,
                'avg_bonus': 0.0
            }
        }
    
    # Calculate average score
    avg_score = np.mean(background_samples)
    
    return {
        'cleanliness_score': avg_score,
        'object_summary': {
            'avg_penalty': 0.0,
            'avg_bonus': 0.0
        }
    }
