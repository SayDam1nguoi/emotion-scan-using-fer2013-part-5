# -*- coding: utf-8 -*-
"""
Dress Detection using YOLO
Nhan dien loai trang phuc bang YOLO
"""
import cv2
import numpy as np

# YOLO will be imported only if available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


class DressDetector:
    """
    Dress detector using YOLO (Singleton)
    """
    _instance = None
    _model = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # Clothing categories and their formality scores
    CLOTHING_SCORES = {
        # Formal (90-100)
        'suit': 100,
        'blazer': 95,
        'dress_shirt': 90,
        'tie': 95,
        
        # Business Casual (70-85)
        'shirt': 75,
        'blouse': 75,
        'cardigan': 70,
        
        # Casual (40-65)
        't-shirt': 50,
        'polo': 60,
        'sweater': 65,
        
        # Inappropriate (<40)
        'hoodie': 35,
        'tank_top': 30,
        'sleeveless': 25,
    }
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLO model (only once)
        
        Args:
            model_path: path to YOLO weights
        """
        # Only initialize once
        if DressDetector._model is not None:
            self.model = DressDetector._model
            self.available = YOLO_AVAILABLE
            return
        
        self.model = None
        self.available = YOLO_AVAILABLE
        
        if YOLO_AVAILABLE:
            try:
                # Load YOLOv8 nano model (fastest)
                self.model = YOLO(model_path)
                DressDetector._model = self.model
                print(f"✅ YOLO model loaded: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load YOLO: {e}")
                self.available = False
    
    def detect_clothing(self, image_region):
        """
        Detect clothing items in image region
        
        Args:
            image_region: BGR image
        
        Returns:
            list of detected items with scores
        """
        if not self.available or self.model is None:
            return []
        
        try:
            # Run inference
            results = self.model(image_region, verbose=False)
            
            detected_items = []
            person_detected = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name
                    cls_id = int(box.cls[0])
                    class_name = result.names[cls_id]
                    confidence = float(box.conf[0])
                    
                    # Check if person detected
                    if 'person' in class_name.lower():
                        person_detected = True
                        
                        # Analyze clothing from person region
                        bbox = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, bbox)
                        person_region = image_region[y1:y2, x1:x2]
                        
                        # Infer clothing type from visual features
                        clothing_type = self._infer_clothing_type(person_region)
                        
                        detected_items.append({
                            'type': clothing_type,
                            'confidence': confidence * 0.7,  # Lower confidence for inference
                            'bbox': bbox
                        })
            
            # If no person detected, use simple method
            if not person_detected:
                clothing_type = self._infer_clothing_type(image_region)
                detected_items.append({
                    'type': clothing_type,
                    'confidence': 0.5,
                    'bbox': [0, 0, image_region.shape[1], image_region.shape[0]]
                })
            
            return detected_items
        
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def _infer_clothing_type(self, region):
        """
        Infer clothing type from visual features
        
        Args:
            region: BGR image of clothing region
        
        Returns:
            clothing type string
        """
        if region.size == 0 or region.shape[0] < 20 or region.shape[1] < 20:
            return 'shirt'
        
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Feature 1: Edge density (patterns vs solid)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Feature 2: Color analysis
        h_mean = np.mean(hsv[:,:,0])
        s_mean = np.mean(hsv[:,:,1])
        v_mean = np.mean(hsv[:,:,2])
        s_std = np.std(hsv[:,:,1])
        
        # Feature 3: Texture analysis (upper vs lower region)
        h, w = region.shape[:2]
        upper_region = region[:h//3, :]
        lower_region = region[2*h//3:, :]
        
        upper_brightness = np.mean(cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY))
        lower_brightness = np.mean(cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY))
        brightness_diff = abs(upper_brightness - lower_brightness)
        
        # Decision logic
        # Formal indicators: low edge density, dark colors, uniform
        if edge_density < 0.08 and v_mean < 100 and s_std < 30:
            # Dark, solid, uniform = likely suit/blazer
            return 'suit'
        
        elif edge_density < 0.1 and s_std < 40:
            # Solid color, low pattern = dress shirt or shirt
            if v_mean > 150:
                return 'dress_shirt'  # Light colored formal
            else:
                return 'shirt'  # Regular shirt
        
        elif brightness_diff > 30:
            # Significant brightness difference = might have collar/buttons
            return 'dress_shirt'
        
        elif edge_density > 0.15 or s_std > 60:
            # High pattern or colorful = casual
            if s_mean > 100:
                return 't-shirt'  # Colorful casual
            else:
                return 'sweater'  # Patterned
        
        else:
            # Default: regular shirt
            return 'shirt'
    
    def calculate_dress_type_score(self, detected_items):
        """
        Calculate professionalism score from detected items
        
        Args:
            detected_items: list of detected clothing items
        
        Returns:
            (dress_type, score, confidence)
        """
        if not detected_items:
            return ('Unknown', 70.0, 0.0)
        
        # Get highest confidence item
        best_item = max(detected_items, key=lambda x: x['confidence'])
        
        clothing_type = best_item['type']
        confidence = best_item['confidence']
        
        # Get score from lookup table
        score = self.CLOTHING_SCORES.get(clothing_type, 70.0)
        
        # Adjust by confidence
        adjusted_score = score * (0.5 + 0.5 * confidence)
        
        return (clothing_type, adjusted_score, confidence)


# Fallback: Simple pattern-based detection (no YOLO needed)
def detect_clothing_simple(image_region):
    """
    Simple clothing detection without YOLO
    Uses edge detection and color analysis
    
    Args:
        image_region: BGR image
    
    Returns:
        (dress_type, score)
    """
    if image_region.size == 0:
        return ('Unknown', 70.0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Analyze color variance
    hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
    h_std = np.std(hsv[:,:,0])
    s_std = np.std(hsv[:,:,1])
    
    # Heuristics:
    # - Low edge density + low color variance = solid color shirt (formal)
    # - High edge density = patterns (casual)
    # - High color variance = colorful (casual)
    
    if edge_density < 0.1 and s_std < 30:
        # Solid, uniform color - likely formal
        return ('dress_shirt', 85.0)
    elif edge_density > 0.2 or s_std > 60:
        # Patterns or colorful - casual
        return ('t-shirt', 50.0)
    else:
        # In between - business casual
        return ('shirt', 70.0)


def analyze_dress_type(body_region, use_yolo=True):
    """
    Main function to analyze dress type
    
    Args:
        body_region: BGR image of upper body
        use_yolo: whether to use YOLO (if available)
    
    Returns:
        dict with analysis results
    """
    if body_region.size == 0:
        return {
            'dress_type': 'Unknown',
            'score': 70.0,
            'confidence': 0.0,
            'method': 'none'
        }
    
    # Try YOLO first if requested and available
    if use_yolo and YOLO_AVAILABLE:
        try:
            detector = DressDetector()
            if detector.available:
                detected_items = detector.detect_clothing(body_region)
                if detected_items:
                    dress_type, score, confidence = detector.calculate_dress_type_score(detected_items)
                    return {
                        'dress_type': dress_type,
                        'score': score,
                        'confidence': confidence,
                        'method': 'yolo',
                        'detected_items': detected_items
                    }
        except Exception as e:
            print(f"YOLO detection failed: {e}")
    
    # Fallback to simple detection
    dress_type, score = detect_clothing_simple(body_region)
    return {
        'dress_type': dress_type,
        'score': score,
        'confidence': 0.7,  # Moderate confidence for simple method
        'method': 'simple'
    }
