# -*- coding: utf-8 -*-
"""
MediaPipe Face Detection
Fast face detection using Google MediaPipe
Speed: 100+ FPS on CPU, 200+ FPS on GPU
"""
import cv2
import mediapipe as mp
import numpy as np


class MediaPipeFaceDetector:
    """
    Fast face detection using MediaPipe
    Compatible with MTCNN interface
    """
    
    def __init__(self, min_detection_confidence=0.5, model_selection=0, min_tracking_confidence=0.5):
        """
        Initialize MediaPipe Face Detector
        
        Args:
            min_detection_confidence: Minimum confidence (0.0-1.0), default 0.5
            model_selection: 0 = short-range (2m), 1 = full-range (5m)
            min_tracking_confidence: Minimum tracking confidence (0.0-1.0)
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        self.min_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Face tracking để giảm jitter và false positives
        self.tracked_faces = []
        self.face_id_counter = 0
        
        # Temporal filtering - chỉ giữ faces xuất hiện liên tục
        self.face_history = []  # Lưu faces từ các frames trước
        self.history_length = 5  # TĂNG lên 5 frames để lọc false positives tốt hơn
        self.min_consecutive_frames = 3  # Face phải xuất hiện ít nhất 3 frames liên tiếp
    
    def detect_faces(self, image):
        """
        Detect faces in image
        
        Args:
            image: BGR image (OpenCV format) or RGB image
        
        Returns:
            List of face detections in MTCNN-compatible format:
            [
                {
                    'box': [x, y, width, height],
                    'confidence': 0.95,
                    'keypoints': {
                        'left_eye': (x, y),
                        'right_eye': (x, y),
                        'nose': (x, y),
                        'mouth_left': (x, y),
                        'mouth_right': (x, y)
                    }
                }
            ]
        """
        # Check if image is BGR or RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR, convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Detect faces
        results = self.detector.process(rgb_image)
        
        if not results.detections:
            return []
        
        # Convert to MTCNN-compatible format
        faces = []
        h, w = image.shape[:2]
        
        # Import config for filtering
        try:
            from .config import DETECTION_CONFIG
            min_face_size = DETECTION_CONFIG.get('min_face_size', 50)
            max_faces = DETECTION_CONFIG.get('max_faces', 3)
        except:
            min_face_size = 50
            max_faces = 3
        
        for detection in results.detections:
            # Get bounding box (relative coordinates)
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to absolute coordinates
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            # Filter by minimum size (loại bỏ faces quá nhỏ - false positives)
            if width < min_face_size or height < min_face_size:
                continue
            
            # Get confidence
            confidence = detection.score[0]
            
            # FACE QUALITY CHECK - loại bỏ false positives (NGHIÊM NGẶT HƠN)
            # 1. Kiểm tra tỷ lệ khung hình (aspect ratio) - NGHIÊM NGẶT HƠN
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.75 or aspect_ratio > 1.35:
                # Face không đúng tỷ lệ - NGHIÊM NGẶT HƠN (0.7-1.5 → 0.75-1.35)
                continue
            
            # 2. Kiểm tra diện tích face so với frame - NGHIÊM NGẶT HƠN
            face_area = width * height
            frame_area = w * h
            area_ratio = face_area / frame_area if frame_area > 0 else 0
            if area_ratio < 0.03:  # Face quá nhỏ (< 3% frame) - TĂNG từ 2%
                continue
            if area_ratio > 0.7:  # Face quá lớn (> 70% frame) - GIẢM từ 80%
                continue
            
            # 3. Kiểm tra vị trí - face không nên ở rìa
            center_x = x + width // 2
            center_y = y + height // 2
            # Face phải ở trung tâm frame (không ở 15% rìa)
            if center_x < w * 0.15 or center_x > w * 0.85:
                continue
            if center_y < h * 0.1 or center_y > h * 0.9:
                continue
            
            # Get keypoints (6 points in MediaPipe)
            keypoints = {}
            if detection.location_data.relative_keypoints:
                kps = detection.location_data.relative_keypoints
                
                # MediaPipe keypoints: right_eye, left_eye, nose, mouth, right_ear, left_ear
                # Convert to MTCNN format: left_eye, right_eye, nose, mouth_left, mouth_right
                keypoints = {
                    'right_eye': (int(kps[0].x * w), int(kps[0].y * h)),
                    'left_eye': (int(kps[1].x * w), int(kps[1].y * h)),
                    'nose': (int(kps[2].x * w), int(kps[2].y * h)),
                    'mouth': (int(kps[3].x * w), int(kps[3].y * h)),
                    'right_ear': (int(kps[4].x * w), int(kps[4].y * h)),
                    'left_ear': (int(kps[5].x * w), int(kps[5].y * h))
                }
                
                # Add MTCNN-compatible mouth keypoints (approximate)
                mouth_x, mouth_y = keypoints['mouth']
                mouth_width = abs(keypoints['left_eye'][0] - keypoints['right_eye'][0]) // 3
                keypoints['mouth_left'] = (mouth_x - mouth_width, mouth_y)
                keypoints['mouth_right'] = (mouth_x + mouth_width, mouth_y)
            
            faces.append({
                'box': [x, y, width, height],
                'confidence': confidence,
                'keypoints': keypoints
            })
        
        # Sort by confidence (highest first) và giới hạn số lượng
        faces.sort(key=lambda f: f['confidence'], reverse=True)
        faces = faces[:max_faces]  # Chỉ lấy top N faces
        
        # Temporal filtering - chỉ giữ faces xuất hiện liên tục
        # Thêm faces hiện tại vào history
        self.face_history.append(faces)
        if len(self.face_history) > self.history_length:
            self.face_history.pop(0)
        
        # Nếu chưa đủ history, trả về faces hiện tại
        if len(self.face_history) < self.history_length:
            return faces
        
        # Lọc faces: chỉ giữ faces xuất hiện trong ít nhất 2/3 frames gần đây
        filtered_faces = []
        for face in faces:
            x, y, w, h = face['box']
            
            # Đếm số lần face này xuất hiện trong history
            match_count = 0
            for hist_faces in self.face_history:
                for hist_face in hist_faces:
                    hx, hy, hw, hh = hist_face['box']
                    
                    # Kiểm tra overlap (IoU - Intersection over Union)
                    # Nếu 2 faces overlap > 50%, coi như cùng 1 face
                    x1_max = max(x, hx)
                    y1_max = max(y, hy)
                    x2_min = min(x + w, hx + hw)
                    y2_min = min(y + h, hy + hh)
                    
                    if x2_min > x1_max and y2_min > y1_max:
                        # Có overlap
                        intersection = (x2_min - x1_max) * (y2_min - y1_max)
                        area1 = w * h
                        area2 = hw * hh
                        union = area1 + area2 - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > 0.5:  # Overlap > 50%
                            match_count += 1
                            break
            
            # Chỉ giữ face nếu xuất hiện LIÊN TỤC trong ít nhất min_consecutive_frames
            # Điều này loại bỏ false positives "nhấp nháy"
            if match_count >= self.min_consecutive_frames:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def close(self):
        """Release resources"""
        if hasattr(self, 'detector'):
            self.detector.close()
    
    def __del__(self):
        """Destructor"""
        self.close()


# Test function
def test_mediapipe_detector():
    """Test MediaPipe face detector"""
    import time
    
    print("Testing MediaPipe Face Detector...")
    print("=" * 60)
    
    detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    fps_start = time.time()
    frame_count = 0
    fps_display = 0
    
    print("Press 'q' to quit")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        start_time = time.time()
        faces = detector.detect_faces(frame)
        detection_time = (time.time() - start_time) * 1000  # ms
        
        # Draw faces
        for face in faces:
            x, y, w, h = face['box']
            conf = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(frame, f"{conf:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw keypoints
            if face['keypoints']:
                for name, (kx, ky) in face['keypoints'].items():
                    if 'ear' not in name:  # Skip ears for cleaner display
                        cv2.circle(frame, (kx, ky), 3, (0, 0, 255), -1)
        
        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:
            fps_end = time.time()
            fps_display = frame_count / (fps_end - fps_start)
            fps_start = time.time()
            frame_count = 0
        
        # Display info
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {detection_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('MediaPipe Face Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("\nTest completed!")
    print(f"Average FPS: {fps_display:.1f}")
    print(f"Average Latency: {detection_time:.1f}ms")


if __name__ == "__main__":
    test_mediapipe_detector()
