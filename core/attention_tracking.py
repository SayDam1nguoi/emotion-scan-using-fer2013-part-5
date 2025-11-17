# -*- coding: utf-8 -*-
"""
Attention Tracking
Theo dõi sự tập trung vào màn hình cho phỏng vấn online
Đánh giá xem ứng viên có đang chú ý lắng nghe hay không
"""
import cv2
import numpy as np
from collections import deque


class AttentionTracker:
    """
    Theo dõi sự tập trung của người dùng vào màn hình
    """
    
    def __init__(self, history_size=150):  # 5 seconds at 30fps
        """
        Args:
            history_size: số frames để lưu lịch sử
        """
        self.history_size = history_size
        self.attention_history = deque(maxlen=history_size)
        self.gaze_history = deque(maxlen=history_size)
        self.head_pose_history = deque(maxlen=history_size)
        
    def analyze_attention(self, face_box, frame_shape, face_landmarks=None):
        """
        Phân tích sự tập trung dựa trên:
        - Vị trí khuôn mặt (head pose)
        - Hướng nhìn (gaze direction)
        - Độ ổn định (stability)
        
        Args:
            face_box: (x, y, w, h) face bounding box
            frame_shape: (height, width) của frame
            face_landmarks: optional facial landmarks
        
        Returns:
            dict with attention metrics
        """
        if not face_box:
            result = {
                'is_focused': False,
                'attention_score': 0.0,
                'gaze_status': 'No Face',
                'head_pose': 'Unknown',
                'distraction_level': 'High'
            }
            self.attention_history.append(0.0)
            return result
        
        x, y, w, h = face_box
        frame_h, frame_w = frame_shape[:2]
        
        # 1. Phân tích vị trí khuôn mặt (Head Pose)
        head_pose_score = self._analyze_head_pose(face_box, frame_shape)
        
        # 2. Phân tích hướng nhìn (Gaze Direction)
        gaze_score = self._analyze_gaze_direction(face_box, frame_shape)
        
        # 3. Phân tích độ ổn định (Stability)
        stability_score = self._analyze_stability()
        
        # Tính attention score tổng hợp
        attention_score = (
            head_pose_score * 0.4 +
            gaze_score * 0.4 +
            stability_score * 0.2
        )
        
        # Lưu vào history
        self.attention_history.append(attention_score)
        self.gaze_history.append(gaze_score)
        self.head_pose_history.append(head_pose_score)
        
        # Xác định trạng thái
        is_focused = attention_score >= 70.0
        
        # Gaze status
        if gaze_score >= 80:
            gaze_status = 'Đang nhìn màn hình'
        elif gaze_score >= 60:
            gaze_status = 'Nhìn gần màn hình'
        elif gaze_score >= 40:
            gaze_status = 'Nhìn lệch'
        else:
            gaze_status = 'Không nhìn màn hình'
        
        # Head pose
        if head_pose_score >= 80:
            head_pose = 'Thẳng'
        elif head_pose_score >= 60:
            head_pose = 'Hơi nghiêng'
        else:
            head_pose = 'Quay đi'
        
        # Distraction level
        if attention_score >= 80:
            distraction_level = 'Rất tập trung'
        elif attention_score >= 60:
            distraction_level = 'Tập trung'
        elif attention_score >= 40:
            distraction_level = 'Hơi mất tập trung'
        else:
            distraction_level = 'Mất tập trung'
        
        return {
            'is_focused': is_focused,
            'attention_score': attention_score,
            'gaze_status': gaze_status,
            'head_pose': head_pose,
            'distraction_level': distraction_level,
            'head_pose_score': head_pose_score,
            'gaze_score': gaze_score,
            'stability_score': stability_score
        }
    
    def _analyze_head_pose(self, face_box, frame_shape):
        """
        Phân tích tư thế đầu
        
        Returns:
            score (0-100): 100 = nhìn thẳng, 0 = quay hẳn đi
        """
        x, y, w, h = face_box
        frame_h, frame_w = frame_shape[:2]
        
        # Tính vị trí trung tâm khuôn mặt
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Tính offset từ trung tâm frame
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        offset_x = abs(face_center_x - frame_center_x) / frame_w
        offset_y = abs(face_center_y - frame_center_y) / frame_h
        
        # Tính aspect ratio của face (để detect xoay đầu)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Ideal aspect ratio: 0.75 - 0.85
        aspect_deviation = abs(aspect_ratio - 0.8) / 0.8
        
        # Score dựa trên offset và aspect ratio
        position_score = 100 * (1 - (offset_x + offset_y) / 2)
        aspect_score = 100 * (1 - min(aspect_deviation, 1.0))
        
        # Kết hợp
        head_pose_score = position_score * 0.7 + aspect_score * 0.3
        
        return max(0, min(100, head_pose_score))
    
    def _analyze_gaze_direction(self, face_box, frame_shape):
        """
        Phân tích hướng nhìn
        
        Returns:
            score (0-100): 100 = nhìn thẳng camera, 0 = nhìn xa
        """
        x, y, w, h = face_box
        frame_h, frame_w = frame_shape[:2]
        
        # Tính vị trí mắt ước lượng (1/3 từ trên xuống)
        eye_y = y + h // 3
        eye_center_x = x + w // 2
        
        # Tính khoảng cách từ trung tâm
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2
        
        # Offset ngang (quan trọng hơn cho gaze)
        horizontal_offset = abs(eye_center_x - frame_center_x) / frame_w
        
        # Offset dọc
        vertical_offset = abs(eye_y - frame_center_y) / frame_h
        
        # Gaze score (horizontal quan trọng hơn)
        gaze_score = 100 * (1 - (horizontal_offset * 0.7 + vertical_offset * 0.3))
        
        return max(0, min(100, gaze_score))
    
    def _analyze_stability(self):
        """
        Phân tích độ ổn định (không nhìn lung tung)
        
        Returns:
            score (0-100): 100 = rất ổn định, 0 = nhìn lung tung
        """
        if len(self.gaze_history) < 10:
            return 70.0  # Default
        
        # Tính độ biến thiên của gaze
        recent_gaze = list(self.gaze_history)[-30:]  # 1 second
        gaze_variance = np.var(recent_gaze)
        
        # Variance thấp = ổn định
        # Variance cao = nhìn lung tung
        stability_score = 100 * np.exp(-gaze_variance / 500)
        
        return max(0, min(100, stability_score))
    
    def get_attention_summary(self):
        """
        Tổng hợp attention trong toàn bộ session
        
        Returns:
            dict with summary metrics
        """
        if not self.attention_history:
            return {
                'avg_attention': 0.0,
                'focused_percentage': 0.0,
                'distracted_percentage': 0.0,
                'attention_rating': 'Không có dữ liệu',
                'recommendation': 'Không có dữ liệu'
            }
        
        attention_scores = list(self.attention_history)
        
        # Tính trung bình
        avg_attention = np.mean(attention_scores)
        
        # Tính % thời gian tập trung (>= 70)
        focused_count = sum(1 for score in attention_scores if score >= 70)
        focused_percentage = (focused_count / len(attention_scores)) * 100
        
        # Tính % thời gian mất tập trung (< 40)
        distracted_count = sum(1 for score in attention_scores if score < 40)
        distracted_percentage = (distracted_count / len(attention_scores)) * 100
        
        # Đánh giá
        if avg_attention >= 80:
            attention_rating = 'Xuất sắc'
            recommendation = 'Sự tập trung rất tốt! Tiếp tục duy trì.'
        elif avg_attention >= 70:
            attention_rating = 'Tốt'
            recommendation = 'Tập trung tốt. Cố gắng duy trì ổn định.'
        elif avg_attention >= 60:
            attention_rating = 'Trung bình'
            recommendation = 'Cần cải thiện sự tập trung. Nhìn thẳng vào camera nhiều hơn.'
        elif avg_attention >= 40:
            attention_rating = 'Yếu'
            recommendation = 'Sự tập trung kém. Cần nhìn vào màn hình và tránh nhìn lung tung.'
        else:
            attention_rating = 'Rất yếu'
            recommendation = 'Sự tập trung rất kém! Phải tập trung nhìn vào camera khi phỏng vấn.'
        
        return {
            'avg_attention': avg_attention,
            'focused_percentage': focused_percentage,
            'distracted_percentage': distracted_percentage,
            'attention_rating': attention_rating,
            'recommendation': recommendation,
            'total_frames': len(attention_scores)
        }
    
    def get_attention_timeline(self, window_size=30):
        """
        Lấy timeline attention theo từng khoảng thời gian
        
        Args:
            window_size: số frames mỗi window (30 = 1 giây)
        
        Returns:
            list of attention scores theo thời gian
        """
        if not self.attention_history:
            return []
        
        scores = list(self.attention_history)
        timeline = []
        
        for i in range(0, len(scores), window_size):
            window = scores[i:i+window_size]
            avg_score = np.mean(window)
            timeline.append({
                'time_sec': i // 30,
                'attention_score': avg_score,
                'status': 'Tập trung' if avg_score >= 70 else 'Mất tập trung'
            })
        
        return timeline
    
    def reset(self):
        """Reset tất cả history"""
        self.attention_history.clear()
        self.gaze_history.clear()
        self.head_pose_history.clear()


def format_attention_report(attention_summary):
    """
    Format attention summary thành text report
    
    Args:
        attention_summary: dict from get_attention_summary()
    
    Returns:
        formatted string
    """
    report = "SỰ TẬP TRUNG VÀO MÀN HÌNH:\n"
    report += "="*40 + "\n\n"
    
    report += f"Điểm trung bình: {attention_summary['avg_attention']:.1f}/100\n"
    report += f"Đánh giá: {attention_summary['attention_rating']}\n\n"
    
    report += f"Thời gian tập trung: {attention_summary['focused_percentage']:.1f}%\n"
    report += f"Thời gian mất tập trung: {attention_summary['distracted_percentage']:.1f}%\n\n"
    
    report += f"Gợi ý: {attention_summary['recommendation']}\n"
    
    return report
