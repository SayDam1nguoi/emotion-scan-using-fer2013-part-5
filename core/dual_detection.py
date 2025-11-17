# -*- coding: utf-8 -*-
"""
Dual Detection - Phân tích cả 2 người trong video call
Quét đồng thời:
- Camera: Chính bạn (người gọi)
- Screen: Người đối diện (người nhận)
"""
import cv2
import numpy as np
import threading
import queue
from collections import deque


class DualAnalyzer:
    """
    Phân tích đồng thời 2 nguồn: Camera + Screen Capture
    """
    
    def __init__(self):
        # Queues for frames
        self.camera_queue = queue.Queue(maxsize=5)
        self.screen_queue = queue.Queue(maxsize=5)
        
        # Results storage
        self.person1_results = {
            'name': 'Bạn (Camera)',
            'emotion_counts': [0, 0, 0, 0],  # Angry, Happy, Sad, Neutral
            'emotion_history': deque(maxlen=150),
            'attention_scores': deque(maxlen=150),
            'behavior_samples': {
                'posture': [],
                'eye_contact': [],
                'gestures': []
            }
        }
        
        self.person2_results = {
            'name': 'Người đối diện (Screen)',
            'emotion_counts': [0, 0, 0, 0],
            'emotion_history': deque(maxlen=150),
            'attention_scores': deque(maxlen=150),
            'behavior_samples': {
                'posture': [],
                'eye_contact': [],
                'gestures': []
            }
        }
        
        # Control flags
        self.running = False
        self.camera_thread = None
        self.screen_thread = None
    
    def start_camera_capture(self, camera_id=0):
        """
        Bắt đầu capture từ camera
        
        Args:
            camera_id: ID của camera
        """
        def capture_loop():
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            while self.running:
                ret, frame = cap.read()
                if ret:
                    # Put frame vào queue (non-blocking)
                    try:
                        self.camera_queue.put(frame, block=False)
                    except queue.Full:
                        # Skip frame nếu queue đầy
                        pass
            
            cap.release()
        
        self.camera_thread = threading.Thread(target=capture_loop, daemon=True)
        self.camera_thread.start()
    
    def start_screen_capture(self, capturer):
        """
        Bắt đầu capture từ màn hình
        
        Args:
            capturer: ScreenCapturer instance
        """
        def capture_loop():
            while self.running:
                frame = capturer.capture_frame()
                if frame is not None:
                    try:
                        self.screen_queue.put(frame, block=False)
                    except queue.Full:
                        pass
        
        self.screen_thread = threading.Thread(target=capture_loop, daemon=True)
        self.screen_thread.start()
    
    def get_camera_frame(self):
        """Lấy frame từ camera queue"""
        try:
            return self.camera_queue.get(block=False)
        except queue.Empty:
            return None
    
    def get_screen_frame(self):
        """Lấy frame từ screen queue"""
        try:
            return self.screen_queue.get(block=False)
        except queue.Empty:
            return None
    
    def update_person1_emotion(self, emotion_idx):
        """Cập nhật cảm xúc cho person 1 (camera)"""
        self.person1_results['emotion_counts'][emotion_idx] += 1
    
    def update_person2_emotion(self, emotion_idx):
        """Cập nhật cảm xúc cho person 2 (screen)"""
        self.person2_results['emotion_counts'][emotion_idx] += 1
    
    def update_person1_attention(self, score):
        """Cập nhật attention cho person 1"""
        self.person1_results['attention_scores'].append(score)
    
    def update_person2_attention(self, score):
        """Cập nhật attention cho person 2"""
        self.person2_results['attention_scores'].append(score)
    
    def get_comparison_report(self):
        """
        Tạo báo cáo so sánh 2 người
        
        Returns:
            dict with comparison data
        """
        from core.config import EMOTIONS
        
        # Calculate percentages
        total1 = sum(self.person1_results['emotion_counts'])
        total2 = sum(self.person2_results['emotion_counts'])
        
        if total1 == 0 or total2 == 0:
            return None
        
        person1_pct = [(count/total1)*100 for count in self.person1_results['emotion_counts']]
        person2_pct = [(count/total2)*100 for count in self.person2_results['emotion_counts']]
        
        # Calculate attention averages
        avg_attention1 = np.mean(list(self.person1_results['attention_scores'])) if self.person1_results['attention_scores'] else 0
        avg_attention2 = np.mean(list(self.person2_results['attention_scores'])) if self.person2_results['attention_scores'] else 0
        
        # Determine who is more positive
        happy1 = person1_pct[EMOTIONS.index('Happy')]
        happy2 = person2_pct[EMOTIONS.index('Happy')]
        
        neutral1 = person1_pct[EMOTIONS.index('Neutral')]
        neutral2 = person2_pct[EMOTIONS.index('Neutral')]
        
        positive1 = happy1 + neutral1
        positive2 = happy2 + neutral2
        
        return {
            'person1': {
                'name': self.person1_results['name'],
                'emotion_percentages': person1_pct,
                'emotion_counts': self.person1_results['emotion_counts'],
                'avg_attention': avg_attention1,
                'positive_score': positive1
            },
            'person2': {
                'name': self.person2_results['name'],
                'emotion_percentages': person2_pct,
                'emotion_counts': self.person2_results['emotion_counts'],
                'avg_attention': avg_attention2,
                'positive_score': positive2
            },
            'comparison': {
                'more_positive': 'person1' if positive1 > positive2 else 'person2',
                'more_focused': 'person1' if avg_attention1 > avg_attention2 else 'person2',
                'emotion_difference': [abs(p1 - p2) for p1, p2 in zip(person1_pct, person2_pct)]
            }
        }
    
    def start(self):
        """Bắt đầu dual analysis"""
        self.running = True
    
    def stop(self):
        """Dừng dual analysis"""
        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        if self.screen_thread:
            self.screen_thread.join(timeout=2)


def format_dual_report(comparison_data):
    """
    Format báo cáo so sánh thành text
    
    Args:
        comparison_data: dict from get_comparison_report()
    
    Returns:
        formatted string
    """
    from core.config import EMOTIONS
    
    if not comparison_data:
        return "Không có dữ liệu so sánh"
    
    p1 = comparison_data['person1']
    p2 = comparison_data['person2']
    comp = comparison_data['comparison']
    
    report = "SO SÁNH CẢM XÚC 2 NGƯỜI:\n"
    report += "="*60 + "\n\n"
    
    # Person 1
    report += f"👤 {p1['name']}:\n"
    for i, emotion in enumerate(EMOTIONS):
        report += f"   {emotion}: {p1['emotion_percentages'][i]:.1f}%\n"
    report += f"   Sự tập trung: {p1['avg_attention']:.1f}/100\n"
    report += f"   Điểm tích cực: {p1['positive_score']:.1f}%\n\n"
    
    # Person 2
    report += f"👥 {p2['name']}:\n"
    for i, emotion in enumerate(EMOTIONS):
        report += f"   {emotion}: {p2['emotion_percentages'][i]:.1f}%\n"
    report += f"   Sự tập trung: {p2['avg_attention']:.1f}/100\n"
    report += f"   Điểm tích cực: {p2['positive_score']:.1f}%\n\n"
    
    # Comparison
    report += "📊 SO SÁNH:\n"
    
    more_positive_name = p1['name'] if comp['more_positive'] == 'person1' else p2['name']
    report += f"   Tích cực hơn: {more_positive_name}\n"
    
    more_focused_name = p1['name'] if comp['more_focused'] == 'person1' else p2['name']
    report += f"   Tập trung hơn: {more_focused_name}\n"
    
    # Emotion differences
    report += "\n   Chênh lệch cảm xúc:\n"
    for i, emotion in enumerate(EMOTIONS):
        diff = comp['emotion_difference'][i]
        if diff > 10:  # Only show significant differences
            report += f"   - {emotion}: {diff:.1f}% khác biệt\n"
    
    return report
