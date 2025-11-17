# -*- coding: utf-8 -*-
"""
Affectiva-Style Scoring System
Tính điểm chuyên nghiệp theo chuẩn Affectiva
"""
import numpy as np
from .config import EMOTIONS


class AffectivaScorer:
    """
    Scoring system theo phong cách Affectiva
    """
    
    # Weights cho từng component
    WEIGHTS = {
        'emotion': 0.35,      # 35% - Cảm xúc
        'appearance': 0.30,   # 30% - Ngoại hình
        'behavior': 0.25,     # 25% - Hành vi
        'technical': 0.10     # 10% - Kỹ thuật
    }
    
    # Emotion scoring weights
    EMOTION_WEIGHTS = {
        'Happy': 1.0,      # Tích cực nhất
        'Neutral': 0.8,    # Chuyên nghiệp
        'Sad': 0.3,        # Tiêu cực nhẹ
        'Angry': 0.0       # Tiêu cực nặng
    }
    
    # Appearance sub-weights
    APPEARANCE_WEIGHTS = {
        'dress_code': 0.50,    # 50% của appearance
        'background': 0.33,    # 33% của appearance
        'lighting': 0.17       # 17% của appearance
    }
    
    # Behavior sub-weights
    BEHAVIOR_WEIGHTS = {
        'eye_contact': 0.40,   # 40% của behavior
        'posture': 0.32,       # 32% của behavior
        'gestures': 0.28       # 28% của behavior
    }
    
    def __init__(self):
        """Initialize scorer"""
        self.scores = {}
        
    def calculate_emotion_score(self, emotion_counts):
        """
        Tính điểm cảm xúc (0-100)
        
        Args:
            emotion_counts: list of emotion counts [Angry, Happy, Sad, Neutral]
        
        Returns:
            emotion_score (0-100)
        """
        total = sum(emotion_counts)
        if total == 0:
            return 50.0  # Neutral score
        
        # Calculate percentages
        percentages = [count/total for count in emotion_counts]
        
        # Calculate weighted score
        score = 0.0
        for i, emotion in enumerate(EMOTIONS):
            if emotion in self.EMOTION_WEIGHTS:
                score += percentages[i] * self.EMOTION_WEIGHTS[emotion]
        
        # Convert to 0-100 scale
        emotion_score = score * 100
        
        self.scores['emotion'] = {
            'score': emotion_score,
            'breakdown': {
                EMOTIONS[i]: {
                    'percentage': percentages[i] * 100,
                    'count': emotion_counts[i]
                } for i in range(len(EMOTIONS))
            },
            'dominant': EMOTIONS[np.argmax(emotion_counts)]
        }
        
        return emotion_score
    
    def calculate_appearance_score(self, dress_score=None, background_score=None, 
                                   lighting_score=None):
        """
        Tính điểm ngoại hình (0-100)
        
        Args:
            dress_score: điểm trang phục (0-100)
            background_score: điểm background (0-100)
            lighting_score: điểm ánh sáng (0-100)
        
        Returns:
            appearance_score (0-100)
        """
        # Default scores if not provided
        dress_score = dress_score or 70.0
        background_score = background_score or 70.0
        lighting_score = lighting_score or 70.0
        
        appearance_score = (
            dress_score * self.APPEARANCE_WEIGHTS['dress_code'] +
            background_score * self.APPEARANCE_WEIGHTS['background'] +
            lighting_score * self.APPEARANCE_WEIGHTS['lighting']
        )
        
        self.scores['appearance'] = {
            'score': appearance_score,
            'breakdown': {
                'dress_code': dress_score,
                'background': background_score,
                'lighting': lighting_score
            }
        }
        
        return appearance_score
    
    def calculate_behavior_score(self, eye_contact_score=None, posture_score=None,
                                 gesture_score=None):
        """
        Tính điểm hành vi (0-100)
        
        Args:
            eye_contact_score: điểm eye contact (0-100)
            posture_score: điểm tư thế (0-100)
            gesture_score: điểm cử chỉ (0-100)
        
        Returns:
            behavior_score (0-100)
        """
        # Default scores if not provided
        eye_contact_score = eye_contact_score or 70.0
        posture_score = posture_score or 70.0
        gesture_score = gesture_score or 70.0
        
        behavior_score = (
            eye_contact_score * self.BEHAVIOR_WEIGHTS['eye_contact'] +
            posture_score * self.BEHAVIOR_WEIGHTS['posture'] +
            gesture_score * self.BEHAVIOR_WEIGHTS['gestures']
        )
        
        self.scores['behavior'] = {
            'score': behavior_score,
            'breakdown': {
                'eye_contact': eye_contact_score,
                'posture': posture_score,
                'gestures': gesture_score
            }
        }
        
        return behavior_score
    
    def calculate_technical_score(self, video_quality=None, audio_quality=None,
                                  stability=None):
        """
        Tính điểm kỹ thuật (0-100)
        
        Args:
            video_quality: chất lượng video (0-100)
            audio_quality: chất lượng audio (0-100)
            stability: độ ổn định (0-100)
        
        Returns:
            technical_score (0-100)
        """
        # Default scores
        video_quality = video_quality or 80.0
        audio_quality = audio_quality or 80.0
        stability = stability or 80.0
        
        technical_score = (video_quality + audio_quality + stability) / 3
        
        self.scores['technical'] = {
            'score': technical_score,
            'breakdown': {
                'video_quality': video_quality,
                'audio_quality': audio_quality,
                'stability': stability
            }
        }
        
        return technical_score
    
    def calculate_professional_score(self):
        """
        Tính tổng điểm chuyên nghiệp (0-100)
        
        Returns:
            professional_score (0-100)
        """
        if not self.scores:
            return 0.0
        
        professional_score = (
            self.scores.get('emotion', {}).get('score', 0) * self.WEIGHTS['emotion'] +
            self.scores.get('appearance', {}).get('score', 0) * self.WEIGHTS['appearance'] +
            self.scores.get('behavior', {}).get('score', 0) * self.WEIGHTS['behavior'] +
            self.scores.get('technical', {}).get('score', 0) * self.WEIGHTS['technical']
        )
        
        return professional_score
    
    def get_rating(self, score):
        """
        Chuyen diem so thanh rating
        
        Args:
            score: diem so (0-100)
        
        Returns:
            rating string
        """
        if score >= 90:
            return "Xuat sac"
        elif score >= 80:
            return "Tot"
        elif score >= 70:
            return "Kha"
        elif score >= 60:
            return "Trung binh"
        else:
            return "Can cai thien"
    
    def generate_simple_report(self):
        """
        Tạo báo cáo đơn giản - CHỈ TỔNG ĐIỂM
        
        Returns:
            report string
        """
        if not self.scores:
            return "Chưa có dữ liệu phân tích"
        
        professional_score = self.calculate_professional_score()
        rating = self.get_rating(professional_score)
        
        # Simple report - only total score
        report = f"""
==============================================================
          ĐÁNH GIÁ CHUYÊN NGHIỆP           
==============================================================

🎯 TỔNG ĐIỂM: {professional_score:.1f}/100

📊 XẾP LOẠI: {rating}

==============================================================
"""
        return report
    
    def generate_report(self):
        """
        Tạo báo cáo chi tiết
        
        Returns:
            report string
        """
        if not self.scores:
            return "Chưa có dữ liệu phân tích"
        
        professional_score = self.calculate_professional_score()
        rating = self.get_rating(professional_score)
        
        report = f"""
==============================================================
          AFFECTIVA-STYLE PROFESSIONAL ANALYSIS           
==============================================================

TỔNG ĐIỂM CHUYÊN NGHIỆP: {professional_score:.1f}/100 - {rating}

--------------------------------------------------------------

CAM XUC (35%): {self.scores.get('emotion', {}).get('score', 0):.1f}/100
   - Cam xuc chu dao: {self.scores.get('emotion', {}).get('dominant', 'N/A')}
"""
        
        # Emotion breakdown
        if 'emotion' in self.scores and 'breakdown' in self.scores['emotion']:
            for emotion, data in self.scores['emotion']['breakdown'].items():
                report += f"   - {emotion}: {data['percentage']:.1f}%\n"
        
        report += f"""
--------------------------------------------------------------

NGOAI HINH (30%): {self.scores.get('appearance', {}).get('score', 0):.1f}/100
   - Trang phuc: {self.scores.get('appearance', {}).get('breakdown', {}).get('dress_code', 0):.1f}/100
   - Background: {self.scores.get('appearance', {}).get('breakdown', {}).get('background', 0):.1f}/100
   - Anh sang: {self.scores.get('appearance', {}).get('breakdown', {}).get('lighting', 0):.1f}/100

--------------------------------------------------------------

HANH VI (25%): {self.scores.get('behavior', {}).get('score', 0):.1f}/100
   - Eye Contact: {self.scores.get('behavior', {}).get('breakdown', {}).get('eye_contact', 0):.1f}/100
   - Tu the: {self.scores.get('behavior', {}).get('breakdown', {}).get('posture', 0):.1f}/100
   - Cu chi: {self.scores.get('behavior', {}).get('breakdown', {}).get('gestures', 0):.1f}/100

--------------------------------------------------------------

KY THUAT (10%): {self.scores.get('technical', {}).get('score', 0):.1f}/100
   - Video: {self.scores.get('technical', {}).get('breakdown', {}).get('video_quality', 0):.1f}/100
   - Audio: {self.scores.get('technical', {}).get('breakdown', {}).get('audio_quality', 0):.1f}/100
   - On dinh: {self.scores.get('technical', {}).get('breakdown', {}).get('stability', 0):.1f}/100

--------------------------------------------------------------
"""
        
        # Suggestions
        report += self._generate_suggestions(professional_score)
        
        return report
    
    def _generate_suggestions(self, professional_score):
        """
        Tạo gợi ý cải thiện
        
        Args:
            professional_score: điểm tổng
        
        Returns:
            suggestions string
        """
        suggestions = "\nGOI Y CAI THIEN:\n\n"
        
        # Find weakest areas
        weak_areas = []
        
        if 'emotion' in self.scores:
            emotion_score = self.scores['emotion']['score']
            if emotion_score < 70:
                weak_areas.append(('Cam xuc', emotion_score, 
                    "Co gang giu thai do tich cuc va tu tin hon"))
        
        if 'appearance' in self.scores:
            appearance_score = self.scores['appearance']['score']
            if appearance_score < 70:
                weak_areas.append(('Ngoai hinh', appearance_score,
                    "Cai thien trang phuc va background chuyen nghiep hon"))
        
        if 'behavior' in self.scores:
            behavior_score = self.scores['behavior']['score']
            if behavior_score < 70:
                weak_areas.append(('Hanh vi', behavior_score,
                    "Tang eye contact va giu tu the ngoi thang"))
        
        # Sort by score (lowest first)
        weak_areas.sort(key=lambda x: x[1])
        
        if weak_areas:
            for i, (area, score, suggestion) in enumerate(weak_areas[:3], 1):
                suggestions += f"   {i}. {area} ({score:.1f}/100): {suggestion}\n"
        else:
            suggestions += "   Ban da the hien rat tot! Tiep tuc duy tri.\n"
        
        return suggestions
    
    def get_json_report(self):
        """
        Trả về báo cáo dạng JSON
        
        Returns:
            dict
        """
        professional_score = self.calculate_professional_score()
        
        return {
            'professional_score': round(professional_score, 2),
            'rating': self.get_rating(professional_score),
            'breakdown': {
                'emotion': self.scores.get('emotion', {}),
                'appearance': self.scores.get('appearance', {}),
                'behavior': self.scores.get('behavior', {}),
                'technical': self.scores.get('technical', {})
            },
            'weights': self.WEIGHTS
        }
