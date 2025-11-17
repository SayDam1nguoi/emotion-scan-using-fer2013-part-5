# -*- coding: utf-8 -*-
"""
Professional Score System
Hệ thống tính điểm chuyên nghiệp tổng hợp
"""
import numpy as np
from datetime import datetime


# ===== Task 4.1: Weighted Scoring System =====

def calculate_professional_score(emotion_score, dress_score, background_score, 
                                 posture_score, eye_contact_score):
    """
    Tính điểm chuyên nghiệp tổng hợp
    
    Weights:
    - Emotion (25%): Happy/Neutral cao = tốt
    - Dress Code (25%): Formal = tốt  
    - Background (20%): Clean + Office = tốt
    - Posture (15%): Thẳng lưng = tốt
    - Eye Contact (15%): >70% = tốt
    
    Args:
        emotion_score: 0-100
        dress_score: 0-100
        background_score: 0-100
        posture_score: 0-100
        eye_contact_score: 0-100
    
    Returns:
        professional_score (0-100)
    """
    weights = {
        'emotion': 0.25,
        'dress': 0.25,
        'background': 0.20,
        'posture': 0.15,
        'eye_contact': 0.15
    }
    
    professional_score = (
        emotion_score * weights['emotion'] +
        dress_score * weights['dress'] +
        background_score * weights['background'] +
        posture_score * weights['posture'] +
        eye_contact_score * weights['eye_contact']
    )
    
    return max(0, min(100, professional_score))


def classify_professional_level(score):
    """
    Phân loại mức độ chuyên nghiệp
    
    Args:
        score: professional score (0-100)
    
    Returns:
        (level, color, description)
    """
    if score >= 85:
        return ("Excellent", (0, 200, 0), "Rat chuyen nghiep!")
    elif score >= 70:
        return ("Good", (0, 255, 0), "Chuyen nghiep")
    elif score >= 55:
        return ("Fair", (0, 165, 255), "Kha on, can cai thien")
    elif score >= 40:
        return ("Poor", (0, 100, 255), "Can cai thien nhieu")
    else:
        return ("Very Poor", (0, 0, 255), "Can cai thien gap!")


# ===== Task 4.2: Detailed Report =====

def generate_detailed_report(emotion_score, dress_score, background_score,
                            posture_score, eye_contact_score, 
                            emotion_counts=None, dress_info=None,
                            background_info=None, behavior_info=None):
    """
    Tạo báo cáo chi tiết
    
    Args:
        *_score: các điểm số
        *_info: thông tin chi tiết (optional)
    
    Returns:
        detailed report dict
    """
    # Calculate professional score
    prof_score = calculate_professional_score(
        emotion_score, dress_score, background_score,
        posture_score, eye_contact_score
    )
    
    level, color, description = classify_professional_level(prof_score)
    
    # Breakdown by category
    breakdown = {
        'emotion': {
            'score': emotion_score,
            'weight': '25%',
            'status': 'Good' if emotion_score >= 70 else 'Needs Improvement',
            'details': emotion_counts or {}
        },
        'dress_code': {
            'score': dress_score,
            'weight': '25%',
            'status': 'Good' if dress_score >= 70 else 'Needs Improvement',
            'details': dress_info or {}
        },
        'background': {
            'score': background_score,
            'weight': '20%',
            'status': 'Good' if background_score >= 70 else 'Needs Improvement',
            'details': background_info or {}
        },
        'posture': {
            'score': posture_score,
            'weight': '15%',
            'status': 'Good' if posture_score >= 70 else 'Needs Improvement',
            'details': behavior_info.get('posture', {}) if behavior_info else {}
        },
        'eye_contact': {
            'score': eye_contact_score,
            'weight': '15%',
            'status': 'Good' if eye_contact_score >= 70 else 'Needs Improvement',
            'details': behavior_info.get('eye_contact', {}) if behavior_info else {}
        }
    }
    
    # Identify strengths and weaknesses
    scores_dict = {
        'Emotion': emotion_score,
        'Dress Code': dress_score,
        'Background': background_score,
        'Posture': posture_score,
        'Eye Contact': eye_contact_score
    }
    
    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    
    strengths = [name for name, score in sorted_scores if score >= 75][:3]
    weaknesses = [name for name, score in sorted_scores if score < 70][:3]
    
    # Benchmark comparison
    benchmark = {
        'average_score': 65.0,
        'top_10_percent': 85.0,
        'top_25_percent': 75.0
    }
    
    if prof_score >= benchmark['top_10_percent']:
        benchmark_status = "Top 10%"
    elif prof_score >= benchmark['top_25_percent']:
        benchmark_status = "Top 25%"
    elif prof_score >= benchmark['average_score']:
        benchmark_status = "Above Average"
    else:
        benchmark_status = "Below Average"
    
    return {
        'professional_score': prof_score,
        'level': level,
        'description': description,
        'breakdown': breakdown,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'benchmark_status': benchmark_status,
        'timestamp': datetime.now().isoformat()
    }


# ===== Task 4.3: Improvement Suggestions =====

def generate_improvement_suggestions(breakdown, behavior_info=None):
    """
    Tạo gợi ý cải thiện (top 3 issues)
    
    Args:
        breakdown: breakdown from detailed report
        behavior_info: additional behavior information
    
    Returns:
        list of prioritized suggestions
    """
    issues = []
    
    # Collect all issues with scores
    for category, info in breakdown.items():
        if info['score'] < 70:
            issues.append({
                'category': category,
                'score': info['score'],
                'priority': 100 - info['score']  # Lower score = higher priority
            })
    
    # Sort by priority
    issues.sort(key=lambda x: x['priority'], reverse=True)
    
    # Generate specific suggestions for top 3
    suggestions = []
    
    for issue in issues[:3]:
        category = issue['category']
        score = issue['score']
        
        if category == 'emotion':
            if score < 40:
                suggestions.append({
                    'priority': 'HIGH',
                    'category': 'Emotion',
                    'issue': 'Cam xuc tieu cuc qua nhieu',
                    'action': 'Nen cuoi nhieu hon, giu thai do tich cuc',
                    'expected_improvement': '+20-30 diem'
                })
            else:
                suggestions.append({
                    'priority': 'MEDIUM',
                    'category': 'Emotion',
                    'issue': 'Cam xuc can cai thien',
                    'action': 'Tang cuoi va giu thai do tu tin',
                    'expected_improvement': '+10-15 diem'
                })
        
        elif category == 'dress_code':
            if score < 50:
                suggestions.append({
                    'priority': 'HIGH',
                    'category': 'Dress Code',
                    'issue': 'Trang phuc khong phu hop',
                    'action': 'Mac ao so mi mau toi (den, xanh navy, trang)',
                    'expected_improvement': '+25-35 diem'
                })
            else:
                suggestions.append({
                    'priority': 'MEDIUM',
                    'category': 'Dress Code',
                    'issue': 'Trang phuc can chuyen nghiep hon',
                    'action': 'Chon trang phuc formal hon',
                    'expected_improvement': '+10-20 diem'
                })
        
        elif category == 'background':
            if score < 50:
                suggestions.append({
                    'priority': 'HIGH',
                    'category': 'Background',
                    'issue': 'Background lon xon hoac khong phu hop',
                    'action': 'Don dep background, loai bo do vat ca nhan. Nen chon phong lam viec.',
                    'expected_improvement': '+20-30 diem'
                })
            else:
                suggestions.append({
                    'priority': 'MEDIUM',
                    'category': 'Background',
                    'issue': 'Background can gon gang hon',
                    'action': 'Sap xep lai background cho gon gang',
                    'expected_improvement': '+10-15 diem'
                })
        
        elif category == 'posture':
            suggestions.append({
                'priority': 'MEDIUM',
                'category': 'Posture',
                'issue': 'Tu the can cai thien',
                'action': 'Ngoi thang lung, nhin thang vao camera',
                'expected_improvement': '+10-15 diem'
            })
        
        elif category == 'eye_contact':
            if score < 50:
                suggestions.append({
                    'priority': 'HIGH',
                    'category': 'Eye Contact',
                    'issue': 'Eye contact rat thap',
                    'action': 'Nhin vao camera it nhat 70% thoi gian',
                    'expected_improvement': '+20-30 diem'
                })
            else:
                suggestions.append({
                    'priority': 'MEDIUM',
                    'category': 'Eye Contact',
                    'issue': 'Eye contact can tang',
                    'action': 'Tang eye contact len >70%',
                    'expected_improvement': '+10-15 diem'
                })
    
    # If no major issues, give general tips
    if not suggestions:
        suggestions.append({
            'priority': 'LOW',
            'category': 'General',
            'issue': 'Khong co van de lon',
            'action': 'Tiep tuc duy tri! Co the fine-tune them cac chi tiet nho.',
            'expected_improvement': '+5-10 diem'
        })
    
    return suggestions


# ===== Task 4.4: Visualization Data =====

def prepare_radar_chart_data(breakdown):
    """
    Chuẩn bị data cho radar chart
    
    Args:
        breakdown: breakdown from detailed report
    
    Returns:
        dict with chart data
    """
    categories = []
    scores = []
    
    for category, info in breakdown.items():
        # Format category name
        if category == 'dress_code':
            name = 'Dress Code'
        elif category == 'eye_contact':
            name = 'Eye Contact'
        else:
            name = category.capitalize()
        
        categories.append(name)
        scores.append(info['score'])
    
    return {
        'categories': categories,
        'scores': scores,
        'max_score': 100
    }


def prepare_timeline_data(emotion_history, behavior_history=None):
    """
    Chuẩn bị data cho timeline chart
    
    Args:
        emotion_history: list of emotion predictions over time
        behavior_history: optional behavior data over time
    
    Returns:
        dict with timeline data
    """
    if not emotion_history:
        return {
            'timestamps': [],
            'emotions': [],
            'behaviors': []
        }
    
    # Sample data points (every N frames to reduce size)
    sample_rate = max(1, len(emotion_history) // 100)  # Max 100 points
    
    sampled_emotions = emotion_history[::sample_rate]
    timestamps = list(range(len(sampled_emotions)))
    
    return {
        'timestamps': timestamps,
        'emotions': sampled_emotions,
        'behaviors': behavior_history[::sample_rate] if behavior_history else []
    }


def generate_comparison_data(current_score, previous_score=None):
    """
    Tạo data cho before/after comparison
    
    Args:
        current_score: current professional score
        previous_score: previous score (if available)
    
    Returns:
        comparison dict
    """
    if previous_score is None:
        return {
            'has_previous': False,
            'current': current_score,
            'improvement': 0,
            'message': 'Lan dau tien phan tich'
        }
    
    improvement = current_score - previous_score
    improvement_pct = (improvement / previous_score) * 100 if previous_score > 0 else 0
    
    if improvement > 0:
        message = f"Cai thien +{improvement:.1f} diem ({improvement_pct:+.1f}%)"
    elif improvement < 0:
        message = f"Giam {improvement:.1f} diem ({improvement_pct:.1f}%)"
    else:
        message = "Khong thay doi"
    
    return {
        'has_previous': True,
        'current': current_score,
        'previous': previous_score,
        'improvement': improvement,
        'improvement_pct': improvement_pct,
        'message': message
    }
