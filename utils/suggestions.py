# -*- coding: utf-8 -*-
"""
Emotion Suggestions
Đưa ra lời khuyên dựa trên cảm xúc được detect
"""
from core.config import EMOTIONS, NEGATIVE_EMOTIONS


def get_detailed_suggestions(emotion_counts):
    """
    Phân tích dựa trên 2 cảm xúc chính
    
    Args:
        emotion_counts: list of emotion counts
    
    Returns:
        suggestion message string
    """
    total_frames = sum(emotion_counts)
    if total_frames == 0:
        return "Không có dữ liệu"
    
    # Get top 2 emotions
    import numpy as np
    sorted_indices = np.argsort(emotion_counts)[::-1]  # Descending order
    
    top1_idx = sorted_indices[0]
    top1_emotion = EMOTIONS[top1_idx]
    top1_perc = (emotion_counts[top1_idx]/total_frames)*100
    
    top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else top1_idx
    top2_emotion = EMOTIONS[top2_idx]
    top2_perc = (emotion_counts[top2_idx]/total_frames)*100
    
    # Calculate categories
    positive_count = emotion_counts[EMOTIONS.index('Happy')] if 'Happy' in EMOTIONS else 0
    negative_count = sum(emotion_counts[i] for i in range(len(EMOTIONS)) 
                        if EMOTIONS[i] in NEGATIVE_EMOTIONS)
    
    positive_perc = (positive_count/total_frames)*100
    negative_perc = (negative_count/total_frames)*100
    
    # Message based on top 2 emotions
    msg = f"2 CẢM XÚC CHÍNH:\n"
    msg += f"1. {top1_emotion}: {top1_perc:.1f}%\n"
    msg += f"2. {top2_emotion}: {top2_perc:.1f}%\n\n"
    
    # Analysis based on combination
    msg += "PHÂN TÍCH:\n"
    
    # Happy + Neutral = Ideal
    if top1_emotion == 'Happy' and top2_emotion == 'Neutral':
        msg += "Kết hợp lý tưởng!\nTự tin và chuyên nghiệp"
    elif top1_emotion == 'Neutral' and top2_emotion == 'Happy':
        msg += "Rất tốt!\nChuyên nghiệp và tích cực"
    
    # Happy dominant
    elif top1_emotion == 'Happy' and top1_perc > 50:
        msg += "Rất tích cực!\nThể hiện sự tự tin cao"
    
    # Neutral dominant
    elif top1_emotion == 'Neutral' and top1_perc > 50:
        msg += "Ổn định và chuyên nghiệp\nNhưng có thể thêm tích cực"
    
    # Negative emotions present
    elif top1_emotion in NEGATIVE_EMOTIONS or top2_emotion in NEGATIVE_EMOTIONS:
        if negative_perc > 30:
            msg += "Có dấu hiệu stress\nNên nghỉ ngơi và thư giãn"
        else:
            msg += "Bình thường\nMột ít stress là tự nhiên"
    
    # Balanced
    else:
        msg += "Cân bằng tốt\nCảm xúc đa dạng"
    
    return msg
