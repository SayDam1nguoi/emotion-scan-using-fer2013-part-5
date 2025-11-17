# -*- coding: utf-8 -*-
"""
Interview Suggestions
Goi y cho 2 loai: Live Interview va CV Video
"""
from core.config import EMOTIONS, NEGATIVE_EMOTIONS


def get_live_interview_suggestions(emotion_counts, lighting_samples, dress_samples):
    """
    Gợi ý cho phỏng vấn online (real-time)
    Focus: Hành vi hiện tại, cần chỉnh ngay
    
    Args:
        emotion_counts: list of emotion counts
        lighting_samples: lighting data
        dress_samples: dress analysis data
    
    Returns:
        suggestions string
    """
    total_frames = sum(emotion_counts)
    if total_frames == 0:
        return "Không có dữ liệu"
    
    suggestions = "GỢI Ý CHO PHỎNG VẤN ONLINE:\n"
    suggestions += "="*40 + "\n\n"
    
    # Emotion analysis
    happy_pct = (emotion_counts[EMOTIONS.index('Happy')] / total_frames) * 100
    sad_pct = (emotion_counts[EMOTIONS.index('Sad')] / total_frames) * 100
    angry_pct = (emotion_counts[EMOTIONS.index('Angry')] / total_frames) * 100
    neutral_pct = (emotion_counts[EMOTIONS.index('Neutral')] / total_frames) * 100
    
    suggestions += "1. KHUÔN MẶT:\n"
    
    if happy_pct < 30:
        suggestions += "   - Quá ít cười! Hãy tươi cười tự nhiên hơn\n"
        suggestions += "   - Giữ thái độ tích cực, tự tin\n"
    elif happy_pct > 70:
        suggestions += "   - Cười quá nhiều! Có thể không tự nhiên\n"
        suggestions += "   - Cân bằng giữa tươi cười và nghiêm túc\n"
    else:
        suggestions += "   - Biểu cảm tốt! Tiếp tục duy trì\n"
    
    if sad_pct > 20:
        suggestions += "   - Tránh trạng thái u sầu\n"
        suggestions += "   - Giữ tâm trạng thoải mái\n"
    
    if angry_pct > 15:
        suggestions += "   - Giữ bình tĩnh, tránh lộ cảm xúc\n"
    
    suggestions += "\n"
    
    # Lighting (real-time fix)
    if lighting_samples:
        import numpy as np
        avg_brightness = np.mean(lighting_samples)
        
        suggestions += "2. ÁNH SÁNG:\n"
        if avg_brightness < 80:
            suggestions += "   - Bật thêm đèn NGAY!\n"
            suggestions += "   - Di chuyển gần cửa sổ\n"
        elif avg_brightness > 180:
            suggestions += "   - Giảm ánh sáng lại\n"
            suggestions += "   - Đóng rèm hoặc thay đổi góc camera\n"
        else:
            suggestions += "   - Ánh sáng tốt\n"
        suggestions += "\n"
    
    # Dress (can't change now, just note)
    if dress_samples:
        from core.dress_analysis import generate_dress_report
        dress_report = generate_dress_report(dress_samples)
        
        suggestions += "3. TRANG PHỤC:\n"
        if dress_report['score'] < 70:
            suggestions += "   - Lần sau nên mặc trang phục chính thống hơn\n"
        else:
            suggestions += "   - Trang phục phù hợp\n"
        suggestions += "\n"
    
    # Note about attention tracking
    suggestions += "LƯU Ý:\n"
    suggestions += "   - Sự tập trung vào màn hình đang được theo dõi\n"
    suggestions += "   - Nhìn thẳng vào camera để thể hiện sự chú ý\n"
    suggestions += "   - Tránh nhìn lung tung hoặc quay đầu đi\n"
    
    return suggestions


def get_cv_video_suggestions(emotion_counts, lighting_samples, dress_samples):
    """
    Gợi ý cho video CV (post-production)
    Focus: Chỉnh sửa video, thu hút nhà tuyển dụng
    
    Args:
        emotion_counts: list of emotion counts
        lighting_samples: lighting data
        dress_samples: dress analysis data
    
    Returns:
        suggestions string
    """
    total_frames = sum(emotion_counts)
    if total_frames == 0:
        return "Không có dữ liệu"
    
    suggestions = "GỢI Ý CHỈNH SỬA VIDEO CV:\n"
    suggestions += "="*40 + "\n\n"
    
    # Emotion analysis for video
    happy_pct = (emotion_counts[EMOTIONS.index('Happy')] / total_frames) * 100
    sad_pct = (emotion_counts[EMOTIONS.index('Sad')] / total_frames) * 100
    neutral_pct = (emotion_counts[EMOTIONS.index('Neutral')] / total_frames) * 100
    
    suggestions += "NỘI DUNG VIDEO:\n"
    
    if happy_pct < 40:
        suggestions += "- QUAY LẠI: Cần thêm cảnh tươi cười, tự tin\n"
        suggestions += "- Thêm phần giới thiệu bản thân với năng lượng cao\n"
    elif happy_pct > 65:
        suggestions += "- CẮT BỚT: Giảm cảnh cười quá nhiều\n"
        suggestions += "- Cân bằng với phần nghiêm túc, chuyên nghiệp\n"
    else:
        suggestions += "- Tỷ lệ cảm xúc tốt! Giữ nguyên\n"
    
    if sad_pct > 15:
        suggestions += "- CẮT BỎ: Loại các cảnh trạng thái tiêu cực\n"
        suggestions += "- Thay bằng cảnh tích cực, năng động\n"
    
    suggestions += "\n"
    
    # Lighting (post-production)
    if lighting_samples:
        import numpy as np
        avg_brightness = np.mean(lighting_samples)
        
        suggestions += "CHỈNH SỬA ÁNH SÁNG:\n"
        if avg_brightness < 80:
            suggestions += "- TĂNG ĐỘ SÁNG: Dùng phần mềm chỉnh màu\n"
            suggestions += "- Brightness +20-30%\n"
            suggestions += "- Hoặc QUAY LẠI với ánh sáng tốt hơn\n"
        elif avg_brightness > 180:
            suggestions += "- GIẢM ĐỘ SÁNG: Exposure -10-20%\n"
            suggestions += "- Điều chỉnh Highlights\n"
        else:
            suggestions += "- Ánh sáng tốt, không cần chỉnh\n"
        
        suggestions += "- Thêm color grading để video chuyên nghiệp hơn\n"
        suggestions += "\n"
    
    # Dress
    if dress_samples:
        from core.dress_analysis import generate_dress_report
        dress_report = generate_dress_report(dress_samples)
        
        suggestions += "TRANG PHỤC:\n"
        if dress_report['score'] < 70:
            suggestions += "- QUAY LẠI: Mặc trang phục chính thống\n"
            suggestions += f"- Gợi ý: {', '.join(dress_report['suggestions'][:2])}\n"
        else:
            suggestions += "- Trang phục tốt, giữ nguyên\n"
        suggestions += "\n"
    
    # Video production tips
    suggestions += "TIPS CHUNG:\n"
    suggestions += "- Thêm intro/outro chuyên nghiệp\n"
    suggestions += "- Thêm subtitle nếu cần\n"
    suggestions += "- Nên video: 1-2 phút là lý tưởng\n"
    suggestions += "- Xuất video: 1080p, format MP4\n"
    
    return suggestions


def get_suggestions_by_mode(mode, emotion_counts, lighting_samples, dress_samples):
    """
    Lấy suggestions theo mode
    
    Args:
        mode: 'live' hoặc 'video'
        emotion_counts: emotion data
        lighting_samples: lighting data
        dress_samples: dress data
    
    Returns:
        suggestions string
    """
    if mode == 'live':
        return get_live_interview_suggestions(emotion_counts, lighting_samples, dress_samples)
    else:
        return get_cv_video_suggestions(emotion_counts, lighting_samples, dress_samples)
