# -*- coding: utf-8 -*-
"""
Mode-specific Suggestions
Gợi ý dựa trên mode: Recruiter hoặc Candidate
"""

def get_mode_specific_suggestions(analysis_mode, emotion_counts, lighting_samples, dress_samples, behavior_summary):
    """
    Tạo gợi ý dựa trên mode phân tích
    
    Args:
        analysis_mode: 'recruiter', 'recruiter_self', or 'candidate'
        emotion_counts: list of emotion percentages
        lighting_samples: lighting analysis data
        dress_samples: dress analysis data
        behavior_summary: behavior analysis summary
    
    Returns:
        formatted suggestions string
    """
    from core.config import EMOTIONS
    import numpy as np
    
    if analysis_mode == 'recruiter':
        return _get_recruiter_suggestions(emotion_counts, lighting_samples, dress_samples, behavior_summary)
    elif analysis_mode == 'recruiter_self':
        return _get_recruiter_self_suggestions(emotion_counts, lighting_samples, dress_samples, behavior_summary)
    else:  # candidate
        return _get_candidate_suggestions(emotion_counts, lighting_samples, dress_samples, behavior_summary)


def _get_recruiter_suggestions(emotion_counts, lighting_samples, dress_samples, behavior_summary):
    """Gợi ý cho nhà tuyển dụng - Đánh giá nghiêm ngặt"""
    from core.config import EMOTIONS
    import numpy as np
    
    suggestions = "\n" + "="*60 + "\n"
    suggestions += "📊 ĐÁNH GIÁ CHO NHÀ TUYỂN DỤNG\n"
    suggestions += "="*60 + "\n\n"
    
    # Check if no face detected
    total_emotion = sum(emotion_counts)
    if total_emotion == 0:
        suggestions += "❌ KHÔNG PHÁT HIỆN KHUÔN MẶT!\n"
        suggestions += "="*60 + "\n\n"
        
        # Kiểm tra xem có phải do ánh sáng không
        lighting_issue = False
        avg_brightness = 0
        if lighting_samples:
            avg_brightness = np.mean(lighting_samples)
            # Ánh sáng quá thấp (< 70) hoặc quá cao (> 190)
            if avg_brightness < 70 or avg_brightness > 190:
                lighting_issue = True
        
        if lighting_issue:
            # Nếu do ánh sáng - đánh giá nhẹ nhàng hơn
            suggestions += "⚠️  NGUYÊN NHÂN: VẤN ĐỀ ÁNH SÁNG\n\n"
            suggestions += f"Độ sáng trung bình: {avg_brightness:.0f}/255\n"
            if avg_brightness < 70:
                suggestions += "→ Ánh sáng QUÁ THẤP (< 70)\n\n"
            else:
                suggestions += "→ Ánh sáng QUÁ CAO (> 190)\n\n"
            
            suggestions += "1. Phân tích kỹ thuật:\n"
            suggestions += "   - Đây là vấn đề kỹ thuật, không phải lỗi của ứng viên\n"
            suggestions += "   - Có thể do thiết bị quay hoặc môi trường\n"
            suggestions += "   - Không thể đánh giá cảm xúc và hành vi\n\n"
            
            suggestions += "2. Khuyến nghị:\n"
            suggestions += "   ⚠️  YÊU CẦU ỨNG VIÊN GỬI LẠI VIDEO\n"
            suggestions += "   → Hướng dẫn ứng viên cải thiện ánh sáng:\n"
            if avg_brightness < 70:
                suggestions += "      • Bật thêm đèn trong phòng\n"
                suggestions += "      • Ngồi gần cửa sổ (ánh sáng tự nhiên)\n"
                suggestions += "      • Sử dụng đèn bàn chiếu vào mặt\n"
            else:
                suggestions += "      • Tránh ánh sáng trực tiếp từ phía sau\n"
                suggestions += "      • Đóng rèm hoặc tắt bớt đèn\n"
                suggestions += "      • Điều chỉnh góc camera\n"
            suggestions += "   → Không loại ứng viên chỉ vì vấn đề kỹ thuật này\n"
            suggestions += "   → Cho cơ hội gửi lại video với điều kiện tốt hơn\n\n"
            
            suggestions += "3. Lưu ý:\n"
            suggestions += "   ℹ️  Không thể đánh giá năng lực từ video này\n"
            suggestions += "   → Cần video mới để đánh giá chính xác\n"
            suggestions += "   → Có thể xem xét hồ sơ và kinh nghiệm trước\n\n"
        else:
            # Không phải do ánh sáng - đánh giá nghiêm ngặt hơn
            suggestions += "⚠️  ĐÂY CÓ THỂ LÀ DẤU HIỆU TIÊU CỰC:\n\n"
            suggestions += "1. Video chất lượng kém:\n"
            suggestions += "   - Khuôn mặt không rõ ràng\n"
            suggestions += "   - Camera/thiết bị kém\n"
            suggestions += "   - Góc quay không phù hợp\n\n"
            suggestions += "2. Ứng viên có thể thiếu chuẩn bị:\n"
            suggestions += "   - Không test video trước khi gửi\n"
            suggestions += "   - Thiếu chuyên nghiệp\n"
            suggestions += "   - Không quan tâm đến chất lượng\n\n"
            suggestions += "3. Khuyến nghị:\n"
            suggestions += "   ⚠️  YÊU CẦU ỨNG VIÊN GỬI LẠI VIDEO\n"
            suggestions += "   → Video không đạt yêu cầu tối thiểu\n"
            suggestions += "   → Hướng dẫn ứng viên:\n"
            suggestions += "      • Đảm bảo khuôn mặt rõ ràng\n"
            suggestions += "      • Camera ổn định, góc quay phù hợp\n"
            suggestions += "      • Test video trước khi gửi\n"
            suggestions += "   → Nếu ứng viên gửi lại video kém chất lượng:\n"
            suggestions += "      • Có thể là dấu hiệu thiếu chuyên nghiệp\n"
            suggestions += "      • Cân nhắc loại ứng viên\n\n"
        
        suggestions += "="*60 + "\n"
        return suggestions
    
    # 1. Emotion Analysis
    suggestions += "1️⃣ PHÂN TÍCH CẢM XÚC:\n"
    suggestions += "-" * 40 + "\n"
    
    happy_pct = emotion_counts[EMOTIONS.index('Happy')]
    neutral_pct = emotion_counts[EMOTIONS.index('Neutral')]
    sad_pct = emotion_counts[EMOTIONS.index('Sad')]
    angry_pct = emotion_counts[EMOTIONS.index('Angry')]
    
    positive_total = happy_pct + neutral_pct
    
    if positive_total >= 80:
        suggestions += "✅ Ứng viên thể hiện cảm xúc TÍCH CỰC ({:.1f}%)\n".format(positive_total)
        suggestions += "   → Tự tin, nhiệt tình, phù hợp với môi trường làm việc\n"
    elif positive_total >= 60:
        suggestions += "⚠️  Ứng viên thể hiện cảm xúc KHÁC NHAU ({:.1f}% tích cực)\n".format(positive_total)
        suggestions += "   → Có thể hơi lo lắng, cần đánh giá thêm\n"
    else:
        suggestions += "❌ Ứng viên thể hiện cảm xúc TIÊU CỰC ({:.1f}% tiêu cực)\n".format(100 - positive_total)
        suggestions += "   → Có thể không phù hợp hoặc đang gặp vấn đề\n"
    
    if happy_pct > 40:
        suggestions += "   ✓ Vui vẻ, hòa đồng ({:.1f}%)\n".format(happy_pct)
    if neutral_pct > 40:
        suggestions += "   ✓ Nghiêm túc, chuyên nghiệp ({:.1f}%)\n".format(neutral_pct)
    if sad_pct > 20:
        suggestions += "   ⚠ Có dấu hiệu lo lắng ({:.1f}%)\n".format(sad_pct)
    if angry_pct > 10:
        suggestions += "   ⚠ Có dấu hiệu căng thẳng ({:.1f}%)\n".format(angry_pct)
    
    suggestions += "\n"
    
    # 2. Professional Appearance
    suggestions += "2️⃣ NGOẠI HÌNH CHUYÊN NGHIỆP:\n"
    suggestions += "-" * 40 + "\n"
    
    if dress_samples:
        avg_dress_score = np.mean([s.get('combined_score', s.get('score', 70)) for s in dress_samples])
        if avg_dress_score >= 80:
            suggestions += "✅ Trang phục CHUYÊN NGHIỆP ({:.0f}/100)\n".format(avg_dress_score)
            suggestions += "   → Phù hợp với văn hóa công ty\n"
        elif avg_dress_score >= 60:
            suggestions += "⚠️  Trang phục CHẤP NHẬN ĐƯỢC ({:.0f}/100)\n".format(avg_dress_score)
            suggestions += "   → Có thể cải thiện thêm\n"
        else:
            suggestions += "❌ Trang phục CHƯA PHÙ HỢP ({:.0f}/100)\n".format(avg_dress_score)
            suggestions += "   → Cần lưu ý về dress code\n"
    
    if lighting_samples:
        avg_lighting = np.mean(lighting_samples)
        if 100 <= avg_lighting <= 180:
            suggestions += "✅ Ánh sáng TỐT ({:.0f})\n".format(avg_lighting)
        else:
            suggestions += "⚠️  Ánh sáng CHƯA TỐT ({:.0f})\n".format(avg_lighting)
            suggestions += "   → Ứng viên có thể chưa chuẩn bị kỹ\n"
    
    suggestions += "\n"
    
    # 3. Behavior & Confidence
    suggestions += "3️⃣ HÀNH VI & SỰ TỰ TIN:\n"
    suggestions += "-" * 40 + "\n"
    
    if behavior_summary:
        eye_contact_score = behavior_summary.get('eye_contact', {}).get('score', 70)
        posture_score = behavior_summary.get('posture', {}).get('avg_score', 70)
        
        if eye_contact_score >= 70:
            suggestions += "✅ Eye contact TỐT ({:.0f}/100)\n".format(eye_contact_score)
            suggestions += "   → Tự tin, giao tiếp tốt\n"
        else:
            suggestions += "⚠️  Eye contact YẾU ({:.0f}/100)\n".format(eye_contact_score)
            suggestions += "   → Có thể thiếu tự tin hoặc lo lắng\n"
        
        if posture_score >= 80:
            suggestions += "✅ Tư thế TỐT ({:.0f}/100)\n".format(posture_score)
            suggestions += "   → Tự tin, chuyên nghiệp\n"
        elif posture_score >= 60:
            suggestions += "⚠️  Tư thế TRUNG BÌNH ({:.0f}/100)\n".format(posture_score)
        else:
            suggestions += "❌ Tư thế KÉM ({:.0f}/100)\n".format(posture_score)
            suggestions += "   → Có thể thiếu tự tin\n"
    
    suggestions += "\n"
    
    # 4. Overall Recommendation
    suggestions += "4️⃣ KHUYẾN NGHỊ TỔNG QUAN:\n"
    suggestions += "-" * 40 + "\n"
    
    if positive_total >= 75 and (not dress_samples or avg_dress_score >= 70):
        suggestions += "✅ ỨNG VIÊN PHÙ HỢP\n"
        suggestions += "   → Nên xem xét cho vòng tiếp theo\n"
        suggestions += "   → Cảm xúc tích cực, ngoại hình chuyên nghiệp\n"
    elif positive_total >= 60:
        suggestions += "⚠️  ỨNG VIÊN CÓ TIỀM NĂNG\n"
        suggestions += "   → Cần đánh giá thêm qua phỏng vấn trực tiếp\n"
        suggestions += "   → Một số điểm cần cải thiện\n"
    else:
        suggestions += "❌ ỨNG VIÊN CHƯA PHÙ HỢP\n"
        suggestions += "   → Có thể không phù hợp với vị trí này\n"
        suggestions += "   → Nhiều điểm cần cải thiện\n"
    
    suggestions += "\n" + "="*60 + "\n"
    
    return suggestions


def _get_candidate_suggestions(emotion_counts, lighting_samples, dress_samples, behavior_summary):
    """Gợi ý cho ứng viên"""
    from core.config import EMOTIONS
    import numpy as np
    
    suggestions = "\n" + "="*60 + "\n"
    suggestions += "💡 GỢI Ý CẢI THIỆN CHO ỨNG VIÊN\n"
    suggestions += "="*60 + "\n\n"
    
    # 1. Emotion Improvement
    suggestions += "1️⃣ CẢI THIỆN CẢM XÚC:\n"
    suggestions += "-" * 40 + "\n"
    
    happy_pct = emotion_counts[EMOTIONS.index('Happy')]
    neutral_pct = emotion_counts[EMOTIONS.index('Neutral')]
    sad_pct = emotion_counts[EMOTIONS.index('Sad')]
    angry_pct = emotion_counts[EMOTIONS.index('Angry')]
    
    positive_total = happy_pct + neutral_pct
    
    if positive_total >= 80:
        suggestions += "✅ Cảm xúc của bạn RẤT TỐT! ({:.1f}% tích cực)\n".format(positive_total)
        suggestions += "   → Tiếp tục duy trì thái độ tự tin này\n"
    elif positive_total >= 60:
        suggestions += "⚠️  Cảm xúc của bạn CÒN CẢI THIỆN ({:.1f}% tích cực)\n".format(positive_total)
        suggestions += "   💡 Gợi ý:\n"
        suggestions += "      • Thư giãn trước khi quay video\n"
        suggestions += "      • Mỉm cười tự nhiên hơn\n"
        suggestions += "      • Nghĩ về điều tích cực\n"
    else:
        suggestions += "❌ Cảm xúc của bạn CẦN CẢI THIỆN ({:.1f}% tiêu cực)\n".format(100 - positive_total)
        suggestions += "   💡 Gợi ý QUAN TRỌNG:\n"
        suggestions += "      • Quay lại video khi tâm trạng tốt hơn\n"
        suggestions += "      • Luyện tập trước gương\n"
        suggestions += "      • Tập thở sâu để thư giãn\n"
        suggestions += "      • Tưởng tượng đang nói chuyện với bạn bè\n"
    
    if sad_pct > 20:
        suggestions += "   ⚠ Bạn có vẻ lo lắng ({:.1f}%)\n".format(sad_pct)
        suggestions += "      → Hãy tự tin hơn vào bản thân!\n"
    if angry_pct > 10:
        suggestions += "   ⚠ Bạn có vẻ căng thẳng ({:.1f}%)\n".format(angry_pct)
        suggestions += "      → Thư giãn và quay lại khi sẵn sàng\n"
    
    suggestions += "\n"
    
    # 2. Appearance Improvement
    suggestions += "2️⃣ CẢI THIỆN NGOẠI HÌNH:\n"
    suggestions += "-" * 40 + "\n"
    
    if dress_samples:
        avg_dress_score = np.mean([s.get('combined_score', s.get('score', 70)) for s in dress_samples])
        if avg_dress_score >= 80:
            suggestions += "✅ Trang phục của bạn RẤT CHUYÊN NGHIỆP!\n"
            suggestions += "   → Giữ nguyên phong cách này\n"
        elif avg_dress_score >= 60:
            suggestions += "⚠️  Trang phục CÓ THỂ CẢI THIỆN:\n"
            suggestions += "   💡 Gợi ý:\n"
            suggestions += "      • Chọn màu tối (đen, xanh navy, trắng)\n"
            suggestions += "      • Mặc áo sơ mi hoặc áo vest\n"
            suggestions += "      • Tránh áo thun, áo hoodie\n"
        else:
            suggestions += "❌ Trang phục CHƯA PHÙ HỢP:\n"
            suggestions += "   💡 Gợi ý QUAN TRỌNG:\n"
            suggestions += "      • Mặc áo sơ mi hoặc vest\n"
            suggestions += "      • Chọn màu tối, trang nhã\n"
            suggestions += "      • Tránh màu sặc sỡ (đỏ, cam, vàng)\n"
            suggestions += "      • Quay lại video với trang phục phù hợp\n"
    
    if lighting_samples:
        avg_lighting = np.mean(lighting_samples)
        if 100 <= avg_lighting <= 180:
            suggestions += "✅ Ánh sáng TỐT!\n"
        elif avg_lighting < 100:
            suggestions += "❌ Ánh sáng QUÁ TỐI:\n"
            suggestions += "   💡 Gợi ý:\n"
            suggestions += "      • Mở thêm đèn\n"
            suggestions += "      • Ngồi gần cửa sổ (ánh sáng tự nhiên)\n"
            suggestions += "      • Dùng đèn bàn chiếu vào mặt\n"
        else:
            suggestions += "❌ Ánh sáng QUÁ SÁNG:\n"
            suggestions += "   💡 Gợi ý:\n"
            suggestions += "      • Tắt bớt đèn\n"
            suggestions += "      • Tránh ánh sáng trực tiếp từ phía sau\n"
            suggestions += "      • Dùng rèm che bớt ánh sáng\n"
    
    suggestions += "\n"
    
    # 3. Behavior Improvement
    suggestions += "3️⃣ CẢI THIỆN HÀNH VI:\n"
    suggestions += "-" * 40 + "\n"
    
    if behavior_summary:
        eye_contact_score = behavior_summary.get('eye_contact', {}).get('score', 70)
        posture_score = behavior_summary.get('posture', {}).get('avg_score', 70)
        
        if eye_contact_score >= 70:
            suggestions += "✅ Eye contact TỐT!\n"
        else:
            suggestions += "❌ Eye contact CẦN CẢI THIỆN:\n"
            suggestions += "   💡 Gợi ý:\n"
            suggestions += "      • Nhìn thẳng vào camera (không phải màn hình)\n"
            suggestions += "      • Duy trì >70% thời gian nhìn camera\n"
            suggestions += "      • Tưởng tượng đang nói chuyện với người thật\n"
        
        if posture_score >= 80:
            suggestions += "✅ Tư thế TỐT!\n"
        elif posture_score >= 60:
            suggestions += "⚠️  Tư thế CÓ THỂ CẢI THIỆN:\n"
            suggestions += "   💡 Gợi ý:\n"
            suggestions += "      • Ngồi thẳng lưng\n"
            suggestions += "      • Không gù người\n"
            suggestions += "      • Giữ đầu thẳng, không nghiêng\n"
        else:
            suggestions += "❌ Tư thế CẦN CẢI THIỆN:\n"
            suggestions += "   💡 Gợi ý QUAN TRỌNG:\n"
            suggestions += "      • Ngồi thẳng lưng, vai thả lỏng\n"
            suggestions += "      • Đặt camera ngang tầm mắt\n"
            suggestions += "      • Không gù lưng hoặc cúi đầu\n"
            suggestions += "      • Luyện tập trước gương\n"
    
    suggestions += "\n"
    
    # 4. Action Plan
    suggestions += "4️⃣ KẾ HOẠCH HÀNH ĐỘNG:\n"
    suggestions += "-" * 40 + "\n"
    
    if positive_total >= 75 and (not dress_samples or avg_dress_score >= 70):
        suggestions += "✅ VIDEO CỦA BẠN RẤT TỐT!\n"
        suggestions += "   → Có thể gửi cho nhà tuyển dụng\n"
        suggestions += "   → Chỉ cần kiểm tra lại 1 lần nữa\n"
    elif positive_total >= 60:
        suggestions += "⚠️  VIDEO CÓ THỂ CẢI THIỆN:\n"
        suggestions += "   → Nên quay lại với các cải thiện trên\n"
        suggestions += "   → Luyện tập thêm trước khi gửi\n"
        suggestions += "   → Xem lại video và tự đánh giá\n"
    else:
        suggestions += "❌ NÊN QUAY LẠI VIDEO:\n"
        suggestions += "   → Áp dụng TẤT CẢ gợi ý trên\n"
        suggestions += "   → Luyện tập nhiều lần trước gương\n"
        suggestions += "   → Quay khi tâm trạng tốt, tự tin\n"
        suggestions += "   → Chuẩn bị kỹ: trang phục, ánh sáng, background\n"
    
    suggestions += "\n"
    
    # 5. Quick Checklist
    suggestions += "5️⃣ CHECKLIST TRƯỚC KHI GỬI VIDEO:\n"
    suggestions += "-" * 40 + "\n"
    suggestions += "□ Cảm xúc tích cực (mỉm cười, tự tin)\n"
    suggestions += "□ Trang phục chuyên nghiệp (áo sơ mi/vest, màu tối)\n"
    suggestions += "□ Ánh sáng đủ (100-180, không quá tối/sáng)\n"
    suggestions += "□ Background gọn gàng\n"
    suggestions += "□ Ngồi thẳng lưng\n"
    suggestions += "□ Nhìn thẳng camera >70% thời gian\n"
    suggestions += "□ Nói rõ ràng, tự tin\n"
    suggestions += "□ Đã xem lại video và hài lòng\n"
    
    suggestions += "\n" + "="*60 + "\n"
    suggestions += "💪 Chúc bạn thành công! Hãy tự tin vào bản thân!\n"
    suggestions += "="*60 + "\n"
    
    return suggestions



def _get_recruiter_self_suggestions(emotion_counts, lighting_samples, dress_samples, behavior_summary):
    """Gợi ý cho nhà tuyển dụng tự kiểm tra video tuyển dụng"""
    from core.config import EMOTIONS
    import numpy as np
    
    suggestions = "\n" + "="*60 + "\n"
    suggestions += "🎬 ĐÁNH GIÁ VIDEO TUYỂN DỤNG\n"
    suggestions += "="*60 + "\n\n"
    
    # 1. Attractiveness Analysis
    suggestions += "1️⃣ ĐỘ HẤP DẪN CỦA VIDEO:\n"
    suggestions += "-" * 40 + "\n"
    
    happy_pct = emotion_counts[EMOTIONS.index('Happy')]
    neutral_pct = emotion_counts[EMOTIONS.index('Neutral')]
    sad_pct = emotion_counts[EMOTIONS.index('Sad')]
    angry_pct = emotion_counts[EMOTIONS.index('Angry')]
    
    positive_total = happy_pct + neutral_pct
    
    if positive_total >= 80 and happy_pct >= 40:
        suggestions += "✅ VIDEO RẤT HẤP DẪN! ({:.1f}% tích cực)\n".format(positive_total)
        suggestions += "   → Nhiệt tình, năng động, thu hút ứng viên\n"
        suggestions += "   → Ứng viên sẽ cảm thấy hứng thú với công ty\n"
    elif positive_total >= 60:
        suggestions += "⚠️  VIDEO CÓ THỂ CẢI THIỆN ({:.1f}% tích cực)\n".format(positive_total)
        suggestions += "   💡 Gợi ý:\n"
        suggestions += "      • Thể hiện nhiệt tình hơn\n"
        suggestions += "      • Mỉm cười tự nhiên hơn\n"
        suggestions += "      • Tạo năng lượng tích cực\n"
    else:
        suggestions += "❌ VIDEO CHƯA ĐỦ HẤP DẪN ({:.1f}% tiêu cực)\n".format(100 - positive_total)
        suggestions += "   💡 Gợi ý QUAN TRỌNG:\n"
        suggestions += "      • Quay lại khi tâm trạng tốt hơn\n"
        suggestions += "      • Thể hiện sự nhiệt tình với công việc\n"
        suggestions += "      • Tạo cảm giác tích cực cho ứng viên\n"
        suggestions += "      • Nhấn mạnh cơ hội phát triển\n"
    
    if happy_pct < 30:
        suggestions += "   ⚠ Thiếu sự nhiệt tình ({:.1f}% vui vẻ)\n".format(happy_pct)
        suggestions += "      → Ứng viên có thể không cảm thấy hứng thú\n"
    
    suggestions += "\n"
    
    # 2. Communication Effectiveness (QUAN TRỌNG NHẤT)
    suggestions += "2️⃣ HIỆU QUẢ GIAO TIẾP:\n"
    suggestions += "-" * 40 + "\n"
    
    if behavior_summary:
        eye_contact_score = behavior_summary.get('eye_contact', {}).get('score', 70)
        posture_score = behavior_summary.get('posture', {}).get('avg_score', 70)
        
        if eye_contact_score >= 70:
            suggestions += "✅ Giao tiếp TỰ TIN!\n"
            suggestions += "   → Tạo kết nối tốt với ứng viên\n"
        else:
            suggestions += "❌ Giao tiếp CẦN CẢI THIỆN:\n"
            suggestions += "   💡 Gợi ý:\n"
            suggestions += "      • Nhìn thẳng vào camera\n"
            suggestions += "      • Tạo cảm giác đang nói chuyện trực tiếp\n"
            suggestions += "      • Thể hiện sự chân thành\n"
        
        if posture_score >= 80:
            suggestions += "✅ Tư thế CHUYÊN NGHIỆP!\n"
        else:
            suggestions += "⚠️  Tư thế CẦN CẢI THIỆN:\n"
            suggestions += "   💡 Gợi ý:\n"
            suggestions += "      • Ngồi thẳng, tự tin\n"
            suggestions += "      • Thể hiện sự chuyên nghiệp\n"
    
    suggestions += "\n"
    
    # 3. Content Recommendations
    suggestions += "3️⃣ NỘI DUNG VIDEO NÊN CÓ:\n"
    suggestions += "-" * 40 + "\n"
    suggestions += "✅ Giới thiệu công ty (văn hóa, giá trị)\n"
    suggestions += "✅ Mô tả vị trí tuyển dụng rõ ràng\n"
    suggestions += "✅ Cơ hội phát triển nghề nghiệp\n"
    suggestions += "✅ Phúc lợi và đãi ngộ\n"
    suggestions += "✅ Môi trường làm việc\n"
    suggestions += "✅ Lời kêu gọi hành động (Apply now!)\n"
    
    suggestions += "\n"
    
    # 4. Overall Assessment
    suggestions += "4️⃣ ĐÁNH GIÁ TỔNG QUAN:\n"
    suggestions += "-" * 40 + "\n"
    
    if positive_total >= 75 and happy_pct >= 35:
        suggestions += "✅ VIDEO ĐỦ HẤP DẪN!\n"
        suggestions += "   → Có thể đăng tuyển\n"
        suggestions += "   → Sẽ thu hút được ứng viên chất lượng\n"
        suggestions += "   → Thể hiện được văn hóa công ty\n"
    elif positive_total >= 60:
        suggestions += "⚠️  VIDEO CÓ TIỀM NĂNG:\n"
        suggestions += "   → Nên cải thiện thêm trước khi đăng\n"
        suggestions += "   → Tăng sự nhiệt tình và năng động\n"
        suggestions += "   → Làm nổi bật điểm mạnh của công ty\n"
    else:
        suggestions += "❌ NÊN QUAY LẠI VIDEO:\n"
        suggestions += "   → Video chưa đủ hấp dẫn để thu hút ứng viên\n"
        suggestions += "   → Cần thể hiện nhiệt tình hơn\n"
        suggestions += "   → Tạo năng lượng tích cực\n"
        suggestions += "   → Nhấn mạnh cơ hội và lợi ích cho ứng viên\n"
    
    suggestions += "\n"
    
    # 5. Marketing Tips
    suggestions += "5️⃣ TIPS MARKETING TUYỂN DỤNG:\n"
    suggestions += "-" * 40 + "\n"
    suggestions += "💡 Bắt đầu với hook hấp dẫn (3 giây đầu)\n"
    suggestions += "💡 Nói về lợi ích cho ứng viên (không chỉ yêu cầu)\n"
    suggestions += "💡 Thể hiện văn hóa công ty qua cảm xúc\n"
    suggestions += "💡 Kết thúc với call-to-action rõ ràng\n"
    suggestions += "💡 Độ dài lý tưởng: 1-2 phút\n"
    suggestions += "💡 Thêm subtitle nếu đăng trên mạng xã hội\n"
    
    suggestions += "\n"
    
    # 6. Technical Quality (Optional - chỉ ghi chú nhẹ)
    suggestions += "6️⃣ CHẤT LƯỢNG KỸ THUẬT:\n"
    suggestions += "-" * 40 + "\n"
    
    # Lighting check (không quan trọng lắm)
    if lighting_samples:
        avg_lighting = np.mean(lighting_samples)
        if 100 <= avg_lighting <= 180:
            suggestions += "✅ Ánh sáng tốt\n"
        else:
            suggestions += "ℹ️  Ánh sáng có thể cải thiện (không quan trọng lắm)\n"
    
    # Dress code (không quan trọng lắm)
    if dress_samples:
        avg_dress_score = np.mean([s.get('combined_score', s.get('score', 70)) for s in dress_samples])
        if avg_dress_score >= 70:
            suggestions += "✅ Hình ảnh chuyên nghiệp\n"
        else:
            suggestions += "ℹ️  Có thể chọn trang phục chuyên nghiệp hơn (không bắt buộc)\n"
    
    suggestions += "\n💡 Lưu ý: Với video tuyển dụng, GIAO TIẾP là quan trọng nhất!\n"
    suggestions += "   Trang phục và ánh sáng chỉ là yếu tố phụ.\n"
    
    suggestions += "\n" + "="*60 + "\n"
    suggestions += "🎯 Mục tiêu: Thu hút ứng viên chất lượng cao!\n"
    suggestions += "="*60 + "\n"
    
    return suggestions
