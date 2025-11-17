# -*- coding: utf-8 -*-
"""
Recruiter Notes Module
Cho phép nhà tuyển dụng ghi nhận xét trong quá trình xem video/camera ứng viên
"""
import json
import os
from datetime import datetime


class RecruiterNotes:
    """Quản lý nhận xét của nhà tuyển dụng"""
    
    def __init__(self, candidate_name="Unknown", session_type="video"):
        """
        Initialize recruiter notes
        
        Args:
            candidate_name: Tên ứng viên
            session_type: 'video' hoặc 'camera'
        """
        self.candidate_name = candidate_name
        self.session_type = session_type
        self.notes = []
        self.session_start = datetime.now()
        
    def add_note(self, note_text, timestamp=None, emotion=None):
        """
        Thêm nhận xét mới
        
        Args:
            note_text: Nội dung nhận xét
            timestamp: Thời điểm (giây từ đầu video), None = hiện tại
            emotion: Cảm xúc đang hiển thị (optional)
        """
        if timestamp is None:
            timestamp = (datetime.now() - self.session_start).total_seconds()
        
        note = {
            'timestamp': timestamp,
            'time_formatted': self._format_time(timestamp),
            'note': note_text,
            'emotion': emotion,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.notes.append(note)
        return note
    
    def _format_time(self, seconds):
        """Format thời gian thành MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_all_notes(self):
        """Lấy tất cả nhận xét"""
        return self.notes
    
    def save_to_file(self, filename=None):
        """
        Lưu nhận xét vào file JSON
        
        Args:
            filename: Tên file, None = tự động tạo
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # LƯU VÀO THƯ MỤC test_comment
            filename = f"test_comment/recruiter_notes_{self.candidate_name}_{timestamp}.json"
        
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        data = {
            'candidate_name': self.candidate_name,
            'session_type': self.session_type,
            'session_start': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'total_notes': len(self.notes),
            'notes': self.notes
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def generate_summary(self):
        """Tạo tóm tắt nhận xét"""
        if not self.notes:
            return "Chưa có nhận xét nào."
        
        summary = f"=== NHẬN XÉT CỦA NHÀ TUYỂN DỤNG ===\n\n"
        summary += f"Ứng viên: {self.candidate_name}\n"
        summary += f"Loại: {self.session_type.upper()}\n"
        summary += f"Tổng số nhận xét: {len(self.notes)}\n\n"
        summary += "=" * 60 + "\n\n"
        
        for i, note in enumerate(self.notes, 1):
            summary += f"{i}. [{note['time_formatted']}]"
            if note['emotion']:
                summary += f" (Cảm xúc: {note['emotion']})"
            summary += f"\n   {note['note']}\n\n"
        
        return summary


def create_notes_window(notes_manager, on_note_added=None):
    """
    Tạo cửa sổ nhập nhận xét
    
    Args:
        notes_manager: RecruiterNotes instance
        on_note_added: Callback khi thêm nhận xét mới
    """
    import tkinter as tk
    from tkinter import scrolledtext, messagebox
    
    # Tạo cửa sổ mới
    notes_window = tk.Toplevel()
    notes_window.title("Ghi Nhận Xét - Nhà Tuyển Dụng")
    notes_window.geometry("500x600")
    notes_window.configure(bg="#f0f0f0")
    
    # Header
    header_frame = tk.Frame(notes_window, bg="#2c3e50", height=60)
    header_frame.pack(fill=tk.X)
    header_frame.pack_propagate(False)
    
    tk.Label(header_frame, text="📝 GHI NHẬN XÉT",
            font=("Segoe UI", 16, "bold"), bg="#2c3e50", fg="white").pack(pady=15)
    
    # Info
    info_frame = tk.Frame(notes_window, bg="#f0f0f0")
    info_frame.pack(fill=tk.X, padx=20, pady=10)
    
    tk.Label(info_frame, text=f"Ứng viên: {notes_manager.candidate_name}",
            font=("Segoe UI", 10), bg="#f0f0f0").pack(anchor=tk.W)
    tk.Label(info_frame, text=f"Loại: {notes_manager.session_type.upper()}",
            font=("Segoe UI", 10), bg="#f0f0f0").pack(anchor=tk.W)
    
    # Input area
    input_frame = tk.Frame(notes_window, bg="#f0f0f0")
    input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    tk.Label(input_frame, text="Nhập nhận xét:",
            font=("Segoe UI", 10, "bold"), bg="#f0f0f0").pack(anchor=tk.W)
    
    note_input = scrolledtext.ScrolledText(input_frame, height=5, width=50,
                                          font=("Segoe UI", 10), wrap=tk.WORD)
    note_input.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Buttons
    button_frame = tk.Frame(notes_window, bg="#f0f0f0")
    button_frame.pack(fill=tk.X, padx=20, pady=10)
    
    def add_note():
        note_text = note_input.get("1.0", tk.END).strip()
        if not note_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập nội dung nhận xét!")
            return
        
        # Thêm nhận xét
        note = notes_manager.add_note(note_text)
        
        # Cập nhật danh sách
        update_notes_list()
        
        # Xóa input
        note_input.delete("1.0", tk.END)
        
        # Callback
        if on_note_added:
            on_note_added(note)
        
        messagebox.showinfo("Thành công", "Đã thêm nhận xét!")
    
    tk.Button(button_frame, text="✅ Thêm Nhận Xét", command=add_note,
             font=("Segoe UI", 10, "bold"), bg="#27ae60", fg="white",
             padx=20, pady=10, cursor="hand2").pack(side=tk.LEFT, padx=5)
    
    tk.Button(button_frame, text="💾 Lưu Tất Cả", 
             command=lambda: save_all_notes(notes_manager),
             font=("Segoe UI", 10, "bold"), bg="#3498db", fg="white",
             padx=20, pady=10, cursor="hand2").pack(side=tk.LEFT, padx=5)
    
    # Notes list
    list_frame = tk.Frame(notes_window, bg="#f0f0f0")
    list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    tk.Label(list_frame, text=f"Danh sách nhận xét ({len(notes_manager.notes)}):",
            font=("Segoe UI", 10, "bold"), bg="#f0f0f0").pack(anchor=tk.W)
    
    notes_list = scrolledtext.ScrolledText(list_frame, height=10, width=50,
                                          font=("Segoe UI", 9), wrap=tk.WORD,
                                          state=tk.DISABLED)
    notes_list.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def update_notes_list():
        notes_list.config(state=tk.NORMAL)
        notes_list.delete("1.0", tk.END)
        
        for i, note in enumerate(notes_manager.get_all_notes(), 1):
            text = f"{i}. [{note['time_formatted']}]"
            if note['emotion']:
                text += f" ({note['emotion']})"
            text += f"\n   {note['note']}\n\n"
            notes_list.insert(tk.END, text)
        
        notes_list.config(state=tk.DISABLED)
        
        # Cập nhật số lượng
        list_frame.children['!label'].config(
            text=f"Danh sách nhận xét ({len(notes_manager.notes)}):"
        )
    
    def save_all_notes(manager):
        filename = manager.save_to_file()
        messagebox.showinfo("Đã lưu", f"Đã lưu nhận xét vào:\n{filename}")
    
    # Initial update
    update_notes_list()
    
    return notes_window


# Test function
if __name__ == "__main__":
    import tkinter as tk
    
    # Test
    root = tk.Tk()
    root.withdraw()
    
    notes = RecruiterNotes("Nguyen Van A", "video")
    notes.add_note("Ứng viên trả lời tốt câu hỏi đầu tiên", 30, "Happy")
    notes.add_note("Có vẻ hơi lo lắng khi nói về kinh nghiệm", 120, "Sad")
    
    window = create_notes_window(notes)
    
    root.mainloop()
