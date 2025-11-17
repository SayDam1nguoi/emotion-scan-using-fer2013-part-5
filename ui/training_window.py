# -*- coding: utf-8 -*-
"""
Training Progress Window
Hiển thị tiến trình training model
"""
import tkinter as tk
from tkinter import Tk, Toplevel, Label, ttk
import time
import tensorflow as tf


def show_loading_window(total_epochs):
    """
    Hiển thị window training progress
    
    Args:
        total_epochs: total number of epochs
    
    Returns:
        (root, loading_win, progress, epoch_label, status_label, time_label)
    """
    root = Tk()
    root.withdraw()
    
    loading_win = Toplevel()
    loading_win.title("🤖 Đang Huấn Luyện AI")
    loading_win.geometry("500x280")
    loading_win.resizable(False, False)
    loading_win.configure(bg="#f0f4f8")
    
    # Center window
    loading_win.update_idletasks()
    x = (loading_win.winfo_screenwidth() // 2) - (500 // 2)
    y = (loading_win.winfo_screenheight() // 2) - (280 // 2)
    loading_win.geometry(f"500x280+{x}+{y}")
    
    # Main frame
    main_frame = tk.Frame(loading_win, bg="#ffffff", relief=tk.FLAT)
    main_frame.place(relx=0.5, rely=0.5, anchor="center", width=460, height=240)
    
    # Shadow
    shadow_frame = tk.Frame(loading_win, bg="#d1d9e0")
    shadow_frame.place(relx=0.5, rely=0.505, anchor="center", width=465, height=245)
    shadow_frame.lower()
    
    # Header
    header_frame = tk.Frame(main_frame, bg="#4A90E2", width=460, height=70)
    header_frame.pack()
    header_frame.pack_propagate(False)
    
    tk.Label(header_frame, text="🤖", font=("Segoe UI Emoji", 32), 
             bg="#4A90E2", fg="white").pack(pady=5)
    tk.Label(header_frame, text="Đang Huấn Luyện AI", font=("Segoe UI", 12, "bold"),
             bg="#4A90E2", fg="white").pack()
    
    # Content
    content_frame = tk.Frame(main_frame, bg="#ffffff")
    content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
    
    # Status label
    status_label = tk.Label(content_frame, text="⏳ Đang khởi tạo model...", 
                           font=("Segoe UI", 10), bg="#ffffff", fg="#2c3e50")
    status_label.pack(pady=(0, 10))
    
    # Progress bar
    progress_frame = tk.Frame(content_frame, bg="#ffffff")
    progress_frame.pack(fill=tk.X, pady=(0, 10))
    
    progress = ttk.Progressbar(progress_frame, maximum=total_epochs, length=400, mode='determinate')
    progress.pack()
    
    # Epoch label
    epoch_label = tk.Label(content_frame, text=f"Epoch: 0 / {total_epochs} (0%)", 
                          font=("Segoe UI", 11, "bold"), bg="#ffffff", fg="#4A90E2")
    epoch_label.pack(pady=(0, 5))
    
    # Time estimate
    time_label = tk.Label(content_frame, text="Thời gian ước tính: Đang tính...", 
                         font=("Segoe UI", 9), bg="#ffffff", fg="#7f8c8d")
    time_label.pack()
    
    # Info label
    info_label = tk.Label(content_frame, text="💡 Vui lòng không tắt ứng dụng", 
                         font=("Segoe UI", 9, "italic"), bg="#ffffff", fg="#95a5a6")
    info_label.pack(pady=(10, 0))
    
    return root, loading_win, progress, epoch_label, status_label, time_label


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """
    Callback để update training progress UI
    """
    def __init__(self, progress, epoch_label, status_label, time_label, total_epochs, root):
        super().__init__()
        self.progress = progress
        self.epoch_label = epoch_label
        self.status_label = status_label
        self.time_label = time_label
        self.total_epochs = total_epochs
        self.root = root
        self.start_time = time.time()
        self.epoch_times = []

    def on_epoch_end(self, epoch, logs=None):
        """Update UI sau mỗi epoch"""
        # Update progress
        self.progress['value'] = epoch + 1
        percentage = ((epoch + 1) / self.total_epochs) * 100
        
        # Calculate time
        elapsed = time.time() - self.start_time
        self.epoch_times.append(elapsed / (epoch + 1))
        avg_time_per_epoch = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        estimated_remaining = remaining_epochs * avg_time_per_epoch
        
        # Format time
        if estimated_remaining < 60:
            time_str = f"{int(estimated_remaining)} giây"
        else:
            minutes = int(estimated_remaining // 60)
            seconds = int(estimated_remaining % 60)
            time_str = f"{minutes} phút {seconds} giây"
        
        # Update labels
        self.epoch_label.config(text=f"Epoch: {epoch+1} / {self.total_epochs} ({percentage:.1f}%)")
        self.status_label.config(text=f"✅ Đang training... Accuracy: {logs.get('accuracy', 0):.2%}")
        self.time_label.config(text=f"Thời gian còn lại: ~{time_str}")
        
        self.root.update()
