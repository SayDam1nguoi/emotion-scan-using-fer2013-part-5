import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class MainWindow:
    def __init__(self, username, parent_root):
        self.username = username
        self.parent_root = parent_root
        self.root = tk.Toplevel()
        self.root.title(f"Emotion Scanner - {username}")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set to 90% of screen size for better visibility
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(True, True)  # Allow resizing
        
        # Optional: Start maximized
        # self.root.state('zoomed')  # Windows
        # self.root.attributes('-zoomed', True)  # Linux
        
        # Gradient background (adaptive to window size)
        canvas = tk.Canvas(self.root, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        # Create gradient (green to teal) - will be redrawn on resize
        def draw_gradient(event=None):
            canvas.delete("gradient")
            width = canvas.winfo_width()
            height = canvas.winfo_height()
            if height > 1:  # Ensure valid height
                for i in range(height):
                    ratio = i / height
                    r = int(39 + (26 - 39) * ratio)
                    g = int(174 + (188 - 174) * ratio)
                    b = int(96 + (156 - 96) * ratio)
                    color = f'#{r:02x}{g:02x}{b:02x}'
                    canvas.create_line(0, i, width, i, fill=color, tags="gradient")
        
        canvas.bind("<Configure>", draw_gradient)
        self.root.after(100, draw_gradient)  # Initial draw

        # Use larger frame that adapts to window size
        main_frame = tk.Frame(canvas, bg="#ffffff", relief=tk.FLAT)
        main_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.85, relheight=0.92)

        # Multi-layer shadow (adaptive)
        for offset in range(10, 0, -2):
            shadow = tk.Frame(canvas, bg=f"#{'%02x' % (200 - offset * 4)}{'%02x' % (200 - offset * 4)}{'%02x' % (200 - offset * 4)}")
            shadow.place(relx=0.5, rely=0.5 + offset/1000, anchor="center", 
                        relwidth=0.85 + offset/1000, relheight=0.92 + offset/1000)
            shadow.lower()

        header_frame = tk.Frame(main_frame, bg="#ffffff")
        header_frame.pack(fill=tk.X, pady=(20, 10))

        # XÓA EMOJI MẶT HỀ - chỉ giữ text
        tk.Label(header_frame, text="Emotion Scanner", font=("Segoe UI", 26, "bold"),
                 bg="#ffffff", fg="#2c3e50").pack(pady=(10, 5))
        tk.Label(header_frame, text=f"👋  Xin chào, {username}!", font=("Segoe UI", 13),
                 bg="#ffffff", fg="#7f8c8d").pack(pady=(5, 10))

        # Scrollable content frame
        content_canvas = tk.Canvas(main_frame, bg="#ffffff", highlightthickness=0)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=content_canvas.yview)
        content_frame = tk.Frame(content_canvas, bg="#ffffff")
        
        content_frame.bind(
            "<Configure>",
            lambda e: content_canvas.configure(scrollregion=content_canvas.bbox("all"))
        )
        
        content_canvas.create_window((0, 0), window=content_frame, anchor="nw")
        content_canvas.configure(yscrollcommand=scrollbar.set)
        
        content_canvas.pack(side="left", fill="both", expand=True, padx=60, pady=(5, 30))
        scrollbar.pack(side="right", fill="y")

        # Divider
        tk.Frame(content_frame, bg="#e1e8ed", height=2).pack(fill=tk.X, pady=(0, 20))

        # File card - ẩn đi, chỉ hiển thị status nhẹ
        file_status_frame = tk.Frame(content_frame, bg="#ffffff")
        file_status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_file = tk.Label(file_status_frame, text="",
                                 font=("Segoe UI", 9), bg="#ffffff", fg="#7f8c8d")
        self.lbl_file.pack(anchor="center")

        video_card = tk.Frame(content_frame, bg="#f8f9fa",
                              highlightbackground="#9b59b6", highlightthickness=2)
        video_card.pack(fill=tk.X, pady=(0, 15))

        tk.Label(video_card, text="🎞️  Video để phân tích",
                 font=("Segoe UI", 11, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(
                     anchor="w", padx=18, pady=(15, 8))

        self.lbl_video = tk.Label(video_card, text="ℹ️  Chưa chọn video (tùy chọn)",
                                  font=("Segoe UI", 10), bg="#f8f9fa", fg="#95a5a6")
        self.lbl_video.pack(anchor="w", padx=18, pady=(0, 15))

        video_btn = tk.Button(content_frame, text="🎬  Chọn video (.mp4 / .avi)",
                              font=("Segoe UI", 11, "bold"), bg="#9b59b6", fg="white",
                              relief=tk.FLAT, cursor="hand2", borderwidth=0,
                              activebackground="#8e44ad", activeforeground="white",
                              command=self.choose_video)
        video_btn.pack(fill=tk.X, pady=(0, 25), ipady=14)

        video_btn.bind("<Enter>", lambda e: video_btn.configure(bg="#8e44ad"))
        video_btn.bind("<Leave>", lambda e: video_btn.configure(bg="#9b59b6"))
        
        # Section divider
        tk.Label(content_frame, text="━━━━━━━━━━  Bắt đầu quét  ━━━━━━━━━━",
                 font=("Segoe UI", 9), bg="#ffffff", fg="#bdc3c7").pack(pady=(5, 20))

        # 1. Camera Card
        camera_card = tk.Frame(content_frame, bg="#f8f9fa",
                              highlightbackground="#27ae60", highlightthickness=3)
        camera_card.pack(fill=tk.X, pady=(0, 15))
        
        camera_inner = tk.Frame(camera_card, bg="#f8f9fa")
        camera_inner.pack(fill=tk.X, padx=3, pady=3)
        
        detect_cam_btn = tk.Button(camera_inner, text="📸  Quét cảm xúc qua Camera",
                                   font=("Segoe UI", 12, "bold"), bg="#27ae60", fg="white",
                                   relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                   activebackground="#229954", activeforeground="white",
                                   command=self.detect_emotion_camera)
        detect_cam_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=14)
        detect_cam_btn.bind("<Enter>", lambda e: detect_cam_btn.configure(bg="#229954"))
        detect_cam_btn.bind("<Leave>", lambda e: detect_cam_btn.configure(bg="#27ae60"))
        
        detect_cam_roi_btn = tk.Button(camera_inner, text="🎯",
                                       font=("Segoe UI", 14, "bold"), bg="#229954", fg="white",
                                       relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                       activebackground="#1e8449", activeforeground="white",
                                       command=self.detect_emotion_camera_roi, width=3)
        detect_cam_roi_btn.pack(side=tk.LEFT, padx=(5, 0), ipady=14)
        detect_cam_roi_btn.bind("<Enter>", lambda e: detect_cam_roi_btn.configure(bg="#1e8449"))
        detect_cam_roi_btn.bind("<Leave>", lambda e: detect_cam_roi_btn.configure(bg="#229954"))
        
        select_cam_btn = tk.Button(camera_inner, text="📹",
                                   font=("Segoe UI", 14, "bold"), bg="#16a085", fg="white",
                                   relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                   activebackground="#138d75", activeforeground="white",
                                   command=self.select_camera, width=3)
        select_cam_btn.pack(side=tk.LEFT, padx=(5, 0), ipady=14)
        select_cam_btn.bind("<Enter>", lambda e: select_cam_btn.configure(bg="#138d75"))
        select_cam_btn.bind("<Leave>", lambda e: select_cam_btn.configure(bg="#16a085"))
        
        tk.Label(content_frame, text="💡 🎯 Chọn vùng | 📹 Chọn camera",
                font=("Segoe UI", 8), bg="#ffffff", fg="#7f8c8d").pack(pady=(0, 15))

        # 2. Video Card
        video_detect_card = tk.Frame(content_frame, bg="#f8f9fa",
                                     highlightbackground="#e67e22", highlightthickness=3)
        video_detect_card.pack(fill=tk.X, pady=(0, 15))
        
        detect_video_btn = tk.Button(video_detect_card, text="🎥  Quét cảm xúc từ Video",
                                     font=("Segoe UI", 12, "bold"), bg="#e67e22", fg="white",
                                     relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                     activebackground="#d35400", activeforeground="white",
                                     command=self.detect_emotion_video)
        detect_video_btn.pack(fill=tk.X, padx=3, pady=3, ipady=14)
        detect_video_btn.bind("<Enter>", lambda e: detect_video_btn.configure(bg="#d35400"))
        detect_video_btn.bind("<Leave>", lambda e: detect_video_btn.configure(bg="#e67e22"))
        
        # 3. Screen Capture Card
        screen_card = tk.Frame(content_frame, bg="#f8f9fa",
                              highlightbackground="#8e44ad", highlightthickness=3)
        screen_card.pack(fill=tk.X, pady=(0, 15))
        
        detect_screen_btn = tk.Button(screen_card, text="💻  Quét Toàn Màn hình (Tự động)",
                                      font=("Segoe UI", 12, "bold"), bg="#8e44ad", fg="white",
                                      relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                      activebackground="#6c3483", activeforeground="white",
                                      command=self.detect_emotion_screen)
        detect_screen_btn.pack(fill=tk.X, padx=3, pady=3, ipady=14)
        detect_screen_btn.bind("<Enter>", lambda e: detect_screen_btn.configure(bg="#6c3483"))
        detect_screen_btn.bind("<Leave>", lambda e: detect_screen_btn.configure(bg="#8e44ad"))
        
        tk.Label(content_frame, text="💡 Quét toàn màn hình - Zoom, Teams, YouTube, phim",
                font=("Segoe UI", 8), bg="#ffffff", fg="#7f8c8d").pack(pady=(0, 15))
        
        # 4. Dual Detection Card
        dual_card = tk.Frame(content_frame, bg="#f8f9fa",
                            highlightbackground="#e74c3c", highlightthickness=3)
        dual_card.pack(fill=tk.X, pady=(0, 30))
        
        detect_dual_btn = tk.Button(dual_card, text="👥  Quét CẢ 2 NGƯỜI (Tự động)",
                                    font=("Segoe UI", 12, "bold"), bg="#e74c3c", fg="white",
                                    relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                    activebackground="#c0392b", activeforeground="white",
                                    command=self.detect_emotion_dual)
        detect_dual_btn.pack(fill=tk.X, padx=3, pady=3, ipady=14)
        detect_dual_btn.bind("<Enter>", lambda e: detect_dual_btn.configure(bg="#c0392b"))
        detect_dual_btn.bind("<Leave>", lambda e: detect_dual_btn.configure(bg="#e74c3c"))
        
        tk.Label(content_frame, text="💡 Camera + Toàn màn hình - Tự động tìm 2 khuôn mặt",
                font=("Segoe UI", 8), bg="#ffffff", fg="#7f8c8d").pack(pady=(0, 15))
        
        # Bottom divider
        tk.Frame(content_frame, bg="#e1e8ed", height=2).pack(fill=tk.X, pady=(5, 15))

        logout_btn = tk.Button(content_frame, text="🚪  Đăng xuất",
                               font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#e74c3c",
                               relief=tk.FLAT, cursor="hand2", borderwidth=0,
                               activebackground="#f8f9fa", activeforeground="#c0392b",
                               command=self.logout)
        logout_btn.pack(fill=tk.X, ipady=10)
        
        def on_logout_enter(e):
            logout_btn.configure(bg="#f8f9fa")
        
        def on_logout_leave(e):
            logout_btn.configure(bg="#ffffff")
        
        logout_btn.bind("<Enter>", on_logout_enter)
        logout_btn.bind("<Leave>", on_logout_leave)

        self.csv_path = ""
        self.video_path = ""
        self.camera_id = 0  # Default camera
        
        # Tự động tìm dataset
        self._auto_find_dataset()

        self.root.protocol("WM_DELETE_WINDOW", self.logout)
        
        # THÊM NÚT THOÁT Ở GÓC TRÊN PHẢI
        exit_btn = tk.Button(main_frame, text="✕ Thoát", 
                            font=("Segoe UI", 10, "bold"), 
                            bg="#e74c3c", fg="white",
                            relief=tk.FLAT, cursor="hand2", 
                            borderwidth=0, padx=15, pady=8,
                            command=self.exit_app)
        exit_btn.place(relx=0.98, rely=0.02, anchor="ne")
        
        # Hover effects
        exit_btn.bind("<Enter>", lambda e: exit_btn.configure(bg="#c0392b"))
        exit_btn.bind("<Leave>", lambda e: exit_btn.configure(bg="#e74c3c"))

        # Center window on screen
        self.root.update_idletasks()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def _auto_find_dataset(self):
        """Tự động tìm file dataset"""
        # Tìm dataset trong các vị trí phổ biến (inline để tránh circular import)
        dataset_names = ['fer2013.csv', 'FER2013.csv', 'ckextended.csv', 'CKExtended.csv']
        search_paths = [
            '.',
            './data',
            './datasets',
            '../',
            '../data',
            '../datasets',
            os.path.expanduser('~/Downloads'),
            os.path.expanduser('~/Desktop'),
        ]
        
        dataset_path = None
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            for dataset_name in dataset_names:
                full_path = os.path.join(search_path, dataset_name)
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    dataset_path = os.path.abspath(full_path)
                    break
            if dataset_path:
                break
        
        if dataset_path:
            # Validate dataset
            try:
                with open(dataset_path, 'r') as f:
                    header = f.readline().strip()
                    if 'emotion' not in header.lower() or 'pixels' not in header.lower():
                        self.lbl_file.config(
                            text=f"⚠️  Tìm thấy {os.path.basename(dataset_path)} nhưng không hợp lệ",
                            fg="#e67e22"
                        )
                        return
                    
                    data_line = f.readline().strip()
                    if not data_line:
                        self.lbl_file.config(
                            text=f"⚠️  File {os.path.basename(dataset_path)} rỗng",
                            fg="#e67e22"
                        )
                        return
                
                # Dataset hợp lệ
                self.csv_path = dataset_path
                filename = os.path.basename(dataset_path)
                
                # Xác định loại dataset
                if 'fer2013' in filename.lower():
                    dataset_type = 'FER2013'
                elif 'ck' in filename.lower():
                    dataset_type = 'CK+ Extended'
                else:
                    dataset_type = 'Unknown'
                
                # Cập nhật UI - hiển thị nhẹ
                self.lbl_file.config(
                    text=f"✅ File dữ liệu huấn luyện hợp lệ ({dataset_type})",
                    fg="#27ae60"
                )
            except Exception as e:
                self.lbl_file.config(
                    text=f"⚠️ Lỗi đọc file dữ liệu",
                    fg="#e67e22"
                )
        else:
            # Không tìm thấy dataset
            self.lbl_file.config(
                text="⚠️ Không tìm thấy file dữ liệu huấn luyện",
                fg="#e74c3c"
            )
    
    def choose_file(self):
        """Hidden feature - có thể gọi từ code nhưng không hiển thị button"""
        path = filedialog.askopenfilename(title="Chọn file dữ liệu huấn luyện",
                                          filetypes=[("CSV files", "*.csv")])
        if path:
            # Validate dataset (inline để tránh circular import)
            try:
                with open(path, 'r') as f:
                    header = f.readline().strip()
                    if 'emotion' not in header.lower() or 'pixels' not in header.lower():
                        messagebox.showerror("Lỗi", "File CSV không đúng format (thiếu cột emotion hoặc pixels)")
                        self.lbl_file.config(text="⚠️ File không hợp lệ", fg="#e74c3c")
                        return
                    
                    data_line = f.readline().strip()
                    if not data_line:
                        messagebox.showerror("Lỗi", "File CSV rỗng")
                        self.lbl_file.config(text="⚠️ File không hợp lệ", fg="#e74c3c")
                        return
                
                # Dataset hợp lệ
                self.csv_path = path
                filename = os.path.basename(path)
                
                # Xác định loại dataset
                if 'fer2013' in filename.lower():
                    dataset_type = 'FER2013'
                elif 'ck' in filename.lower():
                    dataset_type = 'CK+ Extended'
                else:
                    dataset_type = 'Unknown'
                
                self.lbl_file.config(
                    text=f"✅ File dữ liệu huấn luyện hợp lệ ({dataset_type})",
                    fg="#27ae60"
                )
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi đọc file:\n{str(e)}")
                self.lbl_file.config(text="⚠️ File không hợp lệ", fg="#e74c3c")

    def choose_video(self):
        path = filedialog.askopenfilename(title="Chọn video",
                                          filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.lbl_video.config(text=f"✅  {os.path.basename(path)}", fg="#9b59b6")
        else:
            self.lbl_video.config(text="ℹ️  Chưa chọn video (tùy chọn)", fg="#95a5a6")

    def select_camera(self):
        """Chọn camera từ danh sách"""
        try:
            from core.camera_selector import select_camera_gui
            
            selected = select_camera_gui()
            if selected is not None:
                self.camera_id = selected
                messagebox.showinfo("Thành công", f"Đã chọn Camera {selected}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể chọn camera:\n{str(e)}")
    
    def detect_emotion_camera(self):
        """Quét toàn bộ camera (không ROI)"""
        if not self.csv_path:
            msg = ("Không tìm thấy file dữ liệu huấn luyện!\n\n"
                   "Vui lòng đặt file fer2013.csv vào thư mục gốc của ứng dụng.\n\n"
                   "File fer2013.csv có thể tải từ:\n"
                   "https://www.kaggle.com/datasets/msambare/fer2013\n\n"
                   "Sau khi tải xong, khởi động lại ứng dụng.")
            messagebox.showerror("Thiếu file dữ liệu", msg)
            return
        
        # Import and start detection (loading window will show inside start_detection)
        from core.detector import start_detection
        
        try:
            start_detection(self.csv_path, camera_id=self.camera_id)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động camera:\n{str(e)}")
    
    def detect_emotion_camera_roi(self):
        """Quét vùng cụ thể trong camera (với ROI)"""
        if not self.csv_path:
            msg = ("Không tìm thấy file dữ liệu huấn luyện!\n\n"
                   "Vui lòng đặt file fer2013.csv vào thư mục gốc của ứng dụng.\n\n"
                   "Sau khi tải xong, khởi động lại ứng dụng.")
            messagebox.showerror("Thiếu file dữ liệu", msg)
            return
        
        # Show instructions
        msg = ("QUÉT VÙNG CỤ THỂ TRONG CAMERA:\n\n"
               "Tính năng này cho phép bạn chọn 1 vùng cụ thể\n"
               "trong khung hình camera để quét.\n\n"
               "HỮU ÍCH KHI:\n"
               "- Có nhiều người trong khung hình\n"
               "- Chỉ muốn quét 1 người cụ thể\n"
               "- Tăng hiệu năng xử lý\n\n"
               "HƯỚNG DẪN:\n"
               "1. Camera sẽ mở\n"
               "2. Kéo chuột để chọn vùng\n"
               "3. Nhấn ENTER để xác nhận\n"
               "4. Nhấn SPACE để chụp lại frame mới\n\n"
               "Bạn có muốn tiếp tục?")
        
        result = messagebox.askokcancel("Quét vùng Camera", msg)
        if not result:
            return
        
        try:
            # Import camera ROI selector
            from core.camera_roi import select_camera_roi
            from core.detector import start_detection_camera_roi
            
            # Let user select ROI
            messagebox.showinfo("Chọn vùng", 
                              "Kéo chuột để chọn vùng cần quét\n"
                              "ENTER: Xác nhận | ESC: Hủy | SPACE: Chụp lại")
            
            roi = select_camera_roi(camera_id=self.camera_id)
            
            print(f"DEBUG UI: ROI returned = {roi}, type = {type(roi)}")
            
            if roi is None:
                messagebox.showinfo("Đã hủy", "Đã hủy chọn vùng")
                return
            
            # Validate ROI format
            if not isinstance(roi, tuple) or len(roi) != 4:
                messagebox.showerror("Lỗi", f"ROI không hợp lệ: type={type(roi)}, value={roi}")
                return
            
            print(f"DEBUG UI: ROI validated, passing to detector")
            
            # Start detection with ROI
            start_detection_camera_roi(self.csv_path, roi, self.camera_id)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động camera ROI:\n{str(e)}")

    def detect_emotion_video(self):
        """Quét video với lựa chọn mode: Nhà tuyển dụng hoặc Ứng viên"""
        if not self.csv_path:
            msg = ("Không tìm thấy file dữ liệu huấn luyện!\n\n"
                   "Vui lòng đặt file fer2013.csv vào thư mục gốc của ứng dụng.\n\n"
                   "Sau khi tải xong, khởi động lại ứng dụng.")
            messagebox.showerror("Thiếu file dữ liệu", msg)
            return
        if not self.video_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn video!")
            return
        
        # Show mode selection dialog
        mode = self._show_video_mode_selection()
        if mode is None:
            return  # User cancelled
        
        # Import and start detection with mode
        from core.detector import start_detection
        
        try:
            start_detection(self.csv_path, video_path=self.video_path, analysis_mode=mode)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải video:\n{str(e)}")
    
    def _show_video_mode_selection(self):
        """Hiển thị dialog chọn mode phân tích video"""
        import tkinter as tk
        
        result = {'mode': None}
        
        # Create dialog - Tăng chiều cao
        dialog = tk.Toplevel(self.root)
        dialog.title("Chọn chế độ phân tích")
        dialog.geometry("600x650")  # Tăng từ 500 lên 650
        dialog.resizable(True, True)  # Cho phép resize
        dialog.configure(bg="#ffffff")
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 300
        y = (dialog.winfo_screenheight() // 2) - 325  # Điều chỉnh y
        dialog.geometry(f"600x650+{x}+{y}")
        
        # Make modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        header_frame = tk.Frame(dialog, bg="#ffffff")
        header_frame.pack(fill=tk.X, pady=(15, 10))
        
        tk.Label(header_frame, text="🎯", font=("Segoe UI Emoji", 35),
                bg="#ffffff").pack()
        tk.Label(header_frame, text="Chọn chế độ phân tích video",
                font=("Segoe UI", 15, "bold"), bg="#ffffff", fg="#2c3e50").pack(pady=(8, 3))
        tk.Label(header_frame, text="Bạn là nhà tuyển dụng hay ứng viên?",
                font=("Segoe UI", 9), bg="#ffffff", fg="#7f8c8d").pack()
        
        # Scrollable content frame
        canvas = tk.Canvas(dialog, bg="#ffffff", highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        content_frame = tk.Frame(canvas, bg="#ffffff")
        
        content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=content_frame, anchor="nw", width=560)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=(0, 10))
        scrollbar.pack(side="right", fill="y", pady=(0, 10))
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def select_mode(mode_value):
            canvas.unbind_all("<MouseWheel>")
            result['mode'] = mode_value
            dialog.destroy()
        
        # Mode 1: Recruiter
        recruiter_card = tk.Frame(content_frame, bg="#f8f9fa",
                                 highlightbackground="#3498db", highlightthickness=2)
        recruiter_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(recruiter_card, text="👔 NHÀ TUYỂN DỤNG",
                font=("Segoe UI", 12, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(
                    anchor="w", padx=20, pady=(15, 5))
        
        tk.Label(recruiter_card, 
                text="Đánh giá video CV của ứng viên\n\n"
                     "✅ Phân tích cảm xúc chuyên nghiệp\n"
                     "✅ Đánh giá trang phục, background\n"
                     "✅ Kiểm tra sự tự tin, tập trung\n"
                     "✅ So sánh giữa các ứng viên",
                font=("Segoe UI", 9), bg="#f8f9fa", fg="#34495e",
                justify=tk.LEFT).pack(anchor="w", padx=20, pady=(0, 10))
        
        recruiter_btn = tk.Button(recruiter_card, text="Chọn chế độ này",
                                 font=("Segoe UI", 10, "bold"), bg="#3498db", fg="white",
                                 relief=tk.FLAT, cursor="hand2",
                                 command=lambda: select_mode('recruiter'))
        recruiter_btn.pack(fill=tk.X, padx=20, pady=(0, 15), ipady=10)
        
        # Mode 2: Recruiter Self-Check
        recruiter_self_card = tk.Frame(content_frame, bg="#f8f9fa",
                                      highlightbackground="#e67e22", highlightthickness=2)
        recruiter_self_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(recruiter_self_card, text="🎬 NHÀ TUYỂN DỤNG (Tự kiểm tra)",
                font=("Segoe UI", 12, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(
                    anchor="w", padx=20, pady=(15, 5))
        
        tk.Label(recruiter_self_card,
                text="Kiểm tra video tuyển dụng của bạn\n\n"
                     "✅ Video có đủ hấp dẫn không?\n"
                     "✅ Cảm xúc có nhiệt tình, chuyên nghiệp?\n"
                     "✅ Có thu hút được ứng viên không?\n"
                     "✅ Gợi ý cải thiện để tăng hiệu quả",
                font=("Segoe UI", 9), bg="#f8f9fa", fg="#34495e",
                justify=tk.LEFT).pack(anchor="w", padx=20, pady=(0, 10))
        
        recruiter_self_btn = tk.Button(recruiter_self_card, text="Chọn chế độ này",
                                       font=("Segoe UI", 10, "bold"), bg="#e67e22", fg="white",
                                       relief=tk.FLAT, cursor="hand2",
                                       command=lambda: select_mode('recruiter_self'))
        recruiter_self_btn.pack(fill=tk.X, padx=20, pady=(0, 15), ipady=10)
        
        # Mode 3: Candidate
        candidate_card = tk.Frame(content_frame, bg="#f8f9fa",
                                 highlightbackground="#27ae60", highlightthickness=2)
        candidate_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(candidate_card, text="🎓 ỨNG VIÊN",
                font=("Segoe UI", 12, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(
                    anchor="w", padx=20, pady=(15, 5))
        
        tk.Label(candidate_card,
                text="Tự kiểm tra video CV của bạn\n\n"
                     "✅ Kiểm tra cảm xúc có phù hợp không\n"
                     "✅ Gợi ý cải thiện trang phục, ánh sáng\n"
                     "✅ Đánh giá độ tự tin, chuyên nghiệp\n"
                     "✅ Lời khuyên để cải thiện video",
                font=("Segoe UI", 9), bg="#f8f9fa", fg="#34495e",
                justify=tk.LEFT).pack(anchor="w", padx=20, pady=(0, 10))
        
        candidate_btn = tk.Button(candidate_card, text="Chọn chế độ này",
                                 font=("Segoe UI", 10, "bold"), bg="#27ae60", fg="white",
                                 relief=tk.FLAT, cursor="hand2",
                                 command=lambda: select_mode('candidate'))
        candidate_btn.pack(fill=tk.X, padx=20, pady=(0, 15), ipady=10)
        
        # Cancel button
        cancel_btn = tk.Button(content_frame, text="Hủy",
                              font=("Segoe UI", 10), bg="#ffffff", fg="#e74c3c",
                              relief=tk.FLAT, cursor="hand2",
                              command=dialog.destroy)
        cancel_btn.pack(pady=(10, 0), ipady=8)
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result['mode']
    
    def detect_emotion_screen(self):
        """Quét cảm xúc từ screen capture (video call) - TOÀN MÀN HÌNH"""
        if not self.csv_path:
            msg = ("Không tìm thấy file dữ liệu huấn luyện!\n\n"
                   "Vui lòng đặt file fer2013.csv vào thư mục gốc của ứng dụng.\n\n"
                   "Sau khi tải xong, khởi động lại ứng dụng.")
            messagebox.showerror("Thiếu file dữ liệu", msg)
            return
        
        # Show mode selection dialog
        mode = self._show_screen_mode_selection()
        if mode is None:
            return  # User cancelled
        
        try:
            # Import screen capture
            from core.detector import start_detection_screen
            import mss
            
            # Get full screen dimensions automatically
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                region = (0, 0, monitor['width'], monitor['height'])
            
            messagebox.showinfo("Bắt đầu", 
                              "Bắt đầu quét TOÀN MÀN HÌNH!\n\n"
                              "- Nhấn 'q' để dừng\n"
                              "- Nhấn 's' để chụp ảnh\n"
                              "- Hệ thống sẽ tự động tìm khuôn mặt")
            
            # Start detection with full screen and mode
            start_detection_screen(self.csv_path, region, analysis_mode=mode)
            
        except ImportError:
            messagebox.showerror("Lỗi", 
                               "Thiếu thư viện 'mss'!\n\n"
                               "Cài đặt bằng lệnh:\n"
                               "pip install mss")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể capture màn hình:\n{str(e)}")
    
    def detect_emotion_dual(self):
        """Quét cảm xúc CẢ 2 NGƯỜI - Camera + Screen (TOÀN MÀN HÌNH)"""
        if not self.csv_path:
            msg = ("Không tìm thấy file dữ liệu huấn luyện!\n\n"
                   "Vui lòng đặt file fer2013.csv vào thư mục gốc của ứng dụng.\n\n"
                   "Sau khi tải xong, khởi động lại ứng dụng.")
            messagebox.showerror("Thiếu file dữ liệu", msg)
            return
        
        # Show instructions
        msg = ("QUÉT CẢ 2 NGƯỜI TRONG VIDEO CALL:\n\n"
               "📹 Camera: Quét CHÍNH BẠN\n"
               "💻 Screen: Quét TOÀN MÀN HÌNH (tìm người đối diện)\n\n"
               "HƯỚNG DẪN:\n"
               "1. Mở ứng dụng video call\n"
               "2. Bắt đầu cuộc gọi\n"
               "3. Nhấn OK để bắt đầu\n"
               "4. Hệ thống sẽ quét CẢ 2 NGƯỜI đồng thời\n\n"
               "KẾT QUẢ:\n"
               "- So sánh cảm xúc 2 bên\n"
               "- Ai tích cực hơn?\n"
               "- Ai tập trung hơn?\n\n"
               "💡 Lưu ý: Quét toàn màn hình, tự động tìm khuôn mặt\n\n"
               "Bạn có muốn tiếp tục?")
        
        result = messagebox.askokcancel("Quét Cả 2 Người", msg)
        if not result:
            return
        
        try:
            # Import modules
            from core.detector import start_detection_dual
            import mss
            
            # Get full screen dimensions automatically
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                region = (0, 0, monitor['width'], monitor['height'])
            
            messagebox.showinfo("Bắt đầu", 
                              "Bắt đầu quét CẢ 2 NGƯỜI!\n\n"
                              "📹 Camera: Quét bạn\n"
                              "💻 Screen: Quét toàn màn hình\n\n"
                              "- Nhấn 'q' để dừng\n"
                              "- Hệ thống tự động tìm khuôn mặt")
            
            # Start dual detection with full screen
            start_detection_dual(self.csv_path, region)
            
        except ImportError:
            messagebox.showerror("Lỗi", 
                               "Thiếu thư viện 'mss'!\n\n"
                               "Cài đặt bằng lệnh:\n"
                               "pip install mss")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể bắt đầu dual detection:\n{str(e)}")

    def _show_screen_mode_selection(self):
        """Hiển thị dialog chọn mode phân tích màn hình"""
        import tkinter as tk
        
        result = {'mode': None}
        
        # Create dialog - Tăng chiều cao
        dialog = tk.Toplevel(self.root)
        dialog.title("Chọn chế độ phân tích")
        dialog.geometry("600x650")  # Tăng từ 500 lên 650
        dialog.resizable(True, True)  # Cho phép resize
        dialog.configure(bg="#ffffff")
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 300
        y = (dialog.winfo_screenheight() // 2) - 325  # Điều chỉnh y
        dialog.geometry(f"600x650+{x}+{y}")
        
        # Make modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Header
        header_frame = tk.Frame(dialog, bg="#ffffff")
        header_frame.pack(fill=tk.X, pady=(15, 10))
        
        tk.Label(header_frame, text="💻", font=("Segoe UI Emoji", 35),
                bg="#ffffff").pack()
        tk.Label(header_frame, text="Chọn chế độ quét màn hình",
                font=("Segoe UI", 15, "bold"), bg="#ffffff", fg="#2c3e50").pack(pady=(8, 3))
        tk.Label(header_frame, text="Bạn đang quét video call hay tự kiểm tra?",
                font=("Segoe UI", 9), bg="#ffffff", fg="#7f8c8d").pack()
        
        # Scrollable content frame
        canvas = tk.Canvas(dialog, bg="#ffffff", highlightthickness=0)
        scrollbar = tk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        content_frame = tk.Frame(canvas, bg="#ffffff")
        
        content_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=content_frame, anchor="nw", width=560)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=(0, 10))
        scrollbar.pack(side="right", fill="y", pady=(0, 10))
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def select_mode(mode_value):
            canvas.unbind_all("<MouseWheel>")
            result['mode'] = mode_value
            dialog.destroy()
        
        # Mode 1: Recruiter (Interview)
        recruiter_card = tk.Frame(content_frame, bg="#f8f9fa",
                                 highlightbackground="#8e44ad", highlightthickness=2)
        recruiter_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(recruiter_card, text="👔 PHỎNG VẤN ONLINE",
                font=("Segoe UI", 12, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(
                    anchor="w", padx=20, pady=(15, 5))
        
        tk.Label(recruiter_card, 
                text="Quét ứng viên trong video call\n\n"
                     "✅ Đánh giá cảm xúc real-time\n"
                     "✅ Kiểm tra sự tự tin, tập trung\n"
                     "✅ Phân tích hành vi, cử chỉ\n"
                     "✅ Báo cáo chuyên nghiệp",
                font=("Segoe UI", 9), bg="#f8f9fa", fg="#34495e",
                justify=tk.LEFT).pack(anchor="w", padx=20, pady=(0, 10))
        
        recruiter_btn = tk.Button(recruiter_card, text="Chọn chế độ này",
                                 font=("Segoe UI", 10, "bold"), bg="#8e44ad", fg="white",
                                 relief=tk.FLAT, cursor="hand2",
                                 command=lambda: select_mode('recruiter'))
        recruiter_btn.pack(fill=tk.X, padx=20, pady=(0, 15), ipady=10)
        
        # Mode 2: Candidate (Practice)
        candidate_card = tk.Frame(content_frame, bg="#f8f9fa",
                                 highlightbackground="#16a085", highlightthickness=2)
        candidate_card.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(candidate_card, text="🎓 TỰ LUYỆN TẬP",
                font=("Segoe UI", 12, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(
                    anchor="w", padx=20, pady=(15, 5))
        
        tk.Label(candidate_card,
                text="Tự kiểm tra trước khi phỏng vấn\n\n"
                     "✅ Luyện tập biểu cảm, cử chỉ\n"
                     "✅ Kiểm tra trang phục, ánh sáng\n"
                     "✅ Đánh giá độ tự tin\n"
                     "✅ Gợi ý cải thiện ngay",
                font=("Segoe UI", 9), bg="#f8f9fa", fg="#34495e",
                justify=tk.LEFT).pack(anchor="w", padx=20, pady=(0, 10))
        
        candidate_btn = tk.Button(candidate_card, text="Chọn chế độ này",
                                 font=("Segoe UI", 10, "bold"), bg="#16a085", fg="white",
                                 relief=tk.FLAT, cursor="hand2",
                                 command=lambda: select_mode('candidate'))
        candidate_btn.pack(fill=tk.X, padx=20, pady=(0, 15), ipady=10)
        
        # Cancel button
        cancel_btn = tk.Button(content_frame, text="Hủy",
                              font=("Segoe UI", 10), bg="#ffffff", fg="#e74c3c",
                              relief=tk.FLAT, cursor="hand2",
                              command=dialog.destroy)
        cancel_btn.pack(pady=(10, 0), ipady=8)
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result['mode']

    def logout(self):
        self.root.destroy()
        self.parent_root.deiconify()
    
    def exit_app(self):
        """Thoát hoàn toàn khỏi ứng dụng"""
        if messagebox.askyesno("Xác nhận thoát", 
                              "Bạn có chắc muốn thoát khỏi ứng dụng?\n\n"
                              "Tất cả cửa sổ sẽ được đóng."):
            # Đóng tất cả cửa sổ và thoát
            self.root.destroy()
            self.parent_root.destroy()
            sys.exit(0)


def open_main_window(username, parent_root):
    MainWindow(username, parent_root)
