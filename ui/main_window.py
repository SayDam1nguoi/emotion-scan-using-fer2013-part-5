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

        tk.Label(header_frame, text="🎭", font=("Segoe UI Emoji", 60),
                 bg="#ffffff", fg="#27ae60").pack(pady=(10, 5))
        tk.Label(header_frame, text="Emotion Scanner", font=("Segoe UI", 26, "bold"),
                 bg="#ffffff", fg="#2c3e50").pack()
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

        file_card = tk.Frame(content_frame, bg="#f8f9fa",
                             highlightbackground="#27ae60", highlightthickness=2)
        file_card.pack(fill=tk.X, pady=(0, 15))

        tk.Label(file_card, text="📁  File dữ liệu huấn luyện",
                 font=("Segoe UI", 11, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(
                     anchor="w", padx=18, pady=(15, 8))

        self.lbl_file = tk.Label(file_card, text="⚠️  Chưa chọn file dữ liệu",
                                 font=("Segoe UI", 10), bg="#f8f9fa", fg="#e74c3c")
        self.lbl_file.pack(anchor="w", padx=18, pady=(0, 15))

        choose_btn = tk.Button(content_frame, text="📂  Chọn file FER2013 (.csv)",
                               font=("Segoe UI", 11, "bold"), bg="#3498db", fg="white",
                               relief=tk.FLAT, cursor="hand2", borderwidth=0,
                               activebackground="#2980b9", activeforeground="white",
                               command=self.choose_file)
        choose_btn.pack(fill=tk.X, pady=(0, 20), ipady=14)

        choose_btn.bind("<Enter>", lambda e: choose_btn.configure(bg="#2980b9"))
        choose_btn.bind("<Leave>", lambda e: choose_btn.configure(bg="#3498db"))

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
        
        detect_screen_btn = tk.Button(screen_card, text="💻  Quét từ Màn hình (App bất kỳ)",
                                      font=("Segoe UI", 12, "bold"), bg="#8e44ad", fg="white",
                                      relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                      activebackground="#6c3483", activeforeground="white",
                                      command=self.detect_emotion_screen)
        detect_screen_btn.pack(fill=tk.X, padx=3, pady=3, ipady=14)
        detect_screen_btn.bind("<Enter>", lambda e: detect_screen_btn.configure(bg="#6c3483"))
        detect_screen_btn.bind("<Leave>", lambda e: detect_screen_btn.configure(bg="#8e44ad"))
        
        tk.Label(content_frame, text="💡 YouTube, Netflix, Zoom, phim, app bất kỳ",
                font=("Segoe UI", 8), bg="#ffffff", fg="#7f8c8d").pack(pady=(0, 15))
        
        # 4. Dual Detection Card
        dual_card = tk.Frame(content_frame, bg="#f8f9fa",
                            highlightbackground="#e74c3c", highlightthickness=3)
        dual_card.pack(fill=tk.X, pady=(0, 30))
        
        detect_dual_btn = tk.Button(dual_card, text="👥  Quét CẢ 2 NGƯỜI (Camera + Màn hình)",
                                    font=("Segoe UI", 12, "bold"), bg="#e74c3c", fg="white",
                                    relief=tk.FLAT, cursor="hand2", borderwidth=0,
                                    activebackground="#c0392b", activeforeground="white",
                                    command=self.detect_emotion_dual)
        detect_dual_btn.pack(fill=tk.X, padx=3, pady=3, ipady=14)
        detect_dual_btn.bind("<Enter>", lambda e: detect_dual_btn.configure(bg="#c0392b"))
        detect_dual_btn.bind("<Leave>", lambda e: detect_dual_btn.configure(bg="#e74c3c"))
        
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

        self.root.protocol("WM_DELETE_WINDOW", self.logout)

        # Center window on screen
        self.root.update_idletasks()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def choose_file(self):
        path = filedialog.askopenfilename(title="Chọn file FER2013 (fer2013.csv)",
                                          filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path = path
            filename = os.path.basename(path)
            self.lbl_file.config(text=f"✅  {filename}", fg="#27ae60")
        else:
            self.lbl_file.config(text="⚠️  Chưa chọn file dữ liệu", fg="#e74c3c")

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
            messagebox.showerror("Lỗi", "Vui lòng chọn file FER2013 (.csv) trước!")
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
            messagebox.showerror("Lỗi", "Vui lòng chọn file FER2013 (.csv) trước!")
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
        if not self.csv_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn file FER2013 (.csv)!")
            return
        if not self.video_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn video!")
            return
        
        # Import and start detection (loading window will show inside start_detection)
        from core.detector import start_detection
        
        try:
            start_detection(self.csv_path, video_path=self.video_path)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải video:\n{str(e)}")
    
    def detect_emotion_screen(self):
        """Quét cảm xúc từ screen capture (video call)"""
        if not self.csv_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn file FER2013 (.csv) trước!")
            return
        
        # Show instructions
        msg = ("QUÉT TỪ MÀN HÌNH:\n\n"
               "Quét khuôn mặt từ BẤT KỲ ứng dụng nào:\n"
               "• Video call (Zoom, Teams, Meet)\n"
               "• Video YouTube, Netflix, phim\n"
               "• Ứng dụng khác có khuôn mặt\n\n"
               "HƯỚNG DẪN:\n"
               "1. Mở ứng dụng cần quét\n"
               "2. Nhấn OK để chọn vùng màn hình\n"
               "3. Kéo chuột chọn vùng có khuôn mặt\n"
               "4. Nhấn ENTER xác nhận\n\n"
               "Bạn có muốn tiếp tục?")
        
        result = messagebox.askokcancel("Quét từ Màn hình", msg)
        if not result:
            return
        
        try:
            # Import screen capture
            from core.screen_capture import select_capture_region_interactive
            from core.detector import start_detection_screen
            
            # Let user select region
            messagebox.showinfo("Chọn vùng màn hình", 
                              "Kéo chuột để chọn vùng có khuôn mặt\n"
                              "Có thể là: Video call, YouTube, phim, app bất kỳ\n"
                              "ENTER: Xác nhận | ESC: Hủy")
            
            region = select_capture_region_interactive()
            
            if region is None:
                messagebox.showinfo("Đã hủy", "Đã hủy chọn vùng")
                return
            
            # Start detection with screen capture
            start_detection_screen(self.csv_path, region)
            
        except ImportError:
            messagebox.showerror("Lỗi", 
                               "Thiếu thư viện 'mss'!\n\n"
                               "Cài đặt bằng lệnh:\n"
                               "pip install mss")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể capture màn hình:\n{str(e)}")
    
    def detect_emotion_dual(self):
        """Quét cảm xúc CẢ 2 NGƯỜI - Camera + Screen"""
        if not self.csv_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn file FER2013 (.csv) trước!")
            return
        
        # Show instructions
        msg = ("QUÉT CẢ 2 NGƯỜI TRONG VIDEO CALL:\n\n"
               "📹 Camera: Quét CHÍNH BẠN\n"
               "💻 Screen: Quét NGƯỜI ĐỐI DIỆN\n\n"
               "HƯỚNG DẪN:\n"
               "1. Mở ứng dụng video call\n"
               "2. Bắt đầu cuộc gọi\n"
               "3. Chọn vùng màn hình chứa khuôn mặt người đối diện\n"
               "4. Hệ thống sẽ quét CẢ 2 NGƯỜI đồng thời\n\n"
               "KẾT QUẢ:\n"
               "- So sánh cảm xúc 2 bên\n"
               "- Ai tích cực hơn?\n"
               "- Ai tập trung hơn?\n\n"
               "Bạn có muốn tiếp tục?")
        
        result = messagebox.askokcancel("Quét Cả 2 Người", msg)
        if not result:
            return
        
        try:
            # Import modules
            from core.screen_capture import select_capture_region_interactive
            from core.detector import start_detection_dual
            
            # Let user select region for person 2 (screen)
            messagebox.showinfo("Chọn vùng người đối diện", 
                              "Kéo chuột để chọn vùng khuôn mặt NGƯỜI ĐỐI DIỆN\n"
                              "Nhấn ENTER để xác nhận, ESC để hủy")
            
            region = select_capture_region_interactive()
            
            if region is None:
                messagebox.showinfo("Đã hủy", "Đã hủy chọn vùng")
                return
            
            # Start dual detection
            start_detection_dual(self.csv_path, region)
            
        except ImportError:
            messagebox.showerror("Lỗi", 
                               "Thiếu thư viện 'mss'!\n\n"
                               "Cài đặt bằng lệnh:\n"
                               "pip install mss")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể bắt đầu dual detection:\n{str(e)}")

    def logout(self):
        self.root.destroy()
        self.parent_root.deiconify()


def open_main_window(username, parent_root):
    MainWindow(username, parent_root)
