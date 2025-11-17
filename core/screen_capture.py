# -*- coding: utf-8 -*-
"""
Screen Capture for Video Call Analysis
Capture màn hình để phân tích khuôn mặt trong video call
Hỗ trợ: Zoom, Teams, Google Meet, Skype, etc.
"""
import cv2
import numpy as np
import mss
import mss.tools
from PIL import Image


class ScreenCapturer:
    """
    Capture màn hình để phân tích video call
    Thread-safe implementation
    """
    
    def __init__(self, monitor_number=1):
        """
        Args:
            monitor_number: số thứ tự màn hình (1 = primary, 2 = secondary, etc.)
        """
        self.monitor_number = monitor_number
        self.sct = None  # Sẽ được khởi tạo khi cần (thread-safe)
        self.monitor = None
        
        # Region of interest (có thể điều chỉnh)
        self.roi = None
        
        # Khởi tạo monitor info
        self._init_monitor()
    
    def _init_monitor(self):
        """Khởi tạo monitor info (thread-safe)"""
        if self.sct is None:
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[self.monitor_number]
        
    def list_monitors(self):
        """
        Liệt kê tất cả màn hình có sẵn
        
        Returns:
            list of monitor info
        """
        self._init_monitor()  # Đảm bảo sct đã được khởi tạo
        monitors = []
        for i, monitor in enumerate(self.sct.monitors):
            if i == 0:  # Skip "all monitors" entry
                continue
            monitors.append({
                'number': i,
                'width': monitor['width'],
                'height': monitor['height'],
                'left': monitor['left'],
                'top': monitor['top']
            })
        return monitors
    
    def set_roi(self, x, y, width, height):
        """
        Đặt vùng quan tâm (Region of Interest) để capture
        Hữu ích khi chỉ muốn capture cửa sổ video call
        
        Args:
            x, y: tọa độ góc trên bên trái
            width, height: kích thước vùng
        """
        self.roi = {
            'left': self.monitor['left'] + x,
            'top': self.monitor['top'] + y,
            'width': width,
            'height': height
        }
    
    def reset_roi(self):
        """Reset ROI về toàn màn hình"""
        self.roi = None
    
    def capture_frame(self):
        """
        Capture 1 frame từ màn hình (Thread-safe)
        
        Returns:
            numpy array (BGR format) hoặc None nếu lỗi
        """
        try:
            # Đảm bảo sct đã được khởi tạo trong thread hiện tại
            self._init_monitor()
            
            # Chọn vùng capture
            region = self.roi if self.roi else self.monitor
            
            # Capture với error handling tốt hơn
            try:
                screenshot = self.sct.grab(region)
            except AttributeError as e:
                # Nếu gặp lỗi thread-local, tạo lại sct
                print(f"Thread-local error, reinitializing: {e}")
                self.sct = None
                self._init_monitor()
                screenshot = self.sct.grab(region)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img
        except Exception as e:
            print(f"Lỗi capture: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def capture_video_stream(self, callback, fps=10):
        """
        Capture liên tục như video stream
        
        Args:
            callback: function(frame) được gọi mỗi frame
            fps: số frame mỗi giây
        
        Returns:
            None (chạy cho đến khi callback return False)
        """
        import time
        frame_delay = 1.0 / fps
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    break
                
                # Call callback
                should_continue = callback(frame)
                if not should_continue:
                    break
                
                # Maintain FPS
                elapsed = time.time() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                    
        except KeyboardInterrupt:
            print("Dừng capture")
    
    def close(self):
        """Đóng screen capturer"""
        if self.sct is not None:
            try:
                self.sct.close()
            except:
                pass
            self.sct = None


def select_capture_region_interactive():
    """
    Cho phép user chọn vùng capture bằng cách vẽ rectangle
    
    Returns:
        (x, y, width, height) hoặc None nếu cancel
    """
    import tkinter as tk
    from tkinter import messagebox
    
    # Capture full screen first
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Variables for selection
    selecting = False
    start_point = None
    end_point = None
    selected_region = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selecting, start_point, end_point, selected_region
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selecting = True
            start_point = (x, y)
            end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if selecting:
                end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            selecting = False
            end_point = (x, y)
            
            # Calculate region (coordinates are already in original scale)
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])
            
            width = x2 - x1
            height = y2 - y1
            
            if width > 50 and height > 50:  # Minimum size
                # Store as tuple (x, y, width, height)
                selected_region = (int(x1), int(y1), int(width), int(height))
    
    # Show image for selection
    window_name = "Chon vung video call - Keo chuot de chon, nhan ENTER de xac nhan, ESC de huy"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Resize for display
    display_img = img.copy()
    scale = 0.7
    display_img = cv2.resize(display_img, None, fx=scale, fy=scale)
    
    while True:
        temp_img = display_img.copy()
        
        # Draw selection rectangle
        if start_point and end_point:
            pt1 = (int(start_point[0] * scale), int(start_point[1] * scale))
            pt2 = (int(end_point[0] * scale), int(end_point[1] * scale))
            cv2.rectangle(temp_img, pt1, pt2, (0, 255, 0), 2)
            
            # Show dimensions
            width = abs(end_point[0] - start_point[0])
            height = abs(end_point[1] - start_point[1])
            text = f"{width}x{height}"
            cv2.putText(temp_img, text, (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break
        elif key == 27:  # ESC
            selected_region = None
            break
    
    cv2.destroyAllWindows()
    return selected_region


def test_screen_capture():
    """
    Test screen capture với preview
    """
    capturer = ScreenCapturer()
    
    print("Danh sách màn hình:")
    monitors = capturer.list_monitors()
    for m in monitors:
        print(f"  Màn hình {m['number']}: {m['width']}x{m['height']}")
    
    print("\nBắt đầu capture... (nhấn 'q' để thoát)")
    
    def show_frame(frame):
        # Resize for display
        display = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("Screen Capture Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    capturer.capture_video_stream(show_frame, fps=10)
    capturer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test
    test_screen_capture()
