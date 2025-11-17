# -*- coding: utf-8 -*-
"""
Camera Selector
Cho phép user chọn camera từ danh sách cameras có sẵn
"""
import cv2
import tkinter as tk
from tkinter import messagebox


def list_available_cameras(max_cameras=10):
    """
    Liệt kê tất cả cameras có sẵn
    
    Args:
        max_cameras: số camera tối đa để kiểm tra
    
    Returns:
        list of (camera_id, camera_name) tuples
    """
    available_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm it works
            ret, _ = cap.read()
            if ret:
                # Get camera name/info if possible
                backend = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                name = f"Camera {i} ({backend}) - {width}x{height}"
                available_cameras.append((i, name))
            cap.release()
    
    return available_cameras


def select_camera_gui():
    """
    Hiển thị GUI để user chọn camera
    
    Returns:
        camera_id (int) hoặc None nếu cancel
    """
    # Get available cameras
    cameras = list_available_cameras()
    
    if not cameras:
        messagebox.showerror("Lỗi", "Không tìm thấy camera nào!")
        return None
    
    if len(cameras) == 1:
        # Only one camera, use it directly
        return cameras[0][0]
    
    # Multiple cameras, let user choose
    selected_camera = [None]  # Use list to modify in nested function
    
    def on_select(camera_id):
        selected_camera[0] = camera_id
        root.destroy()
    
    def on_cancel():
        root.destroy()
    
    # Create GUI
    root = tk.Tk()
    root.title("Chọn Camera")
    root.geometry("500x400")
    root.resizable(False, False)
    
    # Title
    tk.Label(root, text="📹 Chọn Camera", 
             font=("Segoe UI", 16, "bold"),
             fg="#2c3e50").pack(pady=20)
    
    tk.Label(root, text="Chọn camera bạn muốn sử dụng:",
             font=("Segoe UI", 10),
             fg="#7f8c8d").pack(pady=10)
    
    # Camera list frame
    list_frame = tk.Frame(root, bg="#ffffff")
    list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Add camera buttons
    for camera_id, camera_name in cameras:
        btn = tk.Button(
            list_frame,
            text=camera_name,
            font=("Segoe UI", 11),
            bg="#3498db",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=lambda cid=camera_id: on_select(cid),
            pady=15
        )
        btn.pack(fill=tk.X, pady=5)
        
        # Hover effects
        def on_enter(e, b=btn):
            b.configure(bg="#2980b9")
        def on_leave(e, b=btn):
            b.configure(bg="#3498db")
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
    
    # Cancel button
    cancel_btn = tk.Button(
        root,
        text="Hủy",
        font=("Segoe UI", 10),
        bg="#e74c3c",
        fg="white",
        relief=tk.FLAT,
        cursor="hand2",
        command=on_cancel,
        pady=10
    )
    cancel_btn.pack(pady=10, padx=20, fill=tk.X)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (500 // 2)
    y = (root.winfo_screenheight() // 2) - (400 // 2)
    root.geometry(f"500x400+{x}+{y}")
    
    root.mainloop()
    
    return selected_camera[0]


if __name__ == "__main__":
    # Test
    print("Available cameras:")
    cameras = list_available_cameras()
    for cam_id, cam_name in cameras:
        print(f"  {cam_id}: {cam_name}")
    
    print("\nOpening camera selector...")
    selected = select_camera_gui()
    if selected is not None:
        print(f"Selected camera: {selected}")
    else:
        print("No camera selected")
