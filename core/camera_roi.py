# -*- coding: utf-8 -*-
"""
Camera ROI Selection
Cho phép chọn vùng quan tâm (ROI) trong camera để quét
"""
import cv2
import numpy as np


def select_camera_roi(camera_id=0):
    """
    Cho phép user chọn vùng ROI từ camera bằng cách kéo chuột
    
    Args:
        camera_id: ID của camera
    
    Returns:
        (x, y, width, height) hoặc None nếu cancel
    """
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Không thể mở camera!")
        return None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get first frame
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc frame từ camera!")
        cap.release()
        return None
    
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
            
            # Calculate region
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])
            
            width = x2 - x1
            height = y2 - y1
            
            if width > 50 and height > 50:  # Minimum size
                selected_region = (x1, y1, width, height)
    
    # Setup window
    window_name = "Chon vung quet - Keo chuot de chon, ENTER xac nhan, ESC huy, SPACE chup lai"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("Hướng dẫn:")
    print("- Kéo chuột để chọn vùng cần quét")
    print("- Nhấn ENTER để xác nhận")
    print("- Nhấn ESC để hủy")
    print("- Nhấn SPACE để chụp lại frame mới")
    
    current_frame = frame.copy()
    
    while True:
        display_frame = current_frame.copy()
        
        # Draw selection rectangle
        if start_point and end_point:
            pt1 = start_point
            pt2 = end_point
            cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
            
            # Show dimensions
            width = abs(end_point[0] - start_point[0])
            height = abs(end_point[1] - start_point[1])
            text = f"{width}x{height}"
            cv2.putText(display_frame, text, (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show selected region if confirmed
        if selected_region:
            x, y, w, h = selected_region
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display_frame, "DA CHON - Nhan ENTER", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(display_frame, "Keo chuot de chon vung", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "ENTER: Xac nhan | ESC: Huy | SPACE: Chup lai", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            if selected_region:
                break
        elif key == 27:  # ESC
            selected_region = None
            break
        elif key == 32:  # Space - capture new frame
            ret, current_frame = cap.read()
            if ret:
                start_point = None
                end_point = None
                selected_region = None
                print("Đã chụp frame mới!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return selected_region


def apply_roi_to_frame(frame, roi):
    """
    Áp dụng ROI vào frame
    
    Args:
        frame: full frame
        roi: (x, y, width, height)
    
    Returns:
        cropped frame
    """
    if roi is None:
        return frame
    
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def draw_roi_on_frame(frame, roi, color=(0, 255, 0), thickness=2):
    """
    Vẽ ROI lên frame
    
    Args:
        frame: frame to draw on
        roi: (x, y, width, height)
        color: BGR color
        thickness: line thickness
    
    Returns:
        frame with ROI drawn
    """
    if roi is None:
        return frame
    
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    cv2.putText(frame, "ROI", (x, y - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame


if __name__ == "__main__":
    # Test
    print("Testing camera ROI selection...")
    roi = select_camera_roi()
    
    if roi:
        print(f"Selected ROI: {roi}")
        
        # Test with live camera
        cap = cv2.VideoCapture(0)
        print("Showing ROI in live camera (press 'q' to quit)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw ROI
            frame_with_roi = draw_roi_on_frame(frame.copy(), roi)
            
            # Show cropped region
            cropped = apply_roi_to_frame(frame, roi)
            
            cv2.imshow("Full Frame with ROI", frame_with_roi)
            cv2.imshow("Cropped ROI", cropped)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("No ROI selected")
