# -*- coding: utf-8 -*-
"""
File Operations Utilities
Xử lý lưu/đọc files
"""
import os
import csv
import time
from core.config import PATHS


def save_results(emotion_counts, final_emotion):
    """
    Lưu kết quả vào CSV
    
    Args:
        emotion_counts: list of emotion counts
        final_emotion: dominant emotion name
    """
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    path = PATHS['results_csv']
    
    with open(path, mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), final_emotion, emotion_counts])
