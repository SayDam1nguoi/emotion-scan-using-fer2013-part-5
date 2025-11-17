# -*- coding: utf-8 -*-
"""
Visualization
Tạo biểu đồ và charts cho emotion detection results
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from core.config import EMOTIONS, NEGATIVE_EMOTIONS, PATHS

# Configure matplotlib for Vietnamese text
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# Try to use system fonts that support Vietnamese
try:
    import matplotlib.font_manager as fm
    # Try common fonts that support Vietnamese
    for font_name in ['Arial', 'Segoe UI', 'Tahoma', 'Verdana']:
        if any(font_name in f.name for f in fm.fontManager.ttflist):
            matplotlib.rcParams['font.family'] = font_name
            break
except:
    pass  # Use default if font detection fails


def show_scrollable_report(report_text):
    """
    Hiển thị report đầy đủ trong cửa sổ có scrollbar
    
    Args:
        report_text: full report string
    """
    root = tk.Tk()
    root.title("📊 Báo cáo chi tiết - Có thể cuộn xem")
    root.geometry("800x600")
    
    # Create scrolled text widget
    text_area = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        width=100,
        height=35,
        font=("Consolas", 10),
        bg="#FFF8DC",
        fg="#000000"
    )
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Insert report text
    text_area.insert(tk.INSERT, report_text)
    text_area.config(state=tk.DISABLED)  # Make read-only
    
    # Add close button
    btn_close = tk.Button(
        root,
        text="Đóng",
        command=root.destroy,
        font=("Arial", 12),
        bg="#3498db",
        fg="white",
        padx=20,
        pady=5
    )
    btn_close.pack(pady=10)
    
    root.mainloop()


def plot_emotion_chart(emotion_counts, suggestions=None):
    """
    Hiển thị biểu đồ tròn cảm xúc với lời khuyên
    
    Args:
        emotion_counts: list of emotion counts
        suggestions: string with suggestions (optional)
    """
    total_frames = sum(emotion_counts)
    if total_frames == 0:
        return
    
    # Calculate percentages
    percentages = [(count/total_frames)*100 for count in emotion_counts]
    
    # Create figure with larger size for better readability
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle('KET QUA PHAN TICH CAM XUC', fontsize=16, fontweight='bold', y=0.98)
    
    # Define colors
    colors = ['#ff6b6b' if EMOTIONS[i] in NEGATIVE_EMOTIONS 
              else '#51cf66' if EMOTIONS[i]=='Happy' 
              else '#74c0fc'
              for i in range(len(EMOTIONS))]
    
    # Filter non-zero emotions
    non_zero_emotions = [EMOTIONS[i] for i in range(len(EMOTIONS)) if emotion_counts[i] > 0]
    non_zero_counts = [emotion_counts[i] for i in range(len(EMOTIONS)) if emotion_counts[i] > 0]
    non_zero_colors = [colors[i] for i in range(len(EMOTIONS)) if emotion_counts[i] > 0]
    non_zero_percentages = [percentages[i] for i in range(len(EMOTIONS)) if emotion_counts[i] > 0]
    
    # Create grid: pie chart on left (smaller), full report on right (larger)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.4], wspace=0.25)
    
    # Left: Pie Chart (smaller)
    ax_pie = fig.add_subplot(gs[0])
    wedges, texts, autotexts = ax_pie.pie(
        non_zero_counts, 
        labels=non_zero_emotions,
        colors=non_zero_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'},
        explode=[0.05] * len(non_zero_emotions)  # Slight separation
    )
    
    # Style autopct text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Style labels
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    
    ax_pie.set_title('Phan Bo Cam Xuc', fontsize=12, fontweight='bold', pad=15)
    
    # Right: Full Report (Statistics + Affectiva + Lighting)
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    
    # If suggestions provided (full report), use it directly
    if suggestions:
        # Full report already formatted
        display_text = suggestions
    else:
        # Build basic statistics text with top 2 emotions
        display_text = "THONG KE CHI TIET\n" + "="*35 + "\n\n"
        
        # Get top 2 emotions
        sorted_indices = np.argsort(emotion_counts)[::-1]
        
        top1_idx = sorted_indices[0]
        top1_emotion = EMOTIONS[top1_idx]
        top1_pct = percentages[top1_idx]
        
        top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else top1_idx
        top2_emotion = EMOTIONS[top2_idx]
        top2_pct = percentages[top2_idx]
        
        display_text += f"2 CAM XUC CHINH:\n"
        display_text += f"   1. {top1_emotion}: {top1_pct:.1f}%\n"
        display_text += f"   2. {top2_emotion}: {top2_pct:.1f}%\n\n"
        
        # All emotions breakdown
        display_text += "TAT CA CAM XUC:\n"
        for i, (emo, count, pct) in enumerate(zip(EMOTIONS, emotion_counts, percentages)):
            if count > 0:
                marker = ">>>" if i in [top1_idx, top2_idx] else "   "
                display_text += f"{marker} {emo}: {count} frames ({pct:.1f}%)\n"
        
        display_text += f"\nTong frames: {total_frames}\n"
    
    # Display text with scrollable capability
    # Use same font as configured for Vietnamese support
    ax_text.text(0.01, 0.99, display_text, 
                transform=ax_text.transAxes,
                fontsize=7,
                verticalalignment='top',
                fontfamily=matplotlib.rcParams['font.family'],
                wrap=True,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    plt.savefig(PATHS['chart_output'], dpi=150, bbox_inches='tight')
    
    # Show with scrollable window
    plt.show()
    
    # After closing chart, show detailed report in scrollable window
    if suggestions:
        show_scrollable_report(suggestions)
