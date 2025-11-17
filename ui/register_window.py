# ui/register_window.py
import os
import sys
import tkinter as tk
from tkinter import messagebox

# Cho phép import module users
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from users.user_manager import register_user

def open_register_window(back_to_login_callback=None):
    """Giao diện đăng ký người dùng mới hiện đại."""
    reg = tk.Tk()
    reg.title("Emotion Scanner - Đăng ký")
    reg.geometry("480x700")
    reg.resizable(False, False)
    
    # Gradient background
    canvas = tk.Canvas(reg, width=480, height=700, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    
    # Create gradient (purple to blue)
    for i in range(700):
        ratio = i / 700
        r = int(142 + (74 - 142) * ratio)
        g = int(68 + (144 - 68) * ratio)
        b = int(173 + (226 - 173) * ratio)
        color = f'#{r:02x}{g:02x}{b:02x}'
        canvas.create_line(0, i, 480, i, fill=color)

    # Frame chính với shadow
    main_frame = tk.Frame(canvas, bg="#ffffff", relief=tk.FLAT)
    main_frame.place(relx=0.5, rely=0.5, anchor="center", width=400, height=620)

    # Multi-layer shadow
    for offset in range(8, 0, -2):
        shadow = tk.Frame(canvas, bg=f"#{'%02x' % (200 - offset * 5)}{'%02x' % (200 - offset * 5)}{'%02x' % (200 - offset * 5)}")
        shadow.place(relx=0.5, rely=0.5 + offset/1000, anchor="center", 
                    width=400 + offset, height=620 + offset)
        shadow.lower()

    # Header area
    header_frame = tk.Frame(main_frame, bg="#ffffff", width=400, height=120)
    header_frame.pack()
    header_frame.pack_propagate(False)

    tk.Label(header_frame, text="✨", font=("Segoe UI Emoji", 48), 
             bg="#ffffff", fg="#8e44ad").pack(pady=(20, 5))
    tk.Label(header_frame, text="Tạo tài khoản mới", font=("Segoe UI", 18, "bold"), 
             bg="#ffffff", fg="#2c3e50").pack()

    # Content frame
    content_frame = tk.Frame(main_frame, bg="#ffffff")
    content_frame.pack(fill=tk.BOTH, expand=True, padx=45, pady=(10, 30))

    # Username field
    tk.Label(content_frame, text="👤  Tên đăng nhập", font=("Segoe UI", 10, "bold"), 
             bg="#ffffff", fg="#5a6c7d", anchor="w").pack(fill=tk.X, pady=(0, 8))
    
    user_frame = tk.Frame(content_frame, bg="#f8f9fa", highlightbackground="#e1e8ed", 
                         highlightthickness=2, highlightcolor="#8e44ad")
    user_frame.pack(fill=tk.X, pady=(0, 18))
    
    entry_user = tk.Entry(user_frame, font=("Segoe UI", 11), relief=tk.FLAT, 
                          bg="#f8f9fa", fg="#2c3e50", insertbackground="#8e44ad")
    entry_user.pack(fill=tk.BOTH, padx=15, pady=12)

    # Password field
    tk.Label(content_frame, text="🔒  Mật khẩu", font=("Segoe UI", 10, "bold"), 
             bg="#ffffff", fg="#5a6c7d", anchor="w").pack(fill=tk.X, pady=(0, 8))
    
    pass_frame = tk.Frame(content_frame, bg="#f8f9fa", highlightbackground="#e1e8ed", 
                         highlightthickness=2, highlightcolor="#8e44ad")
    pass_frame.pack(fill=tk.X, pady=(0, 18))
    
    entry_pass = tk.Entry(pass_frame, show="●", font=("Segoe UI", 11), relief=tk.FLAT, 
                          bg="#f8f9fa", fg="#2c3e50", insertbackground="#8e44ad")
    entry_pass.pack(fill=tk.BOTH, padx=15, pady=12)

    # Confirm password field
    tk.Label(content_frame, text="🔐  Xác nhận mật khẩu", font=("Segoe UI", 10, "bold"), 
             bg="#ffffff", fg="#5a6c7d", anchor="w").pack(fill=tk.X, pady=(0, 8))
    
    confirm_frame = tk.Frame(content_frame, bg="#f8f9fa", highlightbackground="#e1e8ed", 
                            highlightthickness=2, highlightcolor="#8e44ad")
    confirm_frame.pack(fill=tk.X, pady=(0, 30))
    
    entry_confirm = tk.Entry(confirm_frame, show="●", font=("Segoe UI", 11), relief=tk.FLAT, 
                            bg="#f8f9fa", fg="#2c3e50", insertbackground="#8e44ad")
    entry_confirm.pack(fill=tk.BOTH, padx=15, pady=12)

    # Focus effects
    def on_focus_in(event, frame):
        frame.configure(highlightbackground="#8e44ad", highlightthickness=2, bg="#ffffff")
    
    def on_focus_out(event, frame):
        frame.configure(highlightbackground="#e1e8ed", highlightthickness=2, bg="#f8f9fa")

    entry_user.bind("<FocusIn>", lambda e: on_focus_in(e, user_frame))
    entry_user.bind("<FocusOut>", lambda e: on_focus_out(e, user_frame))
    entry_pass.bind("<FocusIn>", lambda e: on_focus_in(e, pass_frame))
    entry_pass.bind("<FocusOut>", lambda e: on_focus_out(e, pass_frame))
    entry_confirm.bind("<FocusIn>", lambda e: on_focus_in(e, confirm_frame))
    entry_confirm.bind("<FocusOut>", lambda e: on_focus_out(e, confirm_frame))

    def handle_register():
        username = entry_user.get().strip()
        password = entry_pass.get().strip()
        confirm = entry_confirm.get().strip()

        if not username or not password:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.")
            return

        if password != confirm:
            messagebox.showerror("Lỗi", "Mật khẩu xác nhận không khớp.")
            return

        if register_user(username, password):
            messagebox.showinfo("Thành công", f"Tạo tài khoản '{username}' thành công!")
            reg.destroy()
            if back_to_login_callback:
                back_to_login_callback()
        else:
            messagebox.showerror("Lỗi", f"Tài khoản '{username}' đã tồn tại!")

    # Register button with hover effect
    register_btn = tk.Button(content_frame, text="✨  Tạo tài khoản", font=("Segoe UI", 12, "bold"),
                            bg="#8e44ad", fg="white", relief=tk.FLAT, cursor="hand2",
                            activebackground="#732d91", activeforeground="white",
                            borderwidth=0, command=handle_register)
    register_btn.pack(fill=tk.X, pady=(0, 15), ipady=12)

    def on_enter(e):
        register_btn.configure(bg="#732d91")
    
    def on_leave(e):
        register_btn.configure(bg="#8e44ad")
    
    register_btn.bind("<Enter>", on_enter)
    register_btn.bind("<Leave>", on_leave)

    # Divider
    divider_frame = tk.Frame(content_frame, bg="#ffffff", height=1)
    divider_frame.pack(fill=tk.X, pady=10)
    tk.Frame(divider_frame, bg="#e1e8ed", height=1).pack(fill=tk.X)

    # Back to login button
    back_btn = tk.Button(content_frame, text="←  Quay lại đăng nhập", 
                        font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#8e44ad",
                        relief=tk.FLAT, cursor="hand2", activebackground="#f8f9fa",
                        activeforeground="#732d91", borderwidth=0,
                        command=lambda: (reg.destroy(), back_to_login_callback() if back_to_login_callback else None))
    back_btn.pack(fill=tk.X, ipady=8)
    
    def on_back_enter(e):
        back_btn.configure(bg="#f8f9fa")
    
    def on_back_leave(e):
        back_btn.configure(bg="#ffffff")
    
    back_btn.bind("<Enter>", on_back_enter)
    back_btn.bind("<Leave>", on_back_leave)

    # Enter key binding
    entry_confirm.bind("<Return>", lambda e: handle_register())

    # Center window on screen
    reg.update_idletasks()
    x = (reg.winfo_screenwidth() // 2) - (480 // 2)
    y = (reg.winfo_screenheight() // 2) - (700 // 2)
    reg.geometry(f"480x700+{x}+{y}")

    reg.mainloop()

# Cho phép chạy file trực tiếp
if __name__ == "__main__":
    open_register_window()