# ui/login_window.py
import os
import sys
import tkinter as tk
from tkinter import messagebox

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from users.user_manager import authenticate_user
from ui.register_window import open_register_window


def open_login_window():
    """Mở giao diện đăng nhập hiện đại - Dark theme."""
    root = tk.Tk()
    root.title("Emotion Scanner - Đăng nhập")
    root.geometry("600x700")
    root.resizable(False, False)
    
    # Dark background
    canvas = tk.Canvas(root, width=600, height=700, highlightthickness=0, bg="#2d3748")
    canvas.pack(fill="both", expand=True)

    # Main card with rounded effect
    main_frame = tk.Frame(canvas, bg="#3a4556", relief=tk.FLAT)
    main_frame.place(relx=0.5, rely=0.5, anchor="center", width=500, height=580)

    # Subtle shadow
    for offset in range(6, 0, -2):
        shadow = tk.Frame(canvas, bg=f"#{'%02x' % (30 - offset * 2)}{'%02x' % (35 - offset * 2)}{'%02x' % (45 - offset * 2)}")
        shadow.place(relx=0.5, rely=0.5 + offset/800, anchor="center", 
                    width=500 + offset*2, height=580 + offset*2)
        shadow.lower()

    # Header with lock icon
    header_frame = tk.Frame(main_frame, bg="#3a4556")
    header_frame.pack(pady=(50, 40))
    
    # Lock icon (using text)
    tk.Label(header_frame, text="🔓", font=("Segoe UI Emoji", 50), 
             bg="#3a4556", fg="#ffa500").pack()
    
    # Title
    tk.Label(header_frame, text="Đăng nhập", font=("Segoe UI", 32, "bold"), 
             bg="#3a4556", fg="#17c1e8").pack(pady=(15, 0))

    # Content frame
    content_frame = tk.Frame(main_frame, bg="#3a4556")
    content_frame.pack(fill=tk.BOTH, expand=True, padx=60, pady=(0, 50))

    # Username field
    user_frame = tk.Frame(content_frame, bg="#e8f0f7", highlightthickness=0)
    user_frame.pack(fill=tk.X, pady=(0, 25))
    
    entry_user = tk.Entry(user_frame, font=("Segoe UI", 14), relief=tk.FLAT, 
                          bg="#e8f0f7", fg="#2d3748", insertbackground="#17c1e8",
                          bd=0)
    entry_user.pack(fill=tk.BOTH, padx=20, pady=18)
    entry_user.insert(0, "")

    # Password field
    pass_frame = tk.Frame(content_frame, bg="#e8f0f7", highlightthickness=0)
    pass_frame.pack(fill=tk.X, pady=(0, 35))
    
    entry_pass = tk.Entry(pass_frame, show="●", font=("Segoe UI", 14), relief=tk.FLAT, 
                          bg="#e8f0f7", fg="#2d3748", insertbackground="#17c1e8",
                          bd=0)
    entry_pass.pack(fill=tk.BOTH, padx=20, pady=18)

    # Focus effects
    def on_focus_in(event, frame):
        frame.configure(bg="#ffffff")
    
    def on_focus_out(event, frame):
        frame.configure(bg="#e8f0f7")

    entry_user.bind("<FocusIn>", lambda e: on_focus_in(e, user_frame))
    entry_user.bind("<FocusOut>", lambda e: on_focus_out(e, user_frame))
    entry_pass.bind("<FocusIn>", lambda e: on_focus_in(e, pass_frame))
    entry_pass.bind("<FocusOut>", lambda e: on_focus_out(e, pass_frame))

    def handle_login():
        username = entry_user.get().strip()
        password = entry_pass.get().strip()

        if not username or not password:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đầy đủ tài khoản và mật khẩu.")
            return

        if authenticate_user(username, password):
            messagebox.showinfo("Thành công", f"Chào mừng {username}!")
            root.withdraw()
            from ui.main_window import open_main_window
            open_main_window(username, root)
        else:
            messagebox.showerror("Lỗi", "Sai tài khoản hoặc mật khẩu!")

    # Login button with modern hover effect
    login_btn = tk.Button(content_frame, text="🚀  Đăng nhập", font=("Segoe UI", 12, "bold"),
                          bg="#17c1e8", fg="white", relief=tk.FLAT, cursor="hand2",
                          activebackground="#14a8ce", activeforeground="white",
                          borderwidth=0, command=handle_login)
    login_btn.pack(fill=tk.X, pady=(0, 30), ipady=16)

    def on_enter(e):
        login_btn.configure(bg="#14a8ce")
    
    def on_leave(e):
        login_btn.configure(bg="#17c1e8")
    
    login_btn.bind("<Enter>", on_enter)
    login_btn.bind("<Leave>", on_leave)

    # Register section - inline style
    register_frame = tk.Frame(content_frame, bg="#3a4556")
    register_frame.pack(fill=tk.X, pady=(0, 0))
    
    # Create inner frame to center content
    inner_frame = tk.Frame(register_frame, bg="#3a4556")
    inner_frame.pack(expand=True)
    
    # Text label (inline)
    tk.Label(inner_frame, text="Chưa có tài khoản?  ", 
             font=("Segoe UI", 12), bg="#3a4556", fg="#cbd5e0").pack(side=tk.LEFT)
    
    # Register link button (inline)
    register_link_btn = tk.Button(inner_frame, text="Đăng ký ngay", 
                                  font=("Segoe UI", 12, "bold"), bg="#3a4556", fg="#17c1e8",
                                  relief=tk.FLAT, cursor="hand2", activebackground="#3a4556",
                                  activeforeground="#14a8ce", borderwidth=0,
                                  command=lambda: (root.withdraw(), open_register_window(lambda: root.deiconify())))
    register_link_btn.pack(side=tk.LEFT)

    def on_reg_link_enter(e):
        register_link_btn.configure(fg="#14a8ce", font=("Segoe UI", 12, "bold underline"))
    
    def on_reg_link_leave(e):
        register_link_btn.configure(fg="#17c1e8", font=("Segoe UI", 12, "bold"))
    
    register_link_btn.bind("<Enter>", on_reg_link_enter)
    register_link_btn.bind("<Leave>", on_reg_link_leave)

    # Enter key binding
    entry_pass.bind("<Return>", lambda e: handle_login())

    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (600 // 2)
    y = (root.winfo_screenheight() // 2) - (700 // 2)
    root.geometry(f"600x700+{x}+{y}")

    root.mainloop()


if __name__ == "__main__":
    open_login_window()