# users/user_manager.py
import csv
import os
import hashlib

USER_FILE = os.path.join(os.path.dirname(__file__), "users.csv")

def hash_password(password: str) -> str:
    """Mã hoá mật khẩu (SHA256) để bảo mật."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Đọc danh sách người dùng từ CSV."""
    users = {}
    if not os.path.exists(USER_FILE):
        with open(USER_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["username", "password"])
    with open(USER_FILE, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            users[row["username"]] = row["password"]
    return users

def register_user(username: str, password: str) -> bool:
    """Đăng ký người dùng mới."""
    users = load_users()
    if username in users:
        return False  # Tên tài khoản đã tồn tại
    with open(USER_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([username, hash_password(password)])
    return True

def authenticate_user(username: str, password: str) -> bool:
    """Xác thực đăng nhập."""
    users = load_users()
    hashed = hash_password(password)
    return username in users and users[username] == hashed

def get_all_users():
    """Lấy danh sách toàn bộ tài khoản."""
    return list(load_users().keys())

def delete_user(username: str) -> bool:
    """Xoá tài khoản (nếu cần)."""
    users = load_users()
    if username not in users:
        return False
    del users[username]
    with open(USER_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["username", "password"])
        for u, p in users.items():
            writer.writerow([u, p])
    return True
