# -*- coding: utf-8 -*-
"""
Dataset Finder
Tự động tìm file dataset (fer2013.csv, ckextended.csv)
"""
import os


def find_dataset_file():
    """
    Tự động tìm file dataset trong các vị trí phổ biến
    
    Returns:
        str: đường dẫn đến file dataset, hoặc None nếu không tìm thấy
    """
    # Các tên file có thể
    dataset_names = ['fer2013.csv', 'FER2013.csv', 'ckextended.csv', 'CKExtended.csv']
    
    # Các vị trí có thể chứa dataset
    search_paths = [
        '.',  # Thư mục hiện tại
        './data',  # Thư mục data
        './datasets',  # Thư mục datasets
        '../',  # Thư mục cha
        '../data',
        '../datasets',
        os.path.expanduser('~/Downloads'),  # Thư mục Downloads
        os.path.expanduser('~/Desktop'),  # Thư mục Desktop
    ]
    
    # Tìm trong các vị trí
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for dataset_name in dataset_names:
            full_path = os.path.join(search_path, dataset_name)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                return os.path.abspath(full_path)
    
    return None


def get_dataset_info(csv_path):
    """
    Lấy thông tin về dataset
    
    Args:
        csv_path: đường dẫn đến file CSV
    
    Returns:
        dict: thông tin dataset (name, size, type)
    """
    if not csv_path or not os.path.exists(csv_path):
        return None
    
    filename = os.path.basename(csv_path).lower()
    
    # Xác định loại dataset
    if 'fer2013' in filename:
        dataset_type = 'FER2013'
        expected_samples = 35887  # FER2013 có 35887 samples
    elif 'ck' in filename:
        dataset_type = 'CK+ Extended'
        expected_samples = 981  # CK+ có ~981 samples
    else:
        dataset_type = 'Unknown'
        expected_samples = 0
    
    # Lấy kích thước file
    file_size = os.path.getsize(csv_path)
    file_size_mb = file_size / (1024 * 1024)
    
    return {
        'name': dataset_type,
        'filename': os.path.basename(csv_path),
        'path': csv_path,
        'size_mb': file_size_mb,
        'expected_samples': expected_samples
    }


def validate_dataset(csv_path):
    """
    Kiểm tra xem file CSV có phải là dataset hợp lệ không
    
    Args:
        csv_path: đường dẫn đến file CSV
    
    Returns:
        (bool, str): (is_valid, error_message)
    """
    if not csv_path or not os.path.exists(csv_path):
        return False, "File không tồn tại"
    
    try:
        # Đọc vài dòng đầu để kiểm tra format
        with open(csv_path, 'r') as f:
            header = f.readline().strip()
            
            # Kiểm tra header
            if 'emotion' not in header.lower() or 'pixels' not in header.lower():
                return False, "File CSV không đúng format (thiếu cột emotion hoặc pixels)"
            
            # Đọc 1 dòng data để kiểm tra
            data_line = f.readline().strip()
            if not data_line:
                return False, "File CSV rỗng"
            
            parts = data_line.split(',')
            if len(parts) < 2:
                return False, "File CSV không đúng format"
        
        return True, "Dataset hợp lệ"
    
    except Exception as e:
        return False, f"Lỗi đọc file: {str(e)}"


if __name__ == "__main__":
    # Test
    dataset_path = find_dataset_file()
    if dataset_path:
        print(f"✅ Tìm thấy dataset: {dataset_path}")
        
        info = get_dataset_info(dataset_path)
        print(f"   Loại: {info['name']}")
        print(f"   Kích thước: {info['size_mb']:.2f} MB")
        
        is_valid, msg = validate_dataset(dataset_path)
        print(f"   Hợp lệ: {is_valid} - {msg}")
    else:
        print("❌ Không tìm thấy dataset")
