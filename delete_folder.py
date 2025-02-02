import os
import shutil

def safe_delete(directory_paths):
    for directory_path in directory_paths:
        # Xác nhận trước khi xóa
        confirm = input(f"Bạn có chắc chắn muốn xóa '{directory_path}'? (y/n): ")
        if confirm.lower() == 'y':
            try:
                if os.path.exists(directory_path):
                    shutil.rmtree(directory_path)
                    print(f"Đã xóa thư mục: {directory_path}")
                else:
                    print(f"Thư mục không tồn tại: {directory_path}")
            except Exception as e:
                print(f"Lỗi khi xóa thư mục {directory_path}: {e}")
        else:
            print(f"Bỏ qua xóa: {directory_path}")

directory_paths = ["D:/path/to/example"]
safe_delete(directory_paths)

