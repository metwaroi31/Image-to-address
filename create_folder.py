import os

def create_directories(directory_paths):
    for directory_path in directory_paths:
        try:
            # Chuyển đổi đường dẫn Windows sang WSL nếu cần thiết
            directory_path = directory_path.replace("\\", "/").replace("D:", "/mnt/d")

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"Đã tạo thư mục: {directory_path}")
            else:
                print(f"Thư mục đã tồn tại: {directory_path}")
        except Exception as e:
            print(f"Lỗi khi tạo thư mục {directory_path}: {e}")

# Danh sách thư mục cần tạo
directory_paths = [
    "D:/map4d/Image-to-address/predictions_json",
    "D:/map4d/Image-to-address/images_input",
    "D:/map4d/Image-to-address/images_annotated",
    "D:/map4d/Image-to-address/images",
    "D:/map4d/Image-to-address/crop_images"
]

create_directories(directory_paths)

