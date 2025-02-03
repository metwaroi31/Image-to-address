import os
import shutil

def create_directories(base_path, subdirectories):
    """
    Tạo các thư mục con trong thư mục gốc.
    """
    for subdirectory in subdirectories:
        directory_path = os.path.join(base_path, subdirectory)
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"✅ Đã tạo thư mục: {directory_path}")
            else:
                print(f"❌ Thư mục đã tồn tại: {directory_path}")
        except Exception as e:
            print(f"⚠️ Lỗi khi tạo thư mục {directory_path}: {e}")


def delete_directories(base_path, subdirectories):
    """
    Xóa các thư mục con trong thư mục gốc.
    """
    for subdirectory in subdirectories:
        directory_path = os.path.join(base_path, subdirectory)
        try:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
                print(f"✅ Đã xóa thư mục: {directory_path}")
            else:
                print(f"❌ Thư mục không tồn tại: {directory_path}")
        except Exception as e:
            print(f"⚠️ Lỗi khi xóa thư mục {directory_path}: {e}")


def main():
    base_path = "./"
    subdirectories = [
        "predictions_json",
        "images_input",
        "images_annotated",
        "images",
        "crop_images"
    ]

    print("Danh sách thư mục:")
    for subdirectory in subdirectories:
        print(f"- {os.path.join(base_path, subdirectory)}")

    action = input("\nBạn muốn thực hiện thao tác nào? (create/delete): ").strip().lower()

    if action == 'create':
        create_directories(base_path, subdirectories)
    elif action == 'delete':
        confirm = input("\nBạn có chắc chắn muốn xóa những thư mục này? (y/n): ").strip().lower()
        if confirm == 'y':
            delete_directories(base_path, subdirectories)
        else:
            print("❌ Hủy thao tác.")
    else:
        print("⚠️ Thao tác không hợp lệ.")


if __name__ == "__main__":
    main()
