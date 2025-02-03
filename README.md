# GoPro to POI Video Processing

This project processes GoPro Max camera videos to extract Points of Interest (POI) information. It integrates tools like PyTorch, GroundingDINO, and Flash Attention for high-performance video and image processing.

---

## 📦 Requirements and Dependencies

Before running the project, ensure you have the following installed:

- **Python 3.10** and development tools
- **exiftool**
- **ffmpeg**
- **PyTorch** with CUDA 12.1 support
- **GroundingDINO**
- **Flash Attention**

---

## 🚀 Installation Guide

Follow these steps to set up the environment and install the necessary dependencies:

1. Install Python development tools:
   ```bash
   sudo apt-get install python3.10-dev build-essential
   ```

2. Install `exiftool`:
   ```bash
   sudo apt-get install exiftool
   ```

3. Install `ffmpeg`:
   ```bash
   sudo apt-get install ffmpeg
   ```

4. Install PyTorch with CUDA 12.1:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. Clone and set up GroundingDINO:
   ```bash
   cd ../
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   cd GroundingDINO
   pip install -e .
   ```

   - If you encounter errors, locate the file `_C.cpython-310-x86_64-linux-gnu.so` and move it:
     ```bash
     mv _C.cpython-310-x86_64-linux-gnu.so ../Image-to-address/groundingdino/
     ```
   - Return to the project folder:
     ```bash
     cd ../Image-to-address
     ```

6. Install Flash Attention (manual installation required):
   ```bash
   wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   pip install --no-dependencies --upgrade flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   ```

7. Install additional Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

1. Create the following folders in the project directory:
   ```
   images_input, images_video, images, crop_images
   ```

2. Place a GoPro Max video into the `images_video` folder.

3. Run the main script:
   ```bash
   python main.py
   ```

4. Repeat the process:
   - Organize your data as needed.
   - Delete all folders and start again from step 1 for processing new videos.

---

## 📬 Contact

For inquiries or support, please reach out to:

- **Anh Bui**  
  - Email: anhbuiembedded@gmail.com
  - Call: Anh Bui

- **Jimmy Le**  
  - Email: namthanhle1907@gmail.com
  - Call: Jimmy

---

## 🌟 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

---

## 📄 License

This project is licensed under [Your License Name Here]. See the `LICENSE` file for details.
