FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Cài đặt hệ thống
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    opencv-python exiftool ffmpeg \
    libgl1 libglib2.0-0 \
    ninja-build git cmake gcc-11 g++-11

# Chọn GCC 11 làm mặc định
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Cấu hình CUDA architecture
ENV TORCH_CUDA_ARCH_LIST="7.5+PTX"

# Đặt thư mục làm việc
WORKDIR /app

# Copy mã nguồn vào container
COPY . .

# Tạo môi trường ảo và cài thư viện
RUN python3 -m venv env
RUN env/bin/pip install --upgrade pip setuptools wheel
RUN env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN python -m ensurepip --default-pip
RUN cd groundingdino/
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
RUN cd GroundingDINO/
RUN env/bin/pip install -e .
RUN cd ..
RUN cd ..
RUN env/bin/pip install -r requirements.txt