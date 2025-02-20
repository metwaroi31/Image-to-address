# Sử dụng CUDA 12.1.1 với Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Cài đặt hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    exiftool ffmpeg \
    libgl1 libglib2.0-0 \
    libjpeg-dev libpng-dev libtiff-dev \
    ninja-build git cmake gcc-11 g++-11 \
    language-pack-vi locales\
    && apt-get clean

 # Cập nhật locale để hỗ trợ tiếng Việt
RUN locale-gen vi_VN.UTF-8 && \
update-locale LANG=vi_VN.UTF-8 LANGUAGE=vi_VN:vi LC_ALL=vi_VN.UTF-8

# Thiết lập biến môi trường để container nhận diện tiếng Việt
ENV LANG=vi_VN.UTF-8 \
LANGUAGE=vi_VN:vi \
LC_ALL=vi_VN.UTF-8

# Chọn GCC 11 làm mặc định
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# Đặt thư mục làm việc
WORKDIR /app

# Tạo môi trường ảo & cập nhật pip
RUN python3 -m venv env && \
    /app/env/bin/pip install --upgrade pip setuptools wheel

# Cài đặt PyTorch với CUDA 12.1
RUN /app/env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone & Cài đặt GroundingDINO
WORKDIR /app/groundingdino
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git . && \
    /app/env/bin/pip install -e .

# Sao chép toàn bộ mã nguồn vào container
WORKDIR /app
COPY . .

# Cài đặt các thư viện từ requirements.txt (SAU CÙNG)
RUN /app/env/bin/pip install -r requirements.txt



