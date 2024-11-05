# Image-to-address

# installation guide To run groundingdino

Dependency : `python 3.10.12` , `CUDA 12.1`
`sudo apt-get install gdebi`
`sudo gdebi `
`python3 -m venv env`
`pip install -r requirements_groundingdino.txt`
`install nvcc`
```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.2-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

Create `images` folder and `crop_images` folder.
Put all the images in jpg format to `images` folder then run `python main.py`.
