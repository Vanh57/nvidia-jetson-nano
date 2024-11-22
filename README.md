# Real-Time facial detection and emotion recognition system

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Applications](#applications)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [System Flow](#Flow)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a **Real-Time facial detection and emotion recognition system**. It leverages deep learning models for face detection, emotion analysis.

## Features

- **Real-time face detection:** Detects faces in live video streams
- **Emotion recognition:** Analyzes facial expressions to determine emotions Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral

## Applications

- **Smart Interactions:** Adjust responses or actions based on user emotions.
- **Wellness & Accessibility:** Track and respond to emotional cues.
- **Education Tools:** Detect emotions to personalize learning experiences.

## Demo

https://drive.google.com/file/d/1HL4nieko-YTUYC-HZvycslImt_dzazbh/view?usp=drive_link


## Installation

### Prerequisites

- **Hardware:**
  - NVIDIA Jetson Nano
  - A USB Webcam or HIKVISION Camera 

- **Software:**
  - NVIDIA JetPack SDK 4.6.1
  - Python 3.6.9

### Setup steps

1. **Setup NVIDIA Jetson Nano:**
   - Complete the initial setup and boot up

2. **Connect camera:**
   **If you have a USB Webcam, follow these steps: (In this repo, I'm using USB Webcam)**
   - Plug the Webcam into the same network as the NVIDIA Jetson Nano
   - Set up the `camera_channel` parameter in the `main.py` file

   **If you have a HIKVISION Camera, follow these steps:**
   - Plug the HIKVISION Camera into the same network as the NVIDIA Jetson Nano
   - Configure the HIKVISION Camera and note the IP address, USERNAME, and PASSWORD
   - Connect to the camera via the RSTP protocol by setting the `camera_channel` parameter in the `main.py` file

3. **Clone repository into NVIDIA Jetson Nano:**
   ```bash
   git clone https://github.com/Vanh57/nvidia-jetson-nano.git
   cd nvidia-jetson-nano
   ```

4. **Setup environment and install dependencies in NVIDIA Jetson Nano:**
   #### Install OpenCV:
   Run commands:
   ```bash
   wget https://github.com/lanzani/jetson-libraries/raw/main/libraries/opencv/l4t32.7.1/py3.6.9/ocv4.8.0/OpenCV-4.8.0-aarch64.sh
   chmod +x OpenCV-4.8.0-aarch64.sh
   sudo ./OpenCV-4.8.0-aarch64.sh --prefix=/usr/local --skip-license --exclude-subdir
   sudo apt-get update
   sudo apt-get -y install \
      python3-pip \
      libtesseract4 \
      libatlas3-base \
      python3-numpy
   sudo apt-get clean
   vi ~/.bashrc
   ```
   And then add the following line at the bottom of the file:
   ```bash
   export PYTHONPATH=/usr/local/lib/python3.6/site-packages:$PYTHONPATH
   ```
   Exit then run command:
   ```bash
   source ~/.bashrc
   ```

   #### Install Tensorflow::
   Run commands
   ```
   sudo apt-get update
   sudo apt-get install -y python3-pip pkg-config
   sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
   sudo pip3 install -U pip testresources setuptools
   sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
   pip3 install --verbose 'protobuf<4' 'Cython==0.29.36'
   git clone https://github.com/h5py/h5py.git
   cd h5py
   git checkout 3.1.0
   H5PY_SETUP_REQUIRES=0 pip3 install . --no-deps --no-build-isolation
   sudo pip3 install -U numpy==1.19.4 future mock keras_preprocessing keras_applications gast==0.2.1 protobuf pybind11 packaging
   cd ..
   wget --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
   pip3 install --verbose tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
   ```

   #### Install Ultralytics to run YOLOv8 on Python 3.6.9:
   Run commands:
   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   ```
   Then do the following modifications:
   - In `./ultralytics/ultralytics/utils/__init__.py`:
      - Comment out `import importlib.metadata` on line 4 and replace it with:
         ```python
         import pkg_resources
         ```
      - Replace `importlib.metadata.version("torchvision")` with:
         ```python
         TORCHVISION_VERSION = pkg_resources.get_distribution("torchvision").version
         ```

   - In `./ultralytics/ultralytics/utils/checks.py`:
      - Comment out `import importlib.metadata` on line 13.

   - In `./ultralytics/ultralytics/hub/auth.py`:
      - Replace:
         ```python
         if header := self.get_auth_header():
         ```
         with:
         ```python
         header = self.get_auth_header()
         if header:
         ```

   > **NOTE:** After completing the modification steps and running the prediction, you may encounter additional errors related to Ultralytics (Since they could update these code and when you clone, the code is newer). Review the error messages and adjust the code within the Ultralytics folder as needed to ensure compatibility with Python 3.6.9

   #### Install required packages:
   ```bash
   pip3 install --upgrade pip
   pip3 install -r requirements.txt
   ```

   #### Install Pytorch and Torchvision for Jetpack 4.6.1:
   Pytorch:
   ```bash
   wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
   sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
   pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
   ```

   Torchvision::
   ```bash
   sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
   git clone --branch 0.11.1 https://github.com/pytorch/vision torchvision
   cd torchvision
   export BUILD_VERSION=0.11.1
   python3 setup.py install --user
   cd ../
   ```

## Usage
   - Modify the path to the ultralytics folder inside the face_detection if it doesn't correct.
   - Run command:
   ```bash
   python3 src/main.py
   ```

## <Optional>: Train or Retrain the `emotion_detection.h5` model

### Dataset
   This project use the [Facial Expression Recognition (FER) Challenge dataset](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge/data) from Kaggle.

   To use this dataset, download it from the provided link and store it in the `emotion_dataset/` directory.

### Fine-tuning the model
   To enhance the accuracy of the `emotion_detection.h5` model or to customize it to your specific needs, you can retrain the model using the `emotion_detection.py` script. Running this script will generate a new `emotion_detection.h5` model.
   Remember to update the path of the files if they don't correct.

   ```bash
   python3 src/emotion_detection.py
   ```

   After the new `emotion_detection.h5` model is generated, you can rerun the prediction script again.

## Flow

   ```bash
   Camera --> Face Detection (YOLOv8) --> Emotion Recognition (CNN) --> Real-Time Output
   ```

## Troubleshooting
### Low Memory Issues
The Jetson Nano can sometimes run low on memory. To resolve this, set up swap space with the following commands:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Camera Connection Problems:

- If using an IP camera, double-check the IP, username, and password in main.py.
- If using a USB camera, confirm the correct device path (e.g., /dev/video0), if this path is available, this means we can use 0 value for camera

### Slow Frame Rate:
- Check that GPU acceleration is active.
- Ensure you're following my setup steps specifically for the Jetson Nano.
