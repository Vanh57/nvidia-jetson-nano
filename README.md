# Real-Time facial detection and emotion recognition system

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)

## Overview

This project implements a **Real-Time facial detection and emotion recognition system**. It leverages deep learning models for face detection, emotion analysis.

## Features

- **Real-time face detection:** Detects faces in live video streams
- **Emotion recognition:** Analyzes facial expressions to determine emotions Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral

## Demo

![Demo video](videos/demo.mp4)

*Alternatively, provide a link to the video hosted on YouTube or another platform.*

## Installation

### Prerequisites

- **Hardware:**
  - NVIDIA Jetson Nano
  - HIKVISION Camera or a USB Webcam

- **Software:**
  - NVIDIA JetPack SDK
  - Python 3.8.6

### Setup steps

1. **Setup NVIDIA Jetson Nano:**
   - Complete the initial setup and boot up

2. **Connect camera:**
   **If you have a HIKVISION Camera, follow these steps:**
   - Plug the HIKVISION Camera into the same network as the NVIDIA Jetson Nano
   - Configure the HIKVISION Camera and note the IP address, USERNAME, and PASSWORD
   - Connect to the camera via the RSTP protocol by setting the `camera_channel` parameter in the `main.py` file

   **If you have a USB Webcam, follow these steps:**
   - Plug the Webcam into the same network as the NVIDIA Jetson Nano
   - Set up the `camera_channel` parameter in the `main.py` file

3. **Clone repository into NVIDIA Jetson Nano:**
   ```bash
   git clone https://github.com/Vanh57/nvidia-jetson-nano.git
   cd nvidia-jetson-nano
   ```

4. **Create virtual environment inside the cloned repository:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

5. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage
   ```bash
   python src/main.py
   ```

## Optional: Train or Retrain the `emotion_detection.keras` model

### Dataset
   This project use the [Facial Expression Recognition (FER) Challenge dataset](https://www.kaggle.com/datasets/ashishpatel26/facial-expression-recognitionferchallenge/data) from Kaggle.

   To use this dataset, download it from the provided link and store it in the `emotion_dataset/` directory.

### Fine-tuning the model
   To enhance the accuracy of the `emotion_detection.keras` model or to customize it to your specific needs, you can retrain the model using the `emotion_detection.py` script. Running this script will generate a new `emotion_detection.keras` model.

   ```bash
   python src/emotion_detection.py
   ```
