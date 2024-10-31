# Real-Time facial recognition and emotion detection System

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)

## Overview

This project implements a **Real-Time facial recognition and emotion detection System**. It leverages deep learning models for face detection, recognition, and emotion analysis.

## Features

- **Real-time face detection:** Detects faces in live video streams
- **Face recognition:** Identifies known individuals
- **Emotion detection:** Analyzes facial expressions to determine emotions Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral

## Demo

![Demo video](videos/demo.mp4)

*Alternatively, provide a link to the video hosted on YouTube or another platform.*

## Installation

### Prerequisites

- **Hardware:**
  - NVIDIA Jetson Nano
  - HIKVISION Camera

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

   **If you have a Webcam, follow these steps:**
   - Plug the Webcam into the same network as the NVIDIA Jetson Nano
   - Set up the `camera_channel` parameter in the `main.py` file

3. **Clone repository:**
   ```bash
   git clone https://github.com/Vanh57/nvidia-jetson-nano.git
   cd nvidia-jetson-nano
   ```

4. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. **Prepare known faces:**
Add images of known individuals in data/faces/<person_name>/.
Then run:
   ```bash
   python script_to_train_modified_yolov8_model_for_face_reg.py
   ```

## Usage
   ```bash
   python src/main.py
   ```
