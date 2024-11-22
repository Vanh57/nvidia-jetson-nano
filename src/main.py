import cv2
import logging
import numpy as np
from face_detection import FaceDetector
from emotion_detection import EmotionDetector

def main():
    logging.basicConfig(level=logging.INFO)

    face_detector = FaceDetector('models/yolov8n-face.pt')
    emotion_detector = EmotionDetector('models/emotion_detection.h5')

    # Start video capture
    ### Define HIKVISION channel
    # camera_channel = "rtsp://tma:123456xX@192.168.185.212/Streaming/Channels/101"
    ### Define USB Webcam channel
    camera_channel = 0
    cap = cv2.VideoCapture(camera_channel)
    if not cap.isOpened():
        logging.error("Error: Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Frame capture failed")
            break

        # Detect faces in the frame
        faces = face_detector.detect_faces(frame)

        for (box, conf) in faces:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face = frame[y1:y2, x1:x2]

            # Detect emotions
            emotion_idx, confidence = emotion_detector.detect_emotion(face)
            # emotion_label = emotion_detector.map_emotion(emotion_idx)

            # Annotate the frame with detected emotion
            frame = emotion_detector.annotate_emotion(frame, box, emotion_idx + 3, confidence)

        frame = face_detector.draw_faces(frame, faces)
        cv2.imshow('Real-Time facial detection and emotion recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
