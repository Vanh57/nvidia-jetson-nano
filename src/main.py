import cv2
import logging
import numpy as np
from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognizer
from src.emotion_detection import EmotionDetector
from src.utils import load_known_faces, preprocess_face
from sklearn.metrics.pairwise import cosine_similarity

def main():
    logging.basicConfig(level=logging.INFO)

    face_detector = FaceDetector('models/yolov8n-face.pt')
    face_recognizer = FaceRecognizer('models/yolov8-face-recognition.pt')
    emotion_detector = EmotionDetector('models/emotion_cnn.h5')

    known_embeddings, labels = load_known_faces()

    # Start video capture
    camera_channel = "rtsp://tma:123456xX@192.168.185.212/Streaming/Channels/101"
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
        frame = face_detector.draw_faces(frame, faces)

        for (box, conf) in faces:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            face = frame[y1:y2, x1:x2]

            # Preprocess the face for recognition
            preprocessed_face = preprocess_face(face)

            # Recognize face by comparing with known embeddings
            embedding = face_recognizer.get_embedding(preprocessed_face)
            similarities = cosine_similarity([embedding], known_embeddings)
            best_match_idx = np.argmax(similarities)
            identity = labels[best_match_idx]
            similarity_score = similarities[0, best_match_idx]

            # Draw the recognized face label on the frame
            frame = face_recognizer.draw_recognized_faces(
                frame, [(identity, box, similarity_score)]
            )

            # Detect emotions
            emotion_idx, confidence = emotion_detector.detect_emotion(face)
            emotion_label = emotion_detector.map_emotion(emotion_idx)

            # Annotate the frame with detected emotion
            frame = emotion_detector.annotate_emotion(frame, box, emotion_label, confidence)

        cv2.imshow('Real-Time Facial Recognition and Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
