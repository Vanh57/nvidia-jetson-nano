import tensorflow as tf
import numpy as np
import logging
import cv2

logging.basicConfig(level=logging.INFO)

class EmotionDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        logging.info(f"Emotion detection model loaded from {model_path}")
        self.emotion_dict = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }
        self.emotion_colors = {
            0: (0, 0, 255),
            1: (0, 255, 255),
            2: (255, 0, 0),
            3: (0, 255, 0),
            4: (255, 255, 0),
            5: (255, 0, 255),
            6: (128, 128, 128)
        }

    def detect_emotion(self, face):
        """
        Detects emotion in the provided face image
        """
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1).astype('float32') / 255.0
        prediction = self.model.predict(face)
        emotion_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        return emotion_idx, confidence

    def map_emotion(self, emotion_idx):
        """
        Maps the emotion index to the corresponding emotion label.
        """
        return self.emotion_dict.get(emotion_idx, "Unknown")

    def annotate_emotion(self, frame, box, emotion_idx, confidence):
        """
        Annotates the frame with the detected emotion and its confidence score.
        """
        x1, y1, x2, y2 = box
        emotion_label = self.map_emotion(emotion_idx)
        color = self.emotion_colors.get(emotion_idx, (0, 255, 0))
        cv2.putText(
            frame,
            f'{emotion_label}: {confidence:.2f}',
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
        return frame
