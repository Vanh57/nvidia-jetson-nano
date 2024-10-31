import cv2
from ultralytics import YOLO
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class FaceRecognizer:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        logging.info(f"YOLOv8 face recognition model loaded from {model_path}")

    def get_embedding(self, face):
        """
        Extract the embedding from a face using the YOLOv8 model.
        The model outputs a feature vector that can be used to recognize the face.
        """
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32') / 255.0
        embedding = self.model(face)[0].numpy()
        return embedding

    def get_batch_embeddings(self, faces):
        """
        Process a batch of faces and return their embeddings.
        This is optimized for batch processing to reduce inference time.
        """
        embeddings = []
        for face in faces:
            embedding = self.get_embedding(face)
            embeddings.append(embedding)
        return np.array(embeddings)

    # def recognize_face(self, embedding, known_embeddings, labels, threshold=0.5):
    #     """
    #     Compare the embedding of a detected face with known embeddings to recognize the face.
    #     """
    #     distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    #     min_dist = np.min(distances)
    #     if min_dist < threshold:
    #         return labels[np.argmin(distances)], min_dist
    #     return "Unknown", min_dist

    def draw_recognized_faces(self, frame, recognized_faces):
        """
        Draw rectangles around recognized faces, recognized identity threshold = 0.5 for now
        """
        for (identity, (x1, y1, x2, y2), confidence) in recognized_faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f'{identity}: {confidence:.2f}' if confidence >= 0.5 else 'Unknown',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36,255,12),
                2
            )
        return frame
