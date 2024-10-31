import cv2
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)

class FaceDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        logging.info(f"YOLOv8 model loaded from {model_path}")

    def detect_faces(self, frame, conf_threshold=0.5):
        """
        Detect faces in the given frame using the YOLOv8 model.
        Returns a list of detected faces with their bounding boxes and confidence scores.
        """
        results = self.model(frame)
        faces = []
        for result in results:
            for bbox, conf in zip(result.boxes, result.conf):
                if conf > conf_threshold:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox.xyxy[0].numpy()]
                    faces.append(((x1, y1, x2, y2), conf.numpy()))
        return faces

    def draw_faces(self, frame, faces):
        """
        Draw rectangles around detected faces and display confidence scores.
        """
        for (box, conf) in faces:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'{conf:.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        return frame
