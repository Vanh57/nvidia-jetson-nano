import cv2
import numpy as np
import os
from src.face_recognition import FaceRecognizer
from tqdm import tqdm
import logging
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)

def preprocess_face(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return face

def load_known_faces(batch_size=32):
    face_recognizer = FaceRecognizer('models/yolov8-face-recognition.pt')
    known_embeddings = []
    labels = []

    person_dirs = [
        os.path.join('data/faces/', person_name)
        for person_name in os.listdir('data/faces/')
    ]
    all_images = []
    all_labels = []

    # Load images and labels
    for person_dir in person_dirs:
        person_name = os.path.basename(person_dir)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                all_images.append(image)
                all_labels.append(person_name)
            else:
                logging.warning(f"Failed to read image: {image_path}")

    # Process images in batches
    for i in tqdm(range(0, len(all_images), batch_size)):
        batch_images = [preprocess_face(img) for img in all_images[i:i+batch_size]]
        batch_labels = all_labels[i:i+batch_size]

        # Get embeddings for the batch
        batch_embeddings = face_recognizer.get_batch_embeddings(batch_images)

        for embedding, label in zip(batch_embeddings, batch_labels):
            known_embeddings.append(embedding)
            labels.append(label)

    known_embeddings = np.array(known_embeddings)
    labels = np.array(labels)

    known_embeddings = normalize(known_embeddings)

    # Average embeddings for each person
    unique_labels = np.unique(labels)
    averaged_embeddings = []
    averaged_labels = []

    for label in unique_labels:
        idxs = np.where(labels == label)[0]
        avg_embedding = np.mean(known_embeddings[idxs], axis=0)
        averaged_embeddings.append(avg_embedding)
        averaged_labels.append(label)

    logging.info("Known faces loaded and processed.")
    return np.array(averaged_embeddings), np.array(averaged_labels)
