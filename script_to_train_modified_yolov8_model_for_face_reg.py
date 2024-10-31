from ultralytics import YOLO

model = YOLO('yolov8n-face.pt')

model.train(data='data/faces', epochs=10, imgsz=640, batch=16)

model.save('models/yolov8-face-recognition.pt')
