from ultralytics import YOLO

# Load a model
model = YOLO(r'runs/detect/train/weights/best.pt')  # load a pretrained model (recommended for training)

# Use the model
model.train(data="hand.yaml", epochs=200, imgsz=640, batch=24)  # train the model