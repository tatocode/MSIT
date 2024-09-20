from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="hand_public.yaml", epochs=200, imgsz=640, batch=8)  # train the model
