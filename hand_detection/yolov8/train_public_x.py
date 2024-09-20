from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="hand_public.yaml", epochs=50, imgsz=640, batch=28)  # train the model