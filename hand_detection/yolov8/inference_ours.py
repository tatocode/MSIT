from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")
model.predict('notebook/temp.mp4', save=True, imgsz=640, conf=0.5)