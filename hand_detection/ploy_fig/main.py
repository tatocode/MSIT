import os
from ultralytics import YOLO
import random
from pprint import pprint
import supervision as sv
from supervision.draw.color import Color, ColorPalette
from supervision.utils import image
import cv2

if __name__ == "__main__":
  img_dir = r"C:\Users\Tatocode\Documents\desk\dataset\hand_detection\ours\images"
  test_file = r"C:\Users\Tatocode\Documents\desk\dataset\hand_detection\ours\test.txt"
  model_path = r"C:\Users\Tatocode\Documents\desk\code\hand_detection\yolov8\runs\detect\train3\weights\best.pt"
  chose_num = 30
  tages = {0: "HI", 1: "HOI"}
  tages_color = {0: (0, 0, 255), 1: (255, 0, 0)}
  bounding_box_annotator = sv.BoundingBoxAnnotator(
    color=ColorPalette([Color(255, 0, 0), Color(0, 0, 255)]), thickness=5)
  label_annotator = sv.LabelAnnotator(color=ColorPalette([Color(255, 0, 0), Color(
    0, 0, 255)]), text_color=Color(255, 255, 255), text_scale=0.7, text_thickness=2)

  with open(test_file, "r") as rf:
    ct = rf.readlines()
  test_imgs = [c.strip().split("/")[-1] for c in ct]
  test_imgs_path = sorted(random.sample(
    [os.path.join(img_dir, ti) for ti in test_imgs], chose_num))

  # load a custom model
  model = YOLO(model=model_path)

  # inference
  for ret in model(test_imgs_path, stream=True):
    ret.names = tages
    sv_det = sv.Detections.from_ultralytics(ret)
    labels = [
      f"{ret.names[ci]} {sv_det.confidence[ix]:.2f}" for ix, ci in enumerate(sv_det.class_id)]
    annotated_image = bounding_box_annotator.annotate(
      scene=ret.orig_img, detections=sv_det)
    annotated_image = label_annotator.annotate(
      scene=annotated_image, detections=sv_det, labels=labels)
    cv2.imwrite(f"save_result\\{ret.path.split(os.sep)[-1]}", annotated_image)
    # print(type(annotated_image))
    # image.save_image(annotated_image, f"save_result\\{ret.path.split(os.sep)[-1]}")
    
