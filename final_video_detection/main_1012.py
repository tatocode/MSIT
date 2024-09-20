import tqdm
import os
import sys
import json
import supervision as sv
import cv2
import random
import numpy as np
import torch
from region.plan import *
from region.custom import deal_result
from torchvision import transforms as T
from torch import nn
import torchvision
from ultralytics import YOLO
from region.Polynet.polynet import PolyNet
import warnings
warnings.filterwarnings("ignore")

# 队列


class FrameQueue:
  def __init__(self, length=10) -> None:
    self.length = length
    self.data = []

  def add_item(self, item):
    if len(self.data) < self.length:
      self.data.append(item)
    else:
      del self.data[0]
      self.data.append(item)

  def get_data(self):
    return self.data

  def clear_data(self):
    self.data = []

  def __len__(self):
    return len(self.data)


def setup_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True


def build_seg_model(device):
  seg_ckpt_path = r"region/checkpoint/best.pth"
  model = PolyNet(img_size=512)
  model.load_state_dict(torch.load(seg_ckpt_path))
  model.to(device)
  model.eval()
  return model


def build_yolo_model():
  yolo_ckpt_path = r"hand/checkpoint/best.pt"
  model = YOLO(yolo_ckpt_path)
  return model


def build_cls_model(device):
  cls_ckpt_path = r"instrument/checkpoint/best.pth"
  model = torchvision.models.densenet201()
  model.classifier = nn.Linear(model.classifier.in_features, 4)
  model.load_state_dict(torch.load(cls_ckpt_path))
  model.to(device)
  model.eval()
  return model


def infer_seg_model(frame, model, device, img_size=512):
  frame_h, frame_w, _ = frame.shape
  frame_cp = frame.copy()
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  tsfm = T.Compose([
      T.ToPILImage(),
      T.Resize((img_size, img_size),
               interpolation=torchvision.transforms.InterpolationMode.NEAREST),
      T.ToTensor(),
      T.Normalize(mean=mean, std=std)
  ])
  img = tsfm(frame).type(torch.float32).unsqueeze(dim=0).to(device)

  with torch.no_grad():
    _, point_out = model(img)
    point_out = point_out.squeeze(dim=0).cpu().numpy()
    point_out[point_out < 0] = 0
    point_out[point_out > 1] = 1
    point_out *= img_size

  points = restitute_pos(img_size, img_size, frame_w, frame_h, point_out)
  proj_img = deal_result(points, plan2, ori_size=(
    frame.shape[1], frame.shape[0]))  # 根据plan2画出分区投影
  final_img = plot_proj_img(frame_cp, proj_img, palette=[[0, 0, 0], [255, 0, 0], [
                            0, 255, 0], [0, 0, 255]], opacity=0.5)  # 将分区投影图像覆盖到原图中
  return final_img, points, proj_img


def infer_cls_model(fq, model, track_id, device, img_size=224):
  frames = fq.get_data()
  w_scale_L = 0.7
  w_scale_R = 1
  h_scale_T = 0.7
  h_scale_B = 1
  std = [0.229, 0.224, 0.225]
  mean = [0.485, 0.456, 0.406]

  use_f = []
  for f, boxes in frames:
    try:
      idx = list(boxes.id.type(torch.int8)).index(track_id)
      use_f.append((f, idx, boxes))
    except:
      continue

  batch_images = []
  for iou_img, idx, boxes in use_f:
    x, y, w, h = boxes.xywh[idx]
    x, y = int(x), int(y)
    tl = (x - int(w_scale_L * w) if x - int(w_scale_L * w) > 0 else 0,
          y - int(h_scale_T * h) if y - int(h_scale_T * h) > 0 else 0)
    br = (x + int(w_scale_R * w) if x + int(w_scale_R * w) < iou_img.shape[1] else iou_img.shape[1], y + int(
      h_scale_B * h) if y + int(h_scale_B * h) < iou_img.shape[0] else iou_img.shape[0])
    img = iou_img[tl[1]:br[1], tl[0]:br[0], :]
    tsfm = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

    batch_images.append(tsfm(img))

  input = torch.stack(batch_images, dim=0).to(device)

  with torch.no_grad():
    out = model(input).cpu()
    out = torch.sum(out, dim=0)
    prediction = torch.argmax(out)
  if prediction == 0:
    return 'scissors'
  elif prediction == 1:
    return 'forceps'
  elif prediction == 2:
    return 'gauze'
  else:
    return 'kidney-dish'

# 还原图片增强前桌面四个点的坐标


def restitute_pos(origin_w, origin_h, new_w, new_h, points):
  transformed_points = []
  for point in points:
    w, h = point
    new_x = (new_w / origin_w) * w
    new_y = (new_h / origin_h) * h
    transformed_points.append((new_x, new_y))
  return transformed_points

# 将投影图像画到图像上


def plot_proj_img(image, proj_img, palette, opacity):
  color_img = np.zeros(proj_img.shape[0] * proj_img.shape[1] * 3).reshape(
    [proj_img.shape[0], proj_img.shape[1], 3]).astype(np.uint8)
  for x in range(color_img.shape[0]):
    for y in range(color_img.shape[1]):
      color_img[x, y, :] = palette[proj_img[x, y]]
  return cv2.addWeighted(image, 1, color_img, opacity, 0)

# 点和直线的相对位置(上T, 下F)


def is_in(line, point) -> bool:
  start, end = line[0], line[1]
  cross_product = (end[0] - start[0]) * (point[1] - start[1]) - \
      (end[1] - start[1]) * (point[0] - start[0])
  return cross_product < 0

# 检查是否越界，0：不越界，1：进入桌面，2：离开桌面


def check_motion(track_id, position, record_dict):
  if track_id not in record_dict.keys():
    record_dict[track_id] = position
    return 0
  if record_dict[track_id] == position:
    return 0
  else:
    if position:
      record_dict[track_id] = position
      return 2
    else:
      record_dict[track_id] = position
      return 1

# 检查是否放下或拿起东西，0：无变化，1：放下东西，2：拿起东西


def check_pack(track_id, no_thing, thing_record_dict):
  if track_id not in thing_record_dict.keys():
    thing_record_dict[track_id] = no_thing
    return 0
  if thing_record_dict[track_id] == no_thing:
    return 0
  else:
    if no_thing == 0:
      thing_record_dict[track_id] = no_thing
      return 2
    else:
      thing_record_dict[track_id] = no_thing
      return 1

# 在图片上显示统计结果


def plot_picture(frame, show_dict):
  idx = 0
  for k, v in show_dict.items():
    if v < 0:
      color = [0, 0, 255]
    else:
      color = [0, 255, 0]
    frame = cv2.putText(frame, f"{k}: {v}", (50, 50 + (idx * 40)), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                        color, 2)
    idx += 1
  return frame


def frame_idx2time(frame_idx, fps):
  time = frame_idx / fps
  time = round(time) if time - round(time) < 0.5 else round(time) + 1
  if time < 60:
    return rf"00:{time:02d}"
  else:
    return rf"01:{time-60:02d}"


def main(video_path, output_path, init_count):
  # 固定随机数种子
  setup_seed(2023)

  # 选择设备
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 加载模型
  seg_model = build_seg_model(device)
  yolo_model = build_yolo_model()
  cls_model = build_cls_model(device)

  # 读取视频进行计算
  frame_idx = 0  # 用于保存当前循环是处理的第多少帧
  box_annotator = sv.BoxAnnotator(  # 配置box用于画图
      thickness=2,
      text_thickness=1,
      text_scale=0.5)
  record_dict = {}
  thing_record_dict = {}
  cap = cv2.VideoCapture(video_path)
  input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  input_fps = cap.get(cv2.CAP_PROP_FPS)
  input_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

  # 配置输出视频
  writer = None
  output_height = None
  output_width = None
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  output_fps = input_fps
  output_height = int(input_height)
  output_width = int(input_width)
  writer = cv2.VideoWriter(output_path, fourcc, output_fps,
                           (output_width, output_height))

  # 定义输出
  result_dict = {
      "00:00": init_count.copy(),
  }
  pos_dict = {

  }

  # tqdm进度条
  pbar = tqdm.tqdm(total=int(input_frame_count))

  # 存储近10帧图片用于做器械类别检测
  fq = FrameQueue(length=10)

  while True:
    flag, frame = cap.read()
    if not flag:
      break
    frame_idx += 1
    # print(frame_idx)

    # 如果是第一帧，调用seg模型，获取桌面投影
    if frame_idx == 1:
      final_img, points, proj_img = infer_seg_model(frame, seg_model, device)
    else:
      final_img = plot_proj_img(frame, proj_img, palette=[[0, 0, 0], [
                                255, 0, 0], [0, 255, 0], [0, 0, 255]], opacity=0.5)

    # 运行yolov8 tracking这张图片
    yolo_result = yolo_model.track(
      frame, persist=True, show=False, verbose=False)[0]

    # 使用supervision库解析yolov8 tracking结果
    if len(yolo_result.boxes.data) != 0:
      fq.add_item((frame, yolo_result.boxes))
      if yolo_result.boxes.id is not None:
        detections = sv.Detections.from_yolov8(yolo_result)
        detections.tracker_id = yolo_result.boxes.id.cpu().numpy().astype(int)
        class_ids = detections.class_id
        confidences = detections.confidence
        tracker_ids = detections.tracker_id
        labels = ['#{} {} {:.1f}'.format(tracker_ids[i], yolo_model.names[class_ids[i]], confidences[i] * 100) for i
                  in range(len(class_ids))]
        final_img = box_annotator.annotate(
          scene=final_img, detections=detections, labels=labels)

      # 遍历每个box，分析轨迹
      for idx in range(len(yolo_result.boxes.data)):
        try:
          track_id = int(yolo_result.boxes.id[idx])  # 尝试获取当前box的跟踪id
        except:
          continue

        x_center, y_center = int(yolo_result.boxes.xywh[idx, 0]), int(
          yolo_result.boxes.xywh[idx, 1])
        position = is_in((points[0], points[1]), (x_center, y_center))
        sign = check_motion(track_id, position, record_dict)

        # 判断手上是否拿有器械
        no_thing = int(yolo_result.boxes.cls[idx])
        thing_sign = check_pack(track_id, no_thing, thing_record_dict)

        if thing_sign == 1:
          region = proj_img[y_center, x_center]
          if region == 0:
            pass
          elif region == 1:
            pos_dict[frame_idx2time(frame_idx, input_fps)] = "left"
          elif region == 2:
            pos_dict[frame_idx2time(frame_idx, input_fps)] = "middle"
          else:
            pos_dict[frame_idx2time(frame_idx, input_fps)] = "right"

        if sign == 1:  # 进入桌面
          if int(yolo_result.boxes.cls[idx]) == 0:
            surgical_name = infer_cls_model(fq, cls_model, track_id, device)
            init_count[surgical_name] += 1
            result_dict[frame_idx2time(
              frame_idx, input_fps)] = init_count.copy()
          else:
            surgical_name = yolo_model.names[int(yolo_result.boxes.cls[idx])]
        elif sign == 2:  # 离开桌面
          if int(yolo_result.boxes.cls[idx]) == 0:
            surgical_name = infer_cls_model(fq, cls_model, track_id, device)
            init_count[surgical_name] -= 1
            result_dict[frame_idx2time(
              frame_idx, input_fps)] = init_count.copy()
        # else: # 不越界
        #     if int(yolo_result.boxes.cls[idx]) == 0:
        #         surgical_name = infer_cls_model(frame, cls_model, yolo_result.boxes, track_id, device)
        #     else:
        #         surgical_name = yolo_model.names[int(yolo_result.boxes.cls[idx])]

      final_img = plot_picture(final_img, init_count)

    if writer:
      if final_img.shape[0] != output_height or final_img.shape[
              1] != output_width:
        final_img = cv2.resize(final_img,
                                (output_width, output_height))
      writer.write(final_img)
    pbar.update(1)
  if writer:
    writer.release()
  cap.release()
  return result_dict, pos_dict


def load_label(label_path):
  with open(label_path, "r") as rf:
    label = json.load(rf)
  return label


def timestr2second(time_str):
  m, s = [int(i) for i in time_str.split(":")]
  return m * 60 + s


def get_error(count1, count2):
  err = 0
  for k in ["scissors", "forceps", "gauze", "kidney-dish"]:
    err += abs(count1[k] - count2[k])
  return err


def eval_metric(pred, label):
  threshold = 1
  all_err = 0
  get_event = 0
  for k, v in label.items():
    time = timestr2second(k)
    for kk, vv in pred.items():
      kk = timestr2second(kk)
      if kk in [t for t in range(time - threshold, time + threshold + 1)]:
        all_err += get_error(v, vv)
        get_event += 1
        break
  avg_err = all_err / get_event
  return get_event / len(label), avg_err


def eval_metric2(pred, label):
  threshold = 1
  all_err = 0
  get_event = 0
  for k, v in label.items():
    time = timestr2second(k)
    for kk, vv in pred.items():
      kk = timestr2second(kk)
      if kk in [t for t in range(time - threshold, time + threshold + 1)]:
        get_event += 1
        if v != vv:
          all_err += 1
        break
  avg_err = all_err / get_event
  return get_event / len(label), avg_err


def show_dict(pred_count, label_count):
  for k, v in pred_count.items():
    print(f"{k}: {v}")
  print("*" * 20)
  for k, v in label_count.items():
    print(f"{k}: {v}")


if __name__ == '__main__':
  video_dir = r"../../20230710/video"
  label_dir = r"../../20230710/label"
  label2_dir = r"../../20230710/label2"
  output_dir = r"./result"
  for v_name in os.listdir(video_dir):
    print(v_name)
    video_path = os.path.join(video_dir, v_name)
    label_path = os.path.join(label_dir, v_name.replace('.mp4', '.json'))
    label2_path = os.path.join(label2_dir, v_name.replace('.mp4', '.json'))
    output_path = os.path.join(output_dir, v_name)
    label_count = load_label(label_path)
    label_pos = load_label(label2_path)
    init_count = label_count["00:00"].copy()
    pred_count, pred_pos = main(video_path, output_path, init_count)
    # show_dict(pred_count, label_count)
    recall, avg_err = eval_metric(pred_count, label_count)
    print(fr"recall: {recall:.4f}, avg_err: {avg_err:.4f}")
    pos_recall, pos_avg_err = eval_metric2(pred_count, label_count)
    print(fr"pos_recall: {pos_recall:.4f}, pos_avg_err: {pos_avg_err:.4f}")
    print('*' * 20)
