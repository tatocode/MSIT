import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torchvision.transforms as T

import albumentations as A
import os, torch, torchvision
from torch import nn
from region.Polynet.polynet import PolyNet
from region.plan import *
from region.custom import deal_result


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



def transform_points(o_w, o_h, n_w, n_h, points):
    transformed_points = []
    for point in points:
        x, y = point
        new_x = (n_w / o_w) * x
        new_y = (n_h / o_h) * y
        transformed_points.append((new_x, new_y))
    return transformed_points

def show_result(image, result, palette, opacity):
    color_img = np.zeros(result.shape[0] * result.shape[1] *3).reshape([result.shape[0], result.shape[1], 3]).astype(np.uint8)
    for x in range(color_img.shape[0]):
        for y in range(color_img.shape[1]):
            color_img[x, y, :] = palette[result[x, y]]
    return cv2.addWeighted(image, 1, color_img, opacity, 0)


# return 3 results, first is the frame that add region mask with color, 
# second is four points of desk contour, last is region mask with color, 
# but not add to origin image.
def get_region(frame, model, plan, device, palette, opacity):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    image = frame.copy()
    image = T.ToPILImage()(image)
    T_image = T.Compose([
                    T.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                ])
    image = T_image(image).type(torch.float32)
    image = image.unsqueeze(dim=0)
    with torch.no_grad():
        image = image.to(device, dtype=torch.float32)

        # polynet
        _, point_out = model(image)
        # print('point_out sum: ', torch.sum(point_out))
        point_out = point_out.squeeze(dim=0).cpu()

        point_out[point_out < 0] = 0
        point_out[point_out > 1] = 1
        point_out *= 512

        # print(f'point_out shape: {point_out.shape}')
        points = transform_points(512, 512, frame.shape[1], frame.shape[0], points=point_out)


        # prediction = torch.argmax(model(image).cpu(), dim=1)
    deal_ret = deal_result(points, plan, ori_size=(frame.shape[1], frame.shape[0]))
    return show_result(frame, deal_ret, palette=palette, opacity=opacity), points, deal_ret

# 点在直线哪一边(上T, 下F)
def is_in(line, point)->bool:
    start, end = line[0], line[1]
    cross_product = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
    return cross_product < 0

# 0: no motion
# 1: enter desk
# 2: leave deak
def check_motion(track_id, position):
    global record_dict
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

def get_cls(model, fq, track_id, device):
    fs = fq.get_data()
    use_f = []
    for f, boxes in fs:
        try:
            idx = list(boxes.id.type(torch.int8)).index(track_id)
            use_f.append((f, idx, boxes))
        except:
            continue
    
    img_list = []
    for frame, idx, boxes in use_f:
        x, y, w, h = boxes.xywh[idx]
        x, y = int(x), int(y)
        tl = (x-int(w) if x-int(w) > 0 else 0, y-int(h) if y-int(h) > 0 else 0)
        br = (x+int(1.5*w) if x+int(1.5*w) < frame.shape[1] else frame.shape[1], y+int(1.5*h) if y+int(1.5*h) < frame.shape[0] else frame.shape[0])
        image = frame[tl[1]:br[1], tl[0]:br[0], :]
        # cv2.imwrite('a.png', image)
        A_transform = A.Compose([
            A.Resize(224, 224),
        ])
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        T_image = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        img_list.append(T_image(A_transform(image=image)['image']))
    inp = torch.stack(img_list, dim=0).to(device)
    # inp = T_image(A_transform(image=image)['image']).unsqueeze(dim=0).to(device)
    with torch.no_grad():
        out = model(inp).cpu()
        out = torch.sum(out, dim=0)
        # print(f'out shape: {out.shape}')
        prediction = torch.argmax(out)
        # print(prediction.shape)
        if prediction == 0:
            return 'scissors'
        elif prediction == 1:
            return 'forceps'
        elif prediction == 2:
            return 'gauze'
        else:
            return 'kidney-dish'

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

if __name__ == '__main__':
    for video_name in os.listdir('../../20230710'):
        VIDEO_PATH = os.path.join(r"../../20230710", video_name)
        # VIDEO_PATH = '/home/dell/tao/desk_20230630/hand_detection/yolov8/video/video_01.mp4'
        VIDEO_OUTPUT = os.path.join(r"result", video_name)


        # seg_model = 'RefineNet'
        seg_model = 'PolyNet'
        seg_checkpoint = 'region/checkpoint/best.pth'
        plan = plan2
        palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
        opacity = 0.5

        det_model = 'YOLOv8'
        det_checkpoint = 'hand/checkpoint/best.pt'
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5)
        record_dict = {}
        monitor = {}

        cls_model = 'densenet201'
        cls_checkpoint = 'instrument/checkpoint/best.pth'

        show_dict = {
            'scissors': 3,
            'forceps': 1,
            'gauze': 3,
            'kidney-dish': 1
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        if seg_model == 'PolyNet':
            model = PolyNet(img_size=512)
            model.load_state_dict(torch.load(seg_checkpoint))
            model.to(device)
            model.eval()

        if det_model == 'YOLOv8':
            model2 = YOLO(det_checkpoint)
            det_ret = model2.track(VIDEO_PATH, show=False, stream=True, verbose=False, device=device, conf=0.5)

        if cls_model == 'densenet201':
            model3 = torchvision.models.densenet201()
            model3.classifier = nn.Linear(model3.classifier.in_features, 4)
            model3.load_state_dict(torch.load(cls_checkpoint))
            model3.to(device)
            model3.eval()

        # build input video
        cap = cv2.VideoCapture(VIDEO_PATH)
        assert (cap.isOpened())
        input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        input_fps = cap.get(cv2.CAP_PROP_FPS)

        # init output video
        writer = None
        output_height = None
        output_width = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = input_fps
        output_height = int(input_height)
        output_width = int(input_width)
        writer = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, output_fps, (output_width, output_height))

        zone = None
        deal_ret = None

        # 存储近10帧图片用于做器械类别检测
        fq = FrameQueue(length=5)

        while True:
            flag, frame = cap.read()
            if not flag:
                break
            
            if zone is None or deal_ret is None:
                seg_frame, points, deal_ret = get_region(frame, model, plan, device, palette, opacity)
                zone = points
            # zone = np.array(points, dtype=np.int32)
            else:
                seg_frame = show_result(frame, deal_ret, palette=palette, opacity=opacity)

            # seg_frame = cv2.line(seg_frame, zone[0], zone[1], [0, 0, 255], 10)

            result = next(det_ret)
            if len(result.boxes.data) != 0:
                fq.add_item((frame, result.boxes))
                if result.boxes.id is not None:
                    detections = sv.Detections.from_yolov8(result)
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                    class_ids = detections.class_id
                    confidences = detections.confidence
                    tracker_ids = detections.tracker_id
                    labels = ['#{} {} {:.1f}'.format(tracker_ids[i], model2.names[class_ids[i]], confidences[i] * 100) for i
                            in range(len(class_ids))]
                    det_frame = box_annotator.annotate(scene=seg_frame, detections=detections, labels=labels)

                for idx in range(len(result.boxes.data)):
                    try:
                        track_id = int(result.boxes.id[idx])
                    except:
                        continue

                    x_center, y_center = int(result.boxes.xywh[idx, 0]), int(result.boxes.xywh[idx, 1])
                    position = is_in((zone[0], zone[1]), (x_center, y_center))

                    sign = check_motion(track_id, position)
                    # print('{} hand sign: {}'.format(track_id, sign))

                    if sign == 1:
                        # if track_id not in monitor.keys():
                        #     monitor[track_id] = ['enter']
                        if int(result.boxes.cls[idx]) == 0:
                            # mark = get_cls(model3, frame, result.boxes, idx, device)
                            mark = get_cls(model3, fq, track_id, device)
                            show_dict[mark] += 1
                        else:
                            mark = model2.names[int(result.boxes.cls[idx])]
                        # monitor[track_id].append(mark)
                    elif sign == 2:
                        # assert track_id in monitor.keys(), 'failed'
                        # if track_id not in monitor.keys(): 
                        #     print(f'happend leave, and no monitor this id: {track_id}, current monitor keys is: {monitor.keys()}')
                        #     continue
                        if int(result.boxes.cls[idx]) == 0:
                            # mark = get_cls(model3, frame, result.boxes, idx, device)
                            mark = get_cls(model3, fq, track_id, device)
                            show_dict[mark] -= 1
                        # monitor[track_id].append('leave')
                    else:
                        # if track_id in monitor.keys() and monitor[track_id][-1] != 'leave':
                        if int(result.boxes.cls[idx]) == 0:
                            # mark = get_cls(model3, frame, result.boxes, idx, device)
                            mark = get_cls(model3, fq, track_id, device)
                        else:
                            mark = model2.names[int(result.boxes.cls[idx])]
                            # monitor[track_id].append(mark)
                try:
                    ret_frame = plot_picture(det_frame, show_dict)
                except:
                    ret_frame = plot_picture(seg_frame, show_dict)
                if writer:
                    if ret_frame.shape[0] != output_height or ret_frame.shape[
                            1] != output_width:
                        draw_img = cv2.resize(ret_frame,
                                            (output_width, output_height))
                    writer.write(ret_frame)
        if writer:
            writer.release()
        cap.release()
        # print(monitor)
