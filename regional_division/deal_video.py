from argparse import ArgumentParser

import cv2

import torch
from models.refinenet.refinenet.refinenet_4cascade import RefineNet4Cascade
# from mmseg.apis import inference_segmentor, init_segmentor
# from mmseg.core.evaluation import get_palette
from custom import deal_result
from plan import *
import numpy as np
import torchvision.transforms as T
from PIL import Image



def init_segmentor(model_name: str, checkpoint: str, device: str='cuda:0'):
    if model_name == 'RefineNet':
        model = RefineNet4Cascade(input_shape=(3, 512), num_classes=2)
        model.load_state_dict(torch.load(checkpoint))
        model.to(device)
        return model
    
def inference_segmentor(model, image):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    image = T.ToPILImage()(image)
    T_image = T.Compose([
                    T.Resize((512, 512)),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                ])
    image = T_image(image).type(torch.float32)
    image = image.unsqueeze(dim=0)
    # print(image.shape)
    model.eval()
    with torch.no_grad():
        image = image.to(torch.device('cuda:0'), dtype=torch.float32)
        # print(image.shape)
        prediction = torch.argmax(model(image).cpu(), dim=1)
    return prediction

def show_result(image, result, palette, opacity):
    result = cv2.resize(result[0], image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    color_img = np.zeros(result.shape[0] * result.shape[1] *3).reshape([result.shape[0], result.shape[1], 3]).astype(np.uint8)
    for x in range(color_img.shape[0]):
        for y in range(color_img.shape[1]):
            color_img[x, y, :] = palette[result[x, y]]
    # print(type(image),type(color_img), image.dtype, color_img.dtype)
    return cv2.addWeighted(image, 1, color_img, opacity, 0)

def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file or webcam id')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output-file', default=None, type=str, help='Output video file path')
    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--output-fps', default=-1, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    assert args.show or args.output_file, \
        'At least one output should be enabled.'

    # build the model from a config file and a checkpoint file
    model = init_segmentor('RefineNet', args.checkpoint, device=args.device)

    # build input video
    cap = cv2.VideoCapture(args.video)
    assert (cap.isOpened())
    input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # init output video
    writer = None
    output_height = None
    output_width = None
    if args.output_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = args.output_height if args.output_height > 0 else int(
            input_height)
        output_width = args.output_width if args.output_width > 0 else int(
            input_width)
        writer = cv2.VideoWriter(args.output_file, fourcc, output_fps,
                                 (output_width, output_height), True)

    # start looping
    try:
        before = np.zeros_like((output_height, output_width))
        while True:
            flag, frame = cap.read()
            if not flag:
                break

            # test a single image
            result = inference_segmentor(model, frame)
            # print(fr'result: {np.unique(np.array(result))}')
            # 自定义opencv处理
            result = deal_result(result, plan2)
            if np.sum(result) == 0 and np.sum(before) != 0:
                result = before
            elif np.sum(result) != 0:
                before = result

            # blend raw image and prediction
            draw_img = show_result(
                frame,
                result,
                palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
                opacity=args.opacity)

            # if args.show:
            #     cv2.imshow('video_demo', draw_img)
            #     cv2.waitKey(args.show_wait_time)
            if writer:
                if draw_img.shape[0] != output_height or draw_img.shape[
                        1] != output_width:
                    draw_img = cv2.resize(draw_img,
                                          (output_width, output_height))
                writer.write(draw_img)
    finally:
        if writer:
            writer.release()
        cap.release()


if __name__ == '__main__':
    main()