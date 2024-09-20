from pathlib import Path
from models.FCN.fcn import *
from models.segnet.segnet import SegNet
from models.PSPNet.pspnet import PSPNet
from models.deeplabv3plus import modeling
from models.SETR import SETR
from models.UNet.unet_model import UNet
from models.refinenet.refinenet.refinenet_4cascade import RefineNet4Cascade
from models.Polynet.polynet import PolyNet
import torch
import numpy as np
import cv2
import torchvision.transforms as T
import json
from PIL import Image
import torchvision.transforms.functional as F
from torch import nn
import random
import os


# 输入网络输出的粗糙mask,进行锐利化，输出锐利化后的图片和端点坐标
def sharp(mask, h, w):
    # 获取最大前景区域轮廓
    mask = mask * 255
    for u in range(5, 50):
        thresh = cv2.dilate(mask.astype(np.uint8), np.ones(
            (u, u), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        area_max, index = 0, 0
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area_max < area:
                area_max = area
                index = idx
        cnt = contours[index]

        epsilon = 0.001 * cv2.arcLength(cnt, True)
        while True:
            epsilon += epsilon
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) <= 4:
                break
        if len(approx) == 4:
            break
    if len(approx) != 4:
        return np.zeros((h, w), dtype=np.int32), None
    approx = np.squeeze(approx, axis=1)

    background = np.zeros((h, w), dtype=np.int32)
    cv2.fillPoly(background, [approx], 1)
    return background, adjust_position(approx)


# 调整输入端点的相对位置为
# 0 1
# 3 2
def adjust_position(vertices):
    # 1   2
    # 4   3
    # print(len(vertices))
    vertices = list(vertices)
    ret = [0] * 4
    vertices.sort(key=lambda x: x[0] ** 2 + x[1] ** 2)
    ret[0] = vertices[0]
    ret[1], ret[3] = (vertices[1], vertices[2]) if vertices[1][0] > vertices[2][0] else (
        vertices[2], vertices[1])
    ret[2] = vertices[3]
    return np.array(ret, dtype=np.int32)


def rebuild_mask(prediction, img_size=512):
    prediction[prediction >= 1] = 1
    prediction[prediction <= 0] = 0
    bg = np.zeros((img_size, img_size), dtype=np.int32)
    contour = (prediction * img_size).detach().numpy().astype(np.int32)
    # print(f'contour type: {type(contour)}, contour shape: {contour.shape}')
    img = cv2.fillConvexPoly(bg, contour, 1)
    return torch.from_numpy(img)


def main(model_name, model, image_path, mask_path, label_path):
    # 读取图片
    img_size = (512, 512)
    transform_T = T.Compose([
        T.Resize(img_size, interpolation=F._interpolation_modes_from_int(0)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_T(Image.open(image_path)).unsqueeze(dim=0).to(device)
    mask = np.array(Image.open(mask_path))

    if model_name.lower() == 'pspnet':
        prediction, _ = model(img)
        out = torch.argmax(prediction.cpu(), dim=1).squeeze(
            dim=0).cpu().numpy().astype(np.int32)
    elif model_name.lower() == 'setr':
        prediction, _ = model(img, [1, 4, 8, 12])
        out = torch.argmax(prediction.cpu(), dim=1).squeeze(
            dim=0).cpu().numpy().astype(np.int32)
    elif model_name.lower() == 'polynet':
        # tgt_token, gt_token = tgt_token.to(device), gt_token.to(device)
        _, point_predict = model(img)
        prediction = point_predict.squeeze(dim=0).cpu()
        out = rebuild_mask(prediction).numpy().astype(np.int32)
    else:
        out = torch.argmax(model(img).cpu(), dim=1).squeeze(
            dim=0).cpu().numpy().astype(np.int32)

    if model_name.lower() != "polynet":
        out, _ = sharp(out, 512, 512)

    with open(label_path, 'r') as f:
        context = json.load(f)
        w, h = context["imageWidth"], context["imageHeight"]
    out = cv2.resize(
        out, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # mask = (cv2.imread(mask_path)[:, :, 0] / 255).astype(np.uint8)
    img = cv2.imread(image_path)

    # r = out + mask
    # ret = np.zeros_like(img)
    # for i in range(ret.shape[0]):
    #     for j in range(ret.shape[1]):
    #         if r[i, j] == 0:
    #             ret[i, j, :] = [0, 0, 0]
    #         elif r[i, j] == 1:
    #             ret[i, j, :] = [0, 0, 255]
    #         else:
    #             ret[i, j, :] = [0, 255, 0]

    # out = cv2.imread(mask_path, -1)
    # print(out.shape, type(out))

    out = np.stack([out, out, out], axis=2) * \
        np.array([255, 255, 47]).reshape((1, 1, 3))

    # r = cv2.addWeighted(img, 0.5, ret.astype(np.uint8), 0.5, 0)
    r = cv2.addWeighted(img, 0.6, out.astype(np.uint8), 0.4, 0)
    if not os.path.exists(f"temp2/{model_name}"):
        os.makedirs(f"temp2/{model_name}")
    cv2.imwrite(f'temp2/{model_name}/{image_path.split(os.sep)[-1]}', r)
    # raise Exception('error')


if __name__ == '__main__':

    # 随机数种子
    SEED = 2023
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # model_names = ['FCN', 'SETR', 'UNet', 'PSPNet', 'SegNet', 'DeepLabv3+', 'RefineNet', 'Polynet']
    model_names = ['FCN', 'UNet', 'SegNet', 'DeepLabv3+', 'Polynet']
    # model_names = ['mask']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for mn in model_names:

        test_dir = "for_plot"
        test_imgs_dir = os.path.join(test_dir, "images")
        test_annos_dir = os.path.join(test_dir, "annos")
        test_masks_dir = os.path.join(test_dir, "masks")

        if mn.lower() == 'mask':
            for i, a, m in zip(sorted(os.listdir(test_imgs_dir)), sorted(os.listdir(test_annos_dir)),
                               sorted(os.listdir(test_masks_dir))):
                i = os.path.join(test_imgs_dir, i)
                a = os.path.join(test_annos_dir, a)
                m = os.path.join(test_masks_dir, m)
                img = cv2.imread(i)
                mask = (cv2.imread(m)[:, :, 0] / 255).astype(np.uint8)
                out = np.stack([mask, mask, mask], axis=2) * \
                    np.array([255, 255, 47]).reshape((1, 1, 3))
                r = cv2.addWeighted(img, 0.6, out.astype(np.uint8), 0.4, 0)
                if not os.path.exists(f"temp2/{mn}"):
                    os.makedirs(f"temp2/{mn}")
                cv2.imwrite(f'temp2/{mn}/{i.split(os.sep)[-1]}', r)
            break

        print(f'{mn}:')
        temp_dir = f'./result/train-{mn}'
        root_dir = os.path.join(temp_dir, sorted(os.listdir(temp_dir))[-1])

        # build model
        if mn.lower() == 'fcn':
            vgg_model = VGGNet(requires_grad=True)
            model = FCN8s(pretrained_net=vgg_model, n_class=2)
        elif mn.lower() == 'setr':
            _, model = SETR.SETR_MLA_S(dataset='desk')
        elif mn.lower() == 'unet':
            model = UNet(3, 2)
        elif mn.lower() == 'pspnet':
            model = PSPNet(n_classes=2, backend='resnet50', pretrained=False)
        elif mn.lower() == 'segnet':
            model = SegNet(3, 2)
        elif mn.lower() == 'deeplabv3+':
            model = modeling.deeplabv3plus_resnet101(2, 16, False)
        elif mn.lower() == 'refinenet':
            model = RefineNet4Cascade(input_shape=(3, 512), num_classes=2)
        elif mn.lower() == 'polynet':
            model = PolyNet(img_size=512)
        else:
            raise Exception('error')

        fold_dir = os.path.join(root_dir, "0")
        weight_path = os.path.join(fold_dir, 'checkpoint', 'best.pth')

        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        model.eval()

        for i, a, m in zip(sorted(os.listdir(test_imgs_dir)), sorted(os.listdir(test_annos_dir)), sorted(os.listdir(test_masks_dir))):
            i = os.path.join(test_imgs_dir, i)
            a = os.path.join(test_annos_dir, a)
            m = os.path.join(test_masks_dir, m)
            main(mn, model, i, m, a)
