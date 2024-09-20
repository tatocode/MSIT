from pathlib import Path
from models.FCN.fcn import *
from models.segnet.segnet import SegNet
from models.PSPNet.pspnet import PSPNet
from models.deeplabv3plus import modeling
from models.SETR import SETR
from models.UNet.unet_model import UNet
from models.refinenet.refinenet.refinenet_4cascade import RefineNet4Cascade
from models.Polynet.polynet import PolyNet
import torch, numpy as np, cv2, torchvision.transforms as T, json
from PIL import Image
import torchvision.transforms.functional as F
from torch import nn
import os
import albumentations as A

# 输入网络输出的粗糙mask,进行锐利化，输出锐利化后的图片和端点坐标
def sharp(mask, h, w):
    # 获取最大前景区域轮廓
    mask = mask * 255
    for u in range(5, 50):
        thresh = cv2.dilate(mask.astype(np.uint8), np.ones((u, u), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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

# 输入mask预测值和标签、端点预测值和标签，计算ACC,IoU,FWIoU,Dice,ED的值
def metric(mask_pre, mask_label, vertices_pre, vertices_label):
    # 获取CM
    mask = (mask_label >= 0) & (mask_label < 2)
    label = 2 * mask_label[mask] + mask_pre[mask]
    count = np.bincount(label, minlength=2*2)
    CM = count.reshape(2, 2)

    # acc
    acc = np.nanmean(np.diag(CM) / CM.sum(axis=1))

    # iou
    intersection = np.diag(CM)
    union = np.sum(CM, axis=1) + np.sum(CM, axis=0) - np.diag(CM)
    iou = np.nanmean(intersection / union)

    # fwiou
    freq = np.sum(CM, axis=1) / np.sum(CM)
    iu = np.diag(CM) / (np.sum(CM, axis=1) + np.sum(CM, axis=0) - np.diag(CM))
    fwiou = (freq[freq > 0] * iu[freq > 0]).sum()

    # dice
    add = np.sum(CM, axis=1) + np.sum(CM, axis=0)
    dice = np.nanmean(2 * intersection / add)

    # ed
    vertices_pre = torch.from_numpy(vertices_pre)
    vertices_label = torch.from_numpy(vertices_label)
    pdist = nn.PairwiseDistance(p=2)
    ed = np.mean(pdist(vertices_pre, vertices_label).numpy().astype(np.float32))

    return acc, iou, fwiou, dice, ed


# 调整输入端点的相对位置为
# 0 1
# 3 2
def adjust_position(vertices):
    # 1   2
    # 4   3
    # print(len(vertices))
    vertices = list(vertices)
    ret = [0]*4
    vertices.sort(key=lambda x:x[0]**2 + x[1]**2)
    ret[0] = vertices[0]
    ret[1], ret[3] = (vertices[1], vertices[2]) if vertices[1][0] > vertices[2][0] else (vertices[2], vertices[1])
    ret[2] = vertices[3]
    return np.array(ret, dtype=np.int32)


def rebuild_mask(prediction, img_size=512):
    prediction[prediction>=1] = 1
    prediction[prediction<=0] = 0
    bg = np.zeros((img_size, img_size), dtype=np.int32)
    contour = (prediction*img_size).detach().numpy().astype(np.int32)
    # print(f'contour type: {type(contour)}, contour shape: {contour.shape}')
    img = cv2.fillConvexPoly(bg, contour, 1)
    return torch.from_numpy(img)


def main(model_name, model, image_path, mask_path, label_path):
    global Acc, Dice, IoU, FWIoU, ED 
    
    # 读取图片
    if model_name == 'polynet':
        img_size = (512, 512)
    else:
        img_size = (512, 512)
    transform_T = T.Compose([
        T.Resize(img_size, interpolation=F._interpolation_modes_from_int(0)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform_T(Image.open(image_path)).unsqueeze(dim=0).to(device)
    mask = np.array(Image.open(mask_path))

    if model_name == 'pspnet':
        prediction, _ = model(img)
        out = torch.argmax(prediction.cpu(), dim=1).squeeze(dim=0).cpu().numpy().astype(np.int32)
    elif model_name == 'setr':
        prediction, _ = model(img, [1, 4, 8, 12])
        out = torch.argmax(prediction.cpu(), dim=1).squeeze(dim=0).cpu().numpy().astype(np.int32)
    elif model_name == 'polynet':
        # tgt_token, gt_token = tgt_token.to(device), gt_token.to(device) 
        _, point_predict = model(img)
        prediction = point_predict.squeeze(dim=0).cpu()
        out = rebuild_mask(prediction).numpy().astype(np.int32)
    else:
        out = torch.argmax(model(img).cpu(), dim=1).squeeze(dim=0).cpu().numpy().astype(np.int32)


    with open(label_path, 'r') as f:
        context = json.load(f)
        w, h = context["imageWidth"], context["imageHeight"]
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

    # 计算评估指标
    mask_sharp, vertices = sharp(out, h, w)
    if vertices is None:
        return     


    with open(label_path, 'r') as f:
        context = json.load(f)
        vertices_label = np.array(context['shapes'][0]['points'], dtype=np.int32)

    vertices_label = adjust_position(vertices=vertices_label)
    if vertices_label is None:
        return
    acc, iou, fwiou, dice, ed = metric(mask_sharp, mask, vertices, vertices_label)

    Acc.append(acc)
    IoU.append(iou)
    FWIoU.append(fwiou)
    Dice.append(dice)
    ED.append(ed)



def mean(item):
    return float(np.mean(np.array(item, dtype=np.float32)))

def get_avg(metric_lst):
    keys = metric_lst[0].keys()
    ret = metric_lst[0]
    for k in keys:
        for metric in metric_lst[1:]:
            ret[k] += metric[k]
    for k in keys:
        ret[k] /= 4
    return ret


if __name__ == '__main__':

    # eval

    # 统计参数
    Acc = []
    Dice = []
    IoU = []
    FWIoU = []
    ED = []


    # model_names = ['FCN', 'SETR', 'UNet', 'PSPNet', 'SegNet', 'DeepLabv3+', 'RefineNet', 'Polynet']
    model_names = ['Polynet']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for mn in model_names:
        print(f'{mn}:')
        temp_dir = f'./result/train-{mn}'
        root_dir = os.path.join(temp_dir, '230909_202857')

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

        metric_lst = []
        for fold in range(4):
            print(f'\t{fold}:', end=' ')
            fold_dir = os.path.join(root_dir, str(fold))
            weight_path = os.path.join(fold_dir, 'checkpoint', 'best.pth')
            dataset_file = os.path.join('/home/dell/tao/dataset_20230630/regional_division', f'fold_{fold}.txt')
            with open(dataset_file, 'r') as f:
                c = f.readlines()

            model.load_state_dict(torch.load(weight_path))
            model.to(device=device)
            model.eval()

            for _c in c:
                img_path = f'/home/dell/tao/dataset_20230630/regional_division/images/{_c.strip()}.png'
                label_path = f'/home/dell/tao/dataset_20230630/regional_division/annos/{_c.strip()}.json'
                mask_path = f'/home/dell/tao/dataset_20230630/regional_division/masks/{_c.strip()}.png'
                main(mn.lower(), model, img_path, mask_path, label_path)

            metric_lst.append({
                'mAcc': mean(Acc),
                'mIoU': mean(IoU),
                'mFWIoU': mean(FWIoU),
                'mDice': mean(Dice),
                'ED': mean(ED),
                'Det_rate': len(Acc)/len(c),
            })
            print('mAcc: {:.6f}, mIoU: {:.6f}, mFWIoU: {:.6f}, mDice: {:.6f}, ED: {:.6f}, Det_rate: {:.06f}'.format(mean(Acc), mean(IoU), mean(FWIoU), mean(Dice), mean(ED), len(Acc)/len(c)))

            Acc = []
            Dice = []
            IoU = []
            FWIoU = []
            ED = []
        print(f'\tAvg:{get_avg(metric_lst)}')
        print('*'*50)
    
    


    