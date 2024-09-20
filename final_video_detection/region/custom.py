import cv2
import numpy as np
import math
from region.plan import *

def deal_result(points, method, ori_size, out_size=(600, 1000)):
    height, width = out_size
    pts1 = np.array(points).astype(np.float32)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # 定义一张全黑图
    temp_img = np.zeros((height, width))

    W, H = ori_size

    # 画分区
    BEV_view = method(temp_img)
    _height, _weight = H, W
    _matrix = cv2.getPerspectiveTransform(pts2, pts1)
    view = cv2.warpPerspective(BEV_view, _matrix, (_weight, _height)).astype(np.uint8)
    # print(f'view shape: {view.shape}')
    # print('ok')
    return view

    
