import cv2
import numpy as np
import math
from plan import *

def dist(point1, point2):
    x0, y0 = point1
    x1, y1 = point2
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

def adjust_position(cps):
    # 1   4
    # 2   3
    print(cps)
    result = [0, 0, 0, 0]
    cps_np = np.array(cps)
    W_mean, H_mean = np.mean(cps_np, axis=0)
    for W, H in cps:
        if W < W_mean:
            if H < H_mean:
                if result[0] == 0:
                    result[0] = (W, H)
            else:
                if result[1] == 0:
                    result[1] = (W, H)
        else:
            if H < H_mean:
                result[3] = (W, H)
            else:
                result[2] = (W, H)
    if 0 in result:
        return None
    return result

def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取直线1的第一个点坐标
    y1 = line1[1]
    x2 = line1[2]  # 取直线1的第二个点坐标
    y2 = line1[3]

    x3 = line2[0]  # 取直线2的第一个点坐标
    y3 = line2[1]
    x4 = line2[2]  # 取直线2的第二个点坐标
    y4 = line2[3]

    try:
        if x2 - x1 == 0:  # L1 直线斜率不存在
            k1 = None
            b1 = 0
        else:
            k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
            b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键

        if (x4 - x3) == 0:  # L2直线斜率不存在操作
            k2 = None
            b2 = 0
        else:
            k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
            b2 = y3 * 1.0 - x3 * k2 * 1.0

        if k1 is None and k2 is None:  # L1与L2直线斜率都不存在，两条直线均与y轴平行
            if x1 == x3:  # 两条直线实际为同一直线
                return [x1, y1]  # 均为交点，返回任意一个点
            else:
                return None  # 平行线无交点
        elif k1 is None and k2 is None:  # 若L2与y轴平行，L1为一般直线，交点横坐标为L2的x坐标
            x = x3
            y = k1 * x * 1.0 + b1 * 1.0
        elif k1 is None and k2 is not None:  # 若L1与y轴平行，L2为一般直线，交点横坐标为L1的x坐标
            x = x1
            y = k2 * x * 1.0 + b2 * 1.0
        else:  # 两条一般直线
            if k1 == k2:  # 两直线斜率相同
                if b1 == b2:  # 截距相同，说明两直线为同一直线，返回任一点
                    return [x1, y1]
                else:  # 截距不同，两直线平行，无交点
                    return None
            else:  # 两直线不平行，必然存在交点
                x = (b2 - b1) * 1.0 / (k1 - k2)
                y = k1 * x * 1.0 + b1 * 1.0
        return [x, y]
    except:
        return None

def deal_result(result, method, out_size=(600, 1000), threshold=0.2):

    err_return = np.zeros_like(result)

    img_binary = np.squeeze(np.array(result)*255).astype(np.uint8)
    H, W = img_binary.shape
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    area_max = 0
    index = -1
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area_max < area:
            area_max = area
            index = idx
    for idx, contour in enumerate(contours):
        if idx != index:
            cv2.drawContours(img_binary, contours, idx, (0, 0, 0), -1)


    lsd = cv2.createLineSegmentDetector(0, 0.08)  # 直线检测器

    # try:
    lines = lsd.detect(img_binary)[0]
    line_point = [((int(ls[0][0]), int(ls[0][1])), (int(ls[0][2]), int(ls[0][3])),
                        dist((int(ls[0][0]), int(ls[0][1])), (int(ls[0][2]), int(ls[0][3])))) for ls in lines]
    if len(line_point) < 4:
        print(r'This image cannot parse lines < 4')
        return err_return
    else:
        line_point = sorted(line_point, key=lambda x: x[2], reverse=True)
        points = [(line_point[idx][0], line_point[idx][1]) for idx in range(4)]
        cps = []
        for line1_idx in range(len(points) - 1):
            for line2_idx in range(line1_idx + 1, len(points)):
                line1 = points[line1_idx][0] + points[line1_idx][1]
                line2 = points[line2_idx][0] + points[line2_idx][1]
                cp = cross_point(line1, line2)
                if cp is not None and (0 - threshold) * W < cp[0] < (1 + threshold) * W and (
                        0 - threshold) * H < cp[1] < (1 + threshold) * H:
                    cps.append([int(cp[0]), int(cp[1])])
        print(f'cps length: {len(cps)}')
        if len(cps) == 4:
            cps = adjust_position(cps)
            if cps is None:
                return err_return

            height, width = out_size
            pts1 = np.array(cps).astype(np.float32)
            pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

            # 定义一张全黑图
            temp_img = np.zeros((height, width))

            # 画分区
            BEV_view = method(temp_img)
            _height, _weight = H, W
            _matrix = cv2.getPerspectiveTransform(pts2, pts1)
            view = cv2.warpPerspective(BEV_view, _matrix, (_weight, _height)).astype(np.uint8)
            print('ok')
            return view[np.newaxis, :]

        else:
            print(f'Error: cps length not 4!')
            return err_return
    # except:
    #     return err_return

    
