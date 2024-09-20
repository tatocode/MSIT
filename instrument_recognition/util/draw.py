import math

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def save_val_prediction(image: np.array, mask: np.array, prediction: np.array, cmap: np.array, save_path: str) -> None:
    assert image.shape[:-1] == mask.shape == prediction.shape, f'验证集 image({image.shape[-2:]}), mask({mask.shape}), ' \
                                                               f'predict({prediction.shape}) 形状大小不一致'
    result = np.hstack([image[:, :, ::-1], cmap[mask], cmap[prediction]])
    cv2.imwrite(save_path, result)


def save_test_prediction(image: np.array, prediction: np.array, cmap: np.array, save_path: str) -> None:
    assert image.shape[:-1] == prediction.shape, f'测试集 image({image.shape[-2:]}), ' \
                                                               f'predict({prediction.shape}) 形状大小不一致'
    result = cv2.addWeighted(image, 0.6, cmap[prediction], 0.4, 0, dtype=cv2.CV_32F).astype(np.uint8)
    cv2.imwrite(save_path, result)


class Record:
    def __init__(self, epoch_num: int, evaluation: list, save_path: str):
        self.evaluation = evaluation
        self.epoch_num = epoch_num
        self.save_path = save_path
        self.loss_list = []
        self.metric_dict_list = []

    def add_data_mean(self, loss: float, metric_dict: dict):

        assert sorted(metric_dict.keys()) == sorted(self.evaluation), f'传入验证结果和预定不一致'

        self.loss_list.append(loss)
        self.metric_dict_list.append(metric_dict)

    def draw(self):
        assert len(self.loss_list) == len(self.metric_dict_list), f'训练和验证次数不统一，无法记录训练过程'
        assert not len(self.loss_list) == 0, f'训练还未开始，无法记录训练过程'

        loss_max = math.ceil(self.loss_list[0])
        use_loss = [loss / loss_max for loss in self.loss_list]

        draw_dict = {}
        for el in self.evaluation:
            draw_dict[el] = [dc[el] for dc in self.metric_dict_list]

        x = [i + 1 for i in range(len(self.loss_list))]
        y_loss = use_loss

        plt.figure(figsize=(18, 8), dpi=80)
        plt.xticks(range(0, self.epoch_num + 1, self.epoch_num//20))
        plt.yticks([i/10 for i in range(1, 11)])

        plt.xlim(0, self.epoch_num)
        plt.ylim(0, 1)

        plt.xlabel('Epoch')
        plt.ylabel('Parameter')

        plt.grid(alpha=0.3, linestyle=':')

        plt.plot(x, y_loss, c='red', linestyle='-', label=f'Loss(*{loss_max:.2f})')
        for el in self.evaluation:
            plt.plot(x, draw_dict[el], linestyle='--', label=el)

        plt.legend(loc=0)
        plt.savefig(self.save_path)

def ro_curve(y_pred, y_label, figure_file, method_name='AUC'):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    '''
    plt.figure(figsize=(12, 8), dpi=300)
    plt.rcParams['font.size'] = 20
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[0], tpr[0], _ = roc_curve(y_label, y_pred)
    roc_auc[0] = auc(fpr[0], tpr[0])
    lw = 2
    plt.plot(fpr[0], tpr[0],
         lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xticks(font="Times New Roman",size=18,weight="bold")
    # plt.yticks(font="Times New Roman",size=18,weight="bold")
    fontsize = 20
    plt.xlabel('False Positive Rate', fontsize = fontsize)
    plt.ylabel('True Positive Rate', fontsize = fontsize)
    #plt.title('Receiver Operating Characteristic Curve', fontsize = fontsize)
    plt.legend(loc="lower right")
    plt.savefig(figure_file)
    plt.close()
    return
