import torch
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score


"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusion_matrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusion_matrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusion_matrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusion_matrix = np.zeros((self.numClass, self.numClass))


class MultiClassificationMetric:
    def __init__(self, num_classes, labels_name, normalize=True):
        """
		normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        self.num_classes = num_classes
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")
        self.labels_name = labels_name
        self.pred = []
        self.gt = []

    def addBatch(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[label, predict] += 1
        self.pred += predicts.tolist()
        self.gt += labels.tolist()

    def getMatrix(self,normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   # 百分比转换
            self.matrix=np.around(self.matrix, 2)   # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def getAcc(self):
        matrix = self.getMatrix(normalize=False)
        Acc = np.sum(np.diagonal(matrix))/np.sum(matrix)
        return Acc
    
    def getPrecision(self):
        Precision = precision_score(self.gt, self.pred, average="weighted")
        return Precision

    def getRecall(self):
        Recall = recall_score(self.gt, self.pred, average="weighted")
        return Recall

    def getF1Score(self):
        F1_score = f1_score(self.gt, self.pred, average="weighted")
        return F1_score

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def drawMatrix(self, file_name):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        # plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig(file_name, bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()

class BinaryClassificationMetric:
    def __init__(self, num_classes, labels_name, normalize=True):
        """
		normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        self.num_classes = num_classes
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")
        self.labels_name = labels_name

    def addBatch(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[label, predict] += 1

    def getMatrix(self,normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   # 百分比转换
            self.matrix=np.around(self.matrix, 2)   # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def getAcc(self):
        matrix = self.getMatrix(normalize=False)
        Acc = np.sum(np.diagonal(matrix))/np.sum(matrix)
        return Acc

    def getPrecision(self):
        matrix = self.getMatrix(normalize=False)
        Precision = matrix[0][0]/(matrix[0][0] + matrix[1][0])
        return Precision

    def getRecall(self):
        matrix = self.getMatrix(normalize=False)
        Recall = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        return Recall

    def getF1Score(self):
        Precision = self.getPrecision()
        Recall = self.getRecall()
        F1_score = (2 * Precision * Recall) / (Precision + Recall)
        return F1_score

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")


    def drawMatrix(self, file_name):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig(file_name, bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()


