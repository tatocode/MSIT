import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

config = {
    "font.family":'serif',
    "font.size": 18,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def drawMatrix(matrix, file_name):
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap=plt.cm.YlGnBu)  # 仅画出颜色格子，没有值
    # plt.title("Normalized confusion matrix")  # title
    plt.xlabel("Predicted Label", size=15)
    plt.ylabel("Ground Truth", size=15)
    plt.yticks(range(4), ['surgical\nscissors', 'forceps', 'occlusive\ndressings', 'kidney\ndish'], size=12)  # y轴标签
    plt.xticks(range(4), ['surgical\nscissors', 'forceps', 'occlusive\ndressings', 'kidney\ndish'], size=12)  # x轴标签

    for x in range(4):
        for y in range(4):
            if matrix[y, x] == 0:
                value = 0
            elif matrix[y, x] == 1:
                value = 100
            else:
                value = int(matrix[y, x] * 100)  # 数值处理
            if x == y:
                if value == 0:
                    plt.text(x, y, str(value), verticalalignment='center', horizontalalignment='center', size=15, color='w', fontdict={"family": "Times New Roman"})  # 写值
                else:
                    plt.text(x, y, str(value)+'%', verticalalignment='center', horizontalalignment='center', size=15, color='w', fontdict={"family": "Times New Roman"})  # 写值
            else:
                if value == 0:
                    plt.text(x, y, str(value), verticalalignment='center', horizontalalignment='center', size=15, fontdict={"family": "Times New Roman"})  # 写值
                else:
                    plt.text(x, y, str(value)+'%', verticalalignment='center', horizontalalignment='center', size=15, fontdict={"family": "Times New Roman"})  # 写值

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.colorbar()  # 色条
    plt.savefig(file_name, bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
    plt.show()
    

if __name__ == '__main__':
    file_name = r"/mnt/d/04_research/01_desk/code/instrument_recognition/result/ToolNet_r50.png"
    # file_name = r"C:\Users\Tatocode\Documents\desk\code\instrument_recognition\result\ToolNet_r50.png"
    matrix = np.array([[0.94, 0.05, 0.01, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    drawMatrix(matrix, file_name)