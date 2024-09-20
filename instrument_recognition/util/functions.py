import datetime


def get_datetime_str(style='dt'):
    '''
    获取当前时间字符串
    :param style: 'dt':日期+时间；'date'：日期；'time'：时间
    :return: 当前时间字符串
    '''
    cur_time = datetime.datetime.now()

    date_str = cur_time.strftime('%y%m%d')
    time_str = cur_time.strftime('%H%M%S')

    if style == 'date':
        return date_str
    elif style == 'time':
        return time_str
    else:
        return date_str + '_' + time_str


# if __name__ == '__main__':
#     label_name = ['cat', 'dog']
#     d = DrawConfusionMatrix(labels_name=label_name, normalize=True)
#     iter = 100
#     for i in range(iter):
#         pred = np.random.random((10))
#         pred[pred>0.5]=1
#         pred[pred<0.5]=0
#         label = np.random.random((10))
#         label[label>0.5] = 1
#         label[label<0.5]=0
#         d.update(np.array(pred, dtype=np.int64), np.array(label, dtype=np.int64))
#     d.drawMatrix('a.png')

