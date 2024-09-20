import os

def str2dict(metric):
    ret = {}
    metrics = metric.strip().split(', ')
    for m in metrics:
        k, v = m.split(':')
        ret[k.strip()] = float(v.strip())
    return ret


def cal_result(log_fine):
    with open(log_fine, 'r') as f:
        c = f.readlines()
    max_IoU = 0
    SOTA_metric = {}
    for _c in c[1::2]:
        metric = _c.strip().split('\t\t\t')[1].strip()
        metric_dict = str2dict(metric=metric)
        if max_IoU == 0 or len(SOTA_metric) == 0:
            max_IoU = metric_dict['mIoU']
            SOTA_metric = metric_dict
            continue
        if max_IoU < metric_dict['mIoU']:
            max_IoU = metric_dict['mIoU']
            SOTA_metric = metric_dict
    return max_IoU, SOTA_metric

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
    
    models = ['FCN', 'SETR', 'UNet', 'PSPNet', 'SegNet', 'DeepLabv3+', 'RefineNet', 'Polynet']

    # for m_name in models:
    #     temp_dir = f'../result/train-{m_name}'
    #     root_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
    #     for fold in range(4):
    #         fold_dir = os.path.join(root_dir, str(fold))
    #         with open(os.path.join(fold_dir, 'log.txt'), 'r') as f:
    #             c = f.readlines()
    #             print(f'Number of lines is : {len(c)}')
    #         with open(os.path.join(fold_dir, 'log_fine.txt'), 'w') as f:
    #             f.writelines(c[:400])

    for m_name in models:
        temp_dir = f'../result/train-{m_name}'
        root_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        print(f'{m_name}:')
        metric_lst = []
        for fold in range(4):
            fold_dir = os.path.join(root_dir, str(fold))
            max_IoU, SOTA_metric = cal_result(os.path.join(fold_dir, 'log_fine.txt'))
            metric_lst.append(SOTA_metric)
            print(f'\t{fold}: {SOTA_metric}')
        print(f'\tavg: {get_avg(metric_lst=metric_lst)}')
        print('********************')
        
