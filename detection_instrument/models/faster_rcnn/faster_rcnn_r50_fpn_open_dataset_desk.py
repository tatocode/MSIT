_base_ = [r'faster_rcnn_r50_fpn_carafe_1x_coco.py']

model=dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=32,
        ),
    ),
)

data = dict(
    samples_per_gpu=4,
    train=dict(
        ann_file=r'/share/home/taotao/desk_det_dataset/annotations/train.json',
        img_prefix=r'/share/home/taotao/desk_det_dataset/images',
        classes=('wwp', 'wp', 'lag', 'zzq', 'xj', 'xyq', 'wzzxq', 'zwzxq', 'czq', 'zzj',
            'wwzxq', 'ycn', 'xdw', 'jinq', 'shab', 'xyg', 'kuang', 'xdhe', 'fblag',
            'slag', 'blq', 'zq', 'xwzxq', 'dwzxq', 'jd', 'ycb', 'wflq', 'zflq', 'lyq',
            'tjq', 'db', 'wcn'),
    ),
    val=dict(
        ann_file=r'/share/home/taotao/desk_det_dataset/annotations/val.json',
        img_prefix=r'/share/home/taotao/desk_det_dataset/images',
        classes=('wwp', 'wp', 'lag', 'zzq', 'xj', 'xyq', 'wzzxq', 'zwzxq', 'czq', 'zzj',
            'wwzxq', 'ycn', 'xdw', 'jinq', 'shab', 'xyg', 'kuang', 'xdhe', 'fblag',
            'slag', 'blq', 'zq', 'xwzxq', 'dwzxq', 'jd', 'ycb', 'wflq', 'zflq', 'lyq',
            'tjq', 'db', 'wcn'),
    ),
    test=dict(
        ann_file=r'/share/home/taotao/desk_det_dataset/annotations/test.json',
        img_prefix=r'/share/home/taotao/desk_det_dataset/images',
        classes=('wwp', 'wp', 'lag', 'zzq', 'xj', 'xyq', 'wzzxq', 'zwzxq', 'czq', 'zzj',
            'wwzxq', 'ycn', 'xdw', 'jinq', 'shab', 'xyg', 'kuang', 'xdhe', 'fblag',
            'slag', 'blq', 'zq', 'xwzxq', 'dwzxq', 'jd', 'ycb', 'wflq', 'zflq', 'lyq',
            'tjq', 'db', 'wcn'),
    )
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.6, weight_decay=0.0001)
load_from = r'work_dirs/faster_rcnn_r50_fpn_carafe_1k_desk/epoch_12.pth'
runner = dict(type='EpochBasedRunner', max_epochs=1000)
checkpoint_config = dict(interval=50)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.001,
    step=[8, 11])