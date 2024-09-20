_base_ = [r'faster_rcnn_r50_fpn_carafe_1x_coco.py']

model=dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=18,
        ),
    ),
)

data = dict(
    samples_per_gpu=4,
    train=dict(
        ann_file=r'/share/home/taotao/HOSPI_yolo/annotations/train.json',
        img_prefix=r'/share/home/taotao/HOSPI_yolo/images',
        classes=("6_Babcock_Tissue_Forceps", "6_Mayo_Needle_Holder", "7_Metzenbaum_Scissors", "8_Babcock_Tissue_Forceps", \
                     "8_Mayo_Needle_Holder", "9_DeBakey_Dissector", "9_Metzenbaum_Scissors", "Allis Tissue_Forceps",\
                     "Bonneys_Non_Toothed_Dissector", "Bonneys_Toothed_Dissector", "Curved_Mayo_Scissors", "Dressing_Scissors",\
                     "Gillies_Toothed_Dissector", "Lahey_Forceps", "No3_BP_Handle", "No4_BP_Handle", "Sponge_Forceps", "Crile_Artery_Forcep"),
    ),
    val=dict(
        ann_file=r'/share/home/taotao/HOSPI_yolo/annotations/val.json',
        img_prefix=r'/share/home/taotao/HOSPI_yolo/images',
        classes=("6_Babcock_Tissue_Forceps", "6_Mayo_Needle_Holder", "7_Metzenbaum_Scissors", "8_Babcock_Tissue_Forceps", \
                     "8_Mayo_Needle_Holder", "9_DeBakey_Dissector", "9_Metzenbaum_Scissors", "Allis Tissue_Forceps",\
                     "Bonneys_Non_Toothed_Dissector", "Bonneys_Toothed_Dissector", "Curved_Mayo_Scissors", "Dressing_Scissors",\
                     "Gillies_Toothed_Dissector", "Lahey_Forceps", "No3_BP_Handle", "No4_BP_Handle", "Sponge_Forceps", "Crile_Artery_Forcep"),
    ),
    test=dict(
        ann_file=r'/share/home/taotao/HOSPI_yolo/annotations/test.json',
        img_prefix=r'/share/home/taotao/HOSPI_yolo/images',
        classes=("6_Babcock_Tissue_Forceps", "6_Mayo_Needle_Holder", "7_Metzenbaum_Scissors", "8_Babcock_Tissue_Forceps", \
                     "8_Mayo_Needle_Holder", "9_DeBakey_Dissector", "9_Metzenbaum_Scissors", "Allis Tissue_Forceps",\
                     "Bonneys_Non_Toothed_Dissector", "Bonneys_Toothed_Dissector", "Curved_Mayo_Scissors", "Dressing_Scissors",\
                     "Gillies_Toothed_Dissector", "Lahey_Forceps", "No3_BP_Handle", "No4_BP_Handle", "Sponge_Forceps", "Crile_Artery_Forcep"),
    )
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.6, weight_decay=0.0001)
load_from = r'models/faster_rcnn/faster_rcnn_r50_fpn_carafe_1x_coco_bbox_mAP-0.386_20200504_175733-385a75b7.pth'