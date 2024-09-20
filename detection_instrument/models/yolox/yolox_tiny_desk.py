_base_ = ['yolox_tiny_8x8_300e_coco.py']

model = dict(
    bbox_head=dict(
        num_classes=18,
    )
)

data=dict(
    train=dict(
        dataset=dict(
            ann_file=r'/share/home/taotao/HOSPI_yolo/annotations/train.json',
            img_prefix=r'/share/home/taotao/HOSPI_yolo/images',
            classes=("6_Babcock_Tissue_Forceps", "6_Mayo_Needle_Holder", "7_Metzenbaum_Scissors", "8_Babcock_Tissue_Forceps", \
                     "8_Mayo_Needle_Holder", "9_DeBakey_Dissector", "9_Metzenbaum_Scissors", "Allis Tissue_Forceps",\
                     "Bonneys_Non_Toothed_Dissector", "Bonneys_Toothed_Dissector", "Curved_Mayo_Scissors", "Dressing_Scissors",\
                     "Gillies_Toothed_Dissector", "Lahey_Forceps", "No3_BP_Handle", "No4_BP_Handle", "Sponge_Forceps", "Crile_Artery_Forcep"),
        ),
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
    ),
)

load_from = r'models/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'

runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
optimizer = dict(lr=0.001)
evaluation = dict(interval=2)