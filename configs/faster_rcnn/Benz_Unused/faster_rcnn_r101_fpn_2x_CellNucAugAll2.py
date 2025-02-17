_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# 1. dataset settings
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('object', )

img_scale = (int(1360/4*3), int(1024/4*3))
# img_scale = (int(1360/2), int(1024/2))
img_norm_cfg = dict(
    mean=[25.526, 0.386, 52.850], std=[53.347, 9.402, 53.172], to_rgb=True)

# Image augmentation by Albumentations
# albu_train_transforms = [
#     dict(
#         type='RandomBrightnessContrast',
#         contrast_limit=[0.1, 0.5],
#         p=0.45),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='GaussianBlur', blur_limit=[1,9], p=0.4),
#             dict(type='MedianBlur', blur_limit=[1,9], p=0.4),
#             dict(
#                 type='HueSaturationValue',
#                 hue_shift_limit=10,
#                 sat_shift_limit=10,
#                 val_shift_limit=10,
#                 p=0.5) # p (float): probability of applying the transform. Default: 0.5.
#         ],
#         p=0.5),
# ]



train_pipeline = [
    # dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical'] ),
    # dict(type='Pad', size_divisor=32),
    # dict(
    # type='PhotoMetricDistortion',
    # brightness_delta=20,
    # contrast_range=(0.5, 1.5),
    # saturation_range=(0.5, 1.5),
    # hue_delta=5),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical'] ),
    dict(type='Pad', size_divisor=32),
    dict(
    type='PhotoMetricDistortion',
    brightness_delta=2,
    contrast_range=(0.5, 0.9),
    saturation_range=(0.5, 0.9)),


    # dict(
    # type='Albu',
    # transforms=albu_train_transforms,
    # bbox_params=dict(
    #     type='BboxParams',
    #     format='pascal_voc',
    #     label_fields=['gt_labels'],
    #     min_visibility=0.0,
    #     filter_lost_elements=True),
    # keymap={
    #     'img': 'image',
    #     'gt_masks': 'masks',
    #     'gt_bboxes': 'bboxes'
    # },
    # update_pad_shape=False,
    # skip_img_without_anno=True),


    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),

    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))

]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

base = "/workspace/NAS/Benz_Cell/cellLabel-main/"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/TrainCellNuc.json',
        img_prefix= base,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/TestCellNuc.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/TestCellNuc.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    )
)

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)


# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=dict( num_classes=len(classes) ) 
        )
 )



# load_from = 'pretrained_models/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

load_from="pretrained_models/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth"
# resume_from = './work_dirs/faster_rcnn_r101_fpn_2x_CellNucAug/latest.pth'
runner = dict(type='EpochBasedRunner', max_epochs=5)


# python tools/misc/print_config.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py
# python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py
# python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py work_dirs/faster_rcnn_r50_fpn_2x_cell/latest.pth --show-dir work_dirs/faster_rcnn_r50_fpn_2x_cell/results --eval bbox --out faster_rcnn_r50_fpn_2x_cell_result.pkl
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211016_024527.log.json --keys bbox_mAP
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211016_024527.log.json --keys acc
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211016_024527.log.json --keys loss
