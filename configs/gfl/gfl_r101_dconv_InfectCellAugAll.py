_base_ = './gfl_r50_fpn_mstrain_2x_coco.py'

# 1. dataset settings
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('Perfect_cells','NonBorder_cells','NonBorder_nucleus','NonBoth' )

img_scale = (int(1360/4*3), int(1024/4*3))
# img_scale = (int(1360/2), int(1024/2))
img_norm_cfg = dict(
    mean=[25.526, 0.386, 52.850], std=[53.347, 9.402, 53.172], to_rgb=True)


# Image augmentation by Albumentations
albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussianBlur', blur_limit=[1,5], p=0.45),
            dict(type='MedianBlur', blur_limit=[1,5], p=0.45),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=50,
                p=0.45) # p (float): probability of applying the transform. Default: 0.5.
        ],
        p=0.8),
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical'] ),
    dict(type='RandomCrop', crop_size=(img_scale[0], img_scale[0])),
    # dict(type='Rotate', level=10, max_rotate_angle=40),

    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),


    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor')),

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
        ann_file= base+'Coco_File/TrainInfectCell.json',
        img_prefix= base,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/TestInfectCell.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/TestInfectCell.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    )
)

# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.00005)

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')
    ),
    bbox_head=dict(
        type='GFLHead',
        num_classes=len(classes)) ,
        
)




load_from = 'pretrained_models/gfl_r101_fpn_dconv_c3-c5_mstrain_2x_coco_20200630_102002-134b07df.pth'
# resume_from = 'work_dirs/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_cell/latest.pth'


runner = dict(type='EpochBasedRunner', max_epochs=30)