
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('object', )

img_scale = (int(1360/4*3), int(1024/4*3))
# img_scale = (int(1360/2), int(1024/2))
img_norm_cfg = dict(
    mean=[25.526, 0.386, 52.850], std=[53.347, 9.402, 53.172], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal','vertical'] ),
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
        ann_file= base+'Coco_File/TrainCellCell.json',
        img_prefix= base,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/TestCellCell.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file= base+'Coco_File/TestCellCell.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    )
)

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)))
    )


load_from = 'pretrained_models/faster_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-d68aed1e.pth'
# resume_from = 'work_dirs/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco_cell/latest.pth'


runner = dict(type='EpochBasedRunner', max_epochs=40)


# python tools/misc/print_config.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py
# python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py
# python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py work_dirs/faster_rcnn_r50_fpn_2x_cell/latest.pth --show-dir work_dirs/faster_rcnn_r50_fpn_2x_cell/results --eval bbox --out test_result.pkl
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211010_193101.log.json --keys bbox_mAP
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211010_193101.log.json --keys acc
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211010_193101.log.json --keys loss