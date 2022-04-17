_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# 1. dataset settings
# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = (
    'cell', 
    'border_cell', 
    'divide_cell', 
    'border_divided_cell', 
    # "divide_cell",
    # "not_divided_cell",
    )

img_scale = (int(1360/4*3), int(1024/4*3))
# img_scale = (int(1360/2), int(1024/2))
img_norm_cfg = dict(
    mean=[25.526, 0.386, 52.850], std=[53.347, 9.402, 53.172], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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

base = "/workspace/ext_hdd_1/cellLabel-main/"
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file= base+'train80_Whole.json',
        img_prefix= base,
        classes=classes,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file= base+'test20_Whole.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file= base+'test20_Whole.json',
        img_prefix= base,
        classes=classes,
        pipeline=test_pipeline,
    )
)

# 2. model settings
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict( num_classes=len(classes) ) 
        )
 )


# load_from = 'pretrained_models/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

load_from="/workspace/fast_data_1/frameworks_aimlab/mmdetection/work_dirs/faster_rcnn_r50_fpn_2x_cell_CP/epoch_20.pth"
runner = dict(type='EpochBasedRunner', max_epochs=20)


# python tools/misc/print_config.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py
# python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py
# python tools/test.py configs/faster_rcnn/faster_rcnn_r50_fpn_2x_cell.py work_dirs/faster_rcnn_r50_fpn_2x_cell/latest.pth --show-dir work_dirs/faster_rcnn_r50_fpn_2x_cell/results --eval bbox --out faster_rcnn_r50_fpn_2x_cell_result.pkl
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211016_024527.log.json --keys bbox_mAP
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211016_024527.log.json --keys acc
# python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/faster_rcnn_r50_fpn_2x_cell/20211016_024527.log.json --keys loss
