_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='GeneralizedAttention',
                    spatial_range=-1,
                    num_heads=8,
                    attention_type='0010',
                    kv_stride=2),
                stages=(False, False, True, True),
                position='after_conv2')
        ],
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=214,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
    )
)

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# 1. dataset settings
dataset_type = 'CocoDataset'
data_root = '/workspace/fast_data_2/data_zha/card_detection/sdr_human_prototype/'
classes = (
                "big",
                "old",
                "short",
                "small",
                "strong",
                "tall",
                "weak",
                "young",
                "bird",
                "butterfly",
                "cat",
                "chicken",
                "cow",
                "crab",
                "crocodile",
                "deer",
                "dog",
                "duck",
                "elephant",
                "fish",
                "frog",
                "horse",
                "lion",
                "monkey",
                "pig",
                "shell",
                "snake",
                "tiger",
                "comb",
                "shampoo",
                "shower",
                "soap",
                "tissue",
                "toilet",
                "toothbrush",
                "toothpaste",
                "towel",
                "alarmclock",
                "bed",
                "blanket",
                "curtain",
                "dresser",
                "lamp",
                "mirror",
                "pajamas",
                "pillow",
                "wardrobe",
                "arm",
                "chin",
                "ear",
                "elbow",
                "eye",
                "eyebrow",
                "foot",
                "hand",
                "knee",
                "mouth",
                "neck",
                "nose",
                "shoulder",
                "teeth",
                "tongue",
                "black",
                "blue",
                "brown",
                "gray",
                "green",
                "pink",
                "red",
                "white",
                "yellow",
                "belt",
                "cap",
                "ring",
                "shirt",
                "shoes",
                "shorts",
                "skirt",
                "socks",
                "umbrella",
                "watch",
                "friday",
                "monday",
                "saturday",
                "sunday",
                "thursday",
                "tuesday",
                "wednesday",
                "bread",
                "cake",
                "cookies",
                "doughnut",
                "eggs",
                "icecream",
                "milk",
                "noodles",
                "popcorn",
                "rice",
                "tofu",
                "water",
                "apple",
                "banana",
                "grape",
                "mango",
                "mangosteen",
                "orange",
                "papaya",
                "pineapple",
                "rambutan",
                "roseapple",
                "watermelon",
                "broom",
                "bucket",
                "chair",
                "clock",
                "door",
                "fan",
                "mop",
                "radio",
                "refrigerator",
                "table",
                "telephone",
                "television",
                "window",
                "blender",
                "bowl",
                "fork",
                "glass",
                "knife",
                "pan",
                "pitcher",
                "plate",
                "spoon",
                "april",
                "august",
                "december",
                "february",
                "january",
                "july",
                "june",
                "march",
                "may",
                "november",
                "october",
                "september",
                "cloud",
                "flower",
                "moon",
                "mountain",
                "rain",
                "rainbow",
                "river",
                "rock",
                "sand",
                "thunder",
                "eight",
                "five",
                "four",
                "nine",
                "one",
                "seven",
                "six",
                "ten",
                "three",
                "two",
                "chef",
                "doctor",
                "farmer",
                "merchant",
                "monk",
                "nurse",
                "policeman",
                "soldier",
                "teacher",
                "worker",
                "circle",
                "heptagon",
                "hexagon",
                "octagon",
                "oval",
                "pentagon",
                "rectangle",
                "triangle",
                "backpack",
                "book",
                "eraser",
                "glue",
                "notebook",
                "pen",
                "pencil",
                "ruler",
                "scissors",
                "airplane",
                "bicycle",
                "boat",
                "bus",
                "car",
                "helicopter",
                "motorcycle",
                "train",
                "truck",
                "tuktuk",
                "brushmyteeth",
                "dress",
                "drink",
                "eat",
                "rubbodydry",
                "sleep",
                "takeabath",
                "takeapee",
                "takeapoo",
                "washmyhair",
                "washmyhand",
                "person"
            )

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'instances_train.json',
        img_prefix=data_root + 'RGBfd222655-7872-41fd-900c-89a31f829232/',
        ),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/workspace/fast_data_2/data_zha/card_detection/cardBox/robot_card_test_coco-human.json',
        img_prefix='/workspace/fast_data_2/data_zha/card_detection/cardBox/',
        ),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/workspace/fast_data_2/data_zha/card_detection/cardBox/robot_card_test_coco-human.json',
        img_prefix='/workspace/fast_data_2/data_zha/card_detection/cardBox/',
        ),
    )

# fp16 = dict(loss_scale=512.)
    
load_from = 'pretrained_models/faster_rcnn_r50_fpn_attention_0010_dcn_1x_coco_20200130-1a2e831d.pth'
