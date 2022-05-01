import pandas as pd
import os, glob

train = 0
test = 
confusion = 0
flops = 0
project = '/workspace/NAS/Benz_mmdetection/work_dirs/*InfectNuc*'

time = 1
log_path = '/workspace/fast_data_1/frameworks_benz/mmdet_CellDetection_2/work_dirs/*InfectNuc*'

if train == 1:
    nameConfig = []
    for _ in glob.glob(project):
        
        if _.split('/')[-1].split('_')[0] == 'DCN':
            folder = 'dcn'
        elif _.split('/')[-1].split('_')[0] == 'deformable':
            folder = 'deformable_detr' 
        elif _.split('/')[-1].split('_')[0] == 'mask':
            folder = 'mask_rcnn' 
        elif _.split('/')[-1].split('_')[0] == 'faster':
            folder = 'faster_rcnn' 
        else:
            folder = _.split('/')[-1].split('_')[0] 
        
        nameConfig = nameConfig + ['python tools/train.py configs/'+folder+'/'+_.split('/')[-1]+'.py']
    
    for _ in nameConfig:
        print(_)


if test == 1:
    nameConfig = []
    for _ in glob.glob(project):
        
        if _.split('/')[-1].split('_')[0] == 'DCN':
            folder = 'dcn'
        elif _.split('/')[-1].split('_')[0] == 'deformable':
            folder = 'deformable_detr' 
        elif _.split('/')[-1].split('_')[0] == 'mask':
            folder = 'mask_rcnn' 
        elif _.split('/')[-1].split('_')[0] == 'faster':
            folder = 'faster_rcnn' 
        else:
            folder = _.split('/')[-1].split('_')[0] 
    
        nameConfig = nameConfig + ['python tools/test.py configs/'+folder+'/'+_.split('/')[-1]+'.py work_dirs/'+_.split('/')[-1]+'/latest.pth'+' --show-dir work_dirs/'+_.split('/')[-1]+'/results --eval bbox --out work_dirs/'\
        +_.split('/')[-1]+'/test_result.pkl  --eval-option proposal_nums="(200,300,1000)" classwise=True'+' save_path="work_dirs/'+str(_.split('/')[-1])+'/"']
    for _ in nameConfig:
        print(_,'\n')

    
if confusion == 1 :   
    nameConfig = []
    for _ in glob.glob(project):
        
        if _.split('/')[-1].split('_')[0] == 'DCN':
            folder = 'dcn'
        elif _.split('/')[-1].split('_')[0] == 'deformable':
            folder = 'deformable_detr' 
        elif _.split('/')[-1].split('_')[0] == 'mask':
            folder = 'mask_rcnn' 
        elif _.split('/')[-1].split('_')[0] == 'faster':
            folder = 'faster_rcnn' 
        else:
            folder = _.split('/')[-1].split('_')[0] 
            
        nameConfig = nameConfig + ['python tools/analysis_tools/confusion_matrixBenz.py configs/'+folder+'/'+_.split('/')[-1]+'.py work_dirs/'+_.split('/')[-1]+'/test_result.pkl'+' work_dirs/'+_.split('/')[-1]]
    for _ in nameConfig:
        print(_)


if time == 1 :   
    nameConfig = []
    for p in glob.glob(log_path):
        for l in glob.glob(p+'/*.json'):
            name = l.split('mmdet_CellDetection_2/')[1]
            nameConfig = nameConfig + ['python tools/analysis_tools/analyze_logsBenz.py cal_train_time '+name]
    for _ in nameConfig:
        print(_)

if flops ==1:
    nameConfig = []
    for _ in glob.glob(project):
        
        if _.split('/')[-1].split('_')[0] == 'DCN':
            folder = 'dcn'
        elif _.split('/')[-1].split('_')[0] == 'deformable':
            folder = 'deformable_detr' 
        elif _.split('/')[-1].split('_')[0] == 'mask':
            folder = 'mask_rcnn' 
        elif _.split('/')[-1].split('_')[0] == 'faster':
            folder = 'faster_rcnn' 
        else:
            folder = _.split('/')[-1].split('_')[0] 
        
        nameConfig = nameConfig + ['python tools/analysis_tools/get_flopsBenz.py configs/'+folder+'/'+_.split('/')[-1]+'.py']

    for _ in nameConfig:
        print(_)




# Allow permission
# chmod +x ./6_CalTrainTime.sh
# run: ./6_CalTrainTime.sh