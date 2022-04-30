import os, glob, json, argparse, subprocess
import numpy as np
import pandas as pd

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def graphCheck(data,parameter):
    g = sns.catplot(x='Model',y='GT_'+parameter, data=data, 
                col='Image', kind='bar', col_wrap=3, ci=False, 
                palette='rainbow')

    g.fig.subplots_adjust(top=0.9)
    g.set_xticklabels(rotation=90)
    plt.ylim(0, data['GT_'+parameter].max()*1.2)
    g.fig.suptitle('Ground Truth for Cell counting')

    # iterate through axes
    for ax in g.axes.ravel():
        # iterate through the axes containers
        for c in ax.containers:
            labels = [f'{(v.get_height()):.0f}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)


    fig.tight_layout()
    plt.show()
    
    
    g = sns.catplot(x='Model',y='P_'+parameter, data=data, 
            col='Image', kind='bar', col_wrap=3, ci=False, 
            palette='rainbow')

    g.fig.subplots_adjust(top=0.9)
    g.set_xticklabels(rotation=90)
    plt.ylim(0,data['GT_'+parameter].max()*1.2)
    g.fig.suptitle('Prediction for Cell counting')

    # iterate through axes
    for ax in g.axes.ravel():
        # iterate through the axes containers
        for c in ax.containers:
            labels = [f'{(v.get_height()):.0f}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)


    fig.tight_layout()
    plt.show()
    
    
    data['P/GT_'+parameter] = round(data['P_'+parameter]/data['GT_'+parameter],3)
    # fig,ax = plt.subplots(figsize=(15,10))
    # sns.lineplot(x='Image',y='P/GT_'+parameter,data=data, hue='Model', 
    #              ci=False, style='Model', markers=True, palette='nipy_spectral')
    # plt.xticks(rotation=90)
    # plt.ylim(data['P/GT_'+parameter].min()*0.9, data['P/GT_'+parameter].max()*1.1)
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 0.5))
    # plt.show()
    
    return data.groupby(['Model']).mean()


for base in glob.glob('/workspace/NAS/Benz_mmdetection/Backup_ApirlResults/*Nuc*'):
    print(glob.glob(base))
    temp = pd.read_csv(glob.glob(base+'/confusion*.csv')[0]).iloc[:,1:]
    temp.columns = ['image_id', 'results']
    value = []
    for i in range(temp.shape[0]):
        count = pd.DataFrame(eval(temp.loc[i,'results'].replace('\n',',').replace('.',',')))
        Temp = []
        for _ in count.values.tolist():
            for x in _:
                Temp = Temp + [x]
        value = value + [Temp+count.sum(axis=1).values.tolist()+count.sum(axis=0).values.tolist()]
  
    temp['image_id'] = [i.split('/S1/')[-1] for i in temp['image_id']]
    temp['Model'] = ['_'.join(base.split('/work_dirs/')[-1].split('_')[:-1])]*temp.shape[0]
    temp['Augmentation'] = [base.split('/work_dirs/')[-1].split('_')[-1].split('Nuc')[-1]]*temp.shape[0]
    
    condition = base.split('/')[-1].split('_')[-1]
    if 'Cell' in condition:
        name = ['TP','FN','FP','TN','GT_TotalCells','GT_BG','P_TotalCells','P_Bg']    
        biology = pd.concat([temp,pd.DataFrame(value, columns=name)],axis=1)
    else:
        name = ['Raw_'+str(i+1) for i in range(len(Temp))]+\
        ['GT_Infect','GT_Uninfect','GT_Undefined','GT_BG','P_Infect','P_Uninfect','P_Undefined','P_Bg']

        biology = pd.concat([temp,pd.DataFrame(value, columns=name)], axis=1)
        biology['GT_Total_CompletedCell'] = biology.iloc[:,[-8,-7]].sum(axis=1)
        biology['P_Total_CompletedCell'] = biology.iloc[:,[-5,-4]].sum(axis=1)

    biology_folder = base+'/Biology_Analysis/'
    if not os.path.exists(biology_folder): os.makedirs(biology_folder)
        
    biology.to_csv(biology_folder+'Data_Biology.csv') 

cellCounting = pd.DataFrame()
infectCounting = pd.DataFrame()
temp = []
for _ in glob.glob('/workspace/NAS/Benz_mmdetection/Backup_ApirlResults/*Nuc*/Biology_Analysis'):
    path = _
    biology = pd.read_csv(glob.glob(_+'/*.csv')[0], index_col=0)
    for i in biology['image_id']:
        temp = temp + [[_[:6] if j == 2 else _ for j,_ in enumerate(i.split('/'))]]
    
    df = pd.concat([biology,pd.DataFrame(temp, columns=['Plate','Gene','Image'])], axis=1)
    df = df.groupby(['Plate','Gene','Image']).sum()
    df['Model'] = ['_'.join(_.split('/')[-2].split('_')[:-1])]*df.shape[0]
    df['Project'] = [_.split('/')[-2].split('_')[-1].split('Nuc')[0]]*df.shape[0]
    df['Augmentation'] = [_.split('/')[-2].split('_')[-1].split('Nuc')[-1]]*df.shape[0]
    
    df = df.reset_index().sort_values(by=['Plate','Image']).reset_index(drop=True)
    for _ in df.columns:
        if df[_].dtype != 'object':
            df[_] = df[_].astype(int)
    
#     df.to_csv(path+'/Mean_Data_Biology.csv')
#     display(df.head())
    if list(set(df['Project']))[0] == 'Cell':
        cellCounting = cellCounting.append(df)
    else:
        infectCounting = infectCounting.append(df)

outputPath = '/workspace/NAS/Benz_mmdetection/Backup_ApirlResults'
Final_Counting_folder = base+'/Final_Counting/'
if not os.path.exists(Final_Counting_folder): os.makedirs(Final_Counting_folder)
cellCounting.to_csv(Final_Counting_folder+'cellCounting.csv')
infectCounting.to_csv(Final_Counting_folder+'infectCounting.csv')

data = cellCounting[cellCounting['Gene'] == 'Scramble']
output = graphCheck(data,parameter='TotalCells')
output