import os,glob
import pandas as pd

base = os.getcwd()+'/work_dirs/*InfectNuc*'
for _ in glob.glob(base):
    if len(glob.glob(_+'/epoch*')) > 1:
        print('Remove',_)
        for i in sorted(list(set(glob.glob(_+'/epoch*'))-set(glob.glob(_+'/epoch_30*')))):
            os.remove(i)