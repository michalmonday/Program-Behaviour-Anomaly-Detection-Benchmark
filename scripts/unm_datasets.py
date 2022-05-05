import logging
import glob
import pandas as pd
import numpy as np
import os

def get_pids_from_file(f_name):
    return pd.read_csv(f_name, header=None, delimiter='\s+', engine='python')[0].unique().tolist()

import inspect
datasets_dir = os.path.dirname( inspect.getfile(lambda: None) )
datasets_dir = os.path.join(datasets_dir, 'unm_datasets')
# sub_dirs = [ f.path for f in os.scandir(datasets_dir) if f.is_dir() ]

datasets = {
    'lpr_mit'  : {},
    'lpr_unm'  : {},
    'named'    : {},
    'xlock'    : {},
    'login'    : {},
    'ps'       : {},
    'inetd'    : {},
    'stide'    : {},
    'sendmail' : {}
    }

for name in datasets:
    normal_files = glob.glob( os.path.join(datasets_dir, name, 'normal', '*.gz') )
    abnormal_files = glob.glob( os.path.join(datasets_dir, name, 'abnormal', '*.gz') )

    normal_pids = []
    abnormal_pids = []
    for files, pids_list in zip([normal_files, abnormal_files], [normal_pids, abnormal_pids]):
        for f_name in files:
            try:
                pids_list.extend( get_pids_from_file(f_name) )
            except Exception as e:
                print('Exception:', e, f_name)
                pass

    print(f'{name} abnormal_files={len(abnormal_files)}({len(abnormal_pids)} pids), normal_files={len(normal_files)}({len(normal_pids)} pids)')
    # datasets[name] = {
    #         'normal'   : pd.read_csv(normal_dir, header=None, delimiter=' '),
    #         'abnormal' : 
    #         }

def load_lpr_mit():
    print(__file__)  

