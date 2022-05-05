import logging
import glob
import pandas as pd
import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import utils

datasets_names = [
    'lpr_mit',
    'lpr_unm',
    'named',
    'xlock',
    'login',
    'ps',
    'inetd'
    # 'stide' # is a bit slow to load (not good for testing, but may be good for the paper final test)
    # 'sendmail' # was too large to process (required pandas to allocate 100GB
    ]

def get_pids_from_file(f_name):
    return pd.read_csv(f_name, header=None, delimiter='\s+', engine='python')[0].unique().tolist()

datasets_dir = os.path.dirname( inspect.getfile(lambda: None) )
datasets_dir = os.path.join(datasets_dir, 'unm_datasets')
# sub_dirs = [ f.path for f in os.scandir(datasets_dir) if f.is_dir() ]

# df = pd.read_csv(f_name, header=None, delimiter='\s+', engine='python')
# syscall_
# for group_name, indices in df.groupby(df.columns[0]).groups.items():
#     print(group_name)
#     print( df[1].loc[ indices ].reset_index(drop=True) )

# groups = utils.read_syscall_groups(f_name) 
# df = pd.DataFrame.from_dict(groups, dtype=np.int64, orient='index').T
# print(df)

def load_processed_unm_datasets(names=datasets_names, dir_name=os.path.join(datasets_dir,'processed')):
    datasets = { name: {} for name in names}
        # 'lpr_mit'  : {},
        # 'lpr_unm'  : {},
        # 'named'    : {},
        # 'xlock'    : {},
        # 'login'    : {},
        # 'ps'       : {},
        # 'inetd'    : {}
        # # 'stide'    : {}
        # # 'sendmail' : {} # was too large to process (required pandas to 
        # #                   allocate around 100GB)
        # }
    for name in datasets:
        for split_name in ['normal', 'abnormal']:
            print(f'Loading {name} {split_name}')
            datasets[name][split_name] = pd.read_csv(
                    os.path.join(dir_name, f'{name}_{split_name}.csv.gz')
                    )
    print('Loaded UNM datasets')
    return datasets


def load_unm_datasets():
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
        print('started loading', name)
        normal_files = glob.glob( os.path.join(datasets_dir, name, 'normal', '*.gz') )
        abnormal_files = glob.glob( os.path.join(datasets_dir, name, 'abnormal', '*.gz') )

        normal_pids = []
        abnormal_pids = []

        for f_names, pids_list, split_name in zip([normal_files, abnormal_files], [normal_pids, abnormal_pids], ['normal', 'abnormal']):
            # for f_name in f_names:
            #     try:
            #         pids_list.extend( get_pids_from_file(f_name) )
            #     except Exception as e:
            #         print('Exception:', e, f_name)
            #         pass
            df = utils.df_from_syscall_files(f_names, column_prefix = f'{split_name}_')
            datasets[name][split_name] = df

        print(f'{name} abnormal_files={len(abnormal_files)}({len(abnormal_pids)} pids), normal_files={len(normal_files)}({len(normal_pids)} pids)')
        # datasets[name] = {
        #         'normal'   : pd.read_csv(normal_dir, header=None, delimiter=' '),
        #         'abnormal' : 
        #         }
    return datasets

if __name__ == '__main__':
    datasets = load_processed_unm_datasets()
