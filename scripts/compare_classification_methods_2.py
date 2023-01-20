#!/usr/bin/python3.7

'''
# Example run:
./% --normal-pc ../log_files/stack-mission_riscv64_normal.pc --abnormal-pc ../log_files/stack-mission_riscv64_compromised.pc --function-ranges ../log_files/stack-missi on_riscv64_llvm_objdump_ranges.json

./% -n ../log_files/*normal*pc -a ../log_files/*compr*pc -fr ../log_files/*json --window-size 40 --epochs 30 --abnormal-load-address 4 --relative-pc --transition-sequence-size 3 --ignore-non-jumps --autoencoder-forest-size 30

python3.7 compare_classification_methods.py  -n ../log_files/*normal*pc -fr ../log_files/*json --quick-test --plot-data

py -3 compare_classification_methods.py  -n ../log_files/*normal*pc -fr ../log_files/*json --plot-data
'''


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        '-n',
        '--normal-pc',
        nargs='+',
        required=False,
        metavar='',
        type=argparse.FileType('r'),
        help='Program counter files (.pc) of "known/normal" programs as outputted by parse_qtrace_log.py'
        )

parser.add_argument(
        '-a',
        '--abnormal-pc',
        nargs='+',
        required=False,
        metavar='',
        type=argparse.FileType('r'),
        help='Program counter files (.pc) of "unknown/abnormal" programs as outputted by parse_qtrace_log.py'
        )

parser.add_argument(
        '-fr',
        '--function-ranges',
        type=argparse.FileType('r'),
        metavar='',
        help='File name containing output of extract_function_ranges_from_llvm_objdump.py'
        )

parser.add_argument(
        '--plot-data',
        required=False,
        action='store_true',
        help='Plots training and testing data'
        )

parser.add_argument(
        '--quick-test',
        required=False,
        action='store_true',
        help='Overrides sequence/window sizes for quick testing.'
        )

parser.add_argument(
        '--plot-last-results',
        required=False,
        action='store_true',
        help='Plots results from the last run of comparison program (results.csv)'
        )

parser.add_argument(
        '--use-unm-datasets',
        required=False,
        action='store_true',
        help='Use datasets provided by University of Mexico (study from 1998).'
        )

parser.add_argument(
        '--save-models',
        required=False,
        action='store_true',
        help='Saves models as joblib files in models/ directory. So these can be loaded later with --load-models flag (e.g. on ZC706 board).'
        )

parser.add_argument(
        '--load-models',
        required=False,
        action='store_true',
        help='Loads previously stored models from models/ directory (from joblist files).'
        )


args = parser.parse_args()

if args.use_unm_datasets and any([args.normal_pc, args.abnormal_pc, args.function_ranges]):
    raise Exception('--use-unm-datasets flag was used with other parameters that are not compatible with it.')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import re
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import matplotlib.pyplot as plt
plt.rc('font', 
    **{
    'family' : 'Times New Roman',
    'weight' : 'bold',
    'size'   : 12
    })
# plt.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": "\n".join([
#          r"\usepackage[utf8x]{inputenc}",
#          r"\usepackage[T1]{fontenc}",
#          r"\usepackage{cmbright}",
#     ]),
# })
import matplotlib.ticker as ticker
import sys
import json
from copy import deepcopy
from math import ceil, floor, sqrt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time
from tabulate import tabulate
from pathlib import Path
import joblib
import re
import pickle

import utils
from utils import plot_pc_histogram, plot_pc_timeline, plot_vspans, plot_vspans_ranges, print_config
from artificial_anomalies import Artificial_Anomalies
# from lstm_autoencoder import lstm_autoencoder
# from lstm_autoencoder.lstm_autoencoder import LSTM_Autoencoder
from unique_transitions.unique_transitions import Unique_Transitions
from isolation_forest.isolation_forest import Isolation_Forest
from one_class_svm.one_class_svm import OneClass_SVM
from local_outlier_factor.local_outlier_factor import Local_Outlier_Factor
from detection_model import Detection_Model
# from conventional_machine_learning import conventional_machine_learning as conventional_ml
# from cnn import cnn as cnn_module
# import unm_datasets
from normalizer import Normalizer

from compare_classification_methods_GUI.file_load_status import FileLoadStatus

import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

import configparser
conf = configparser.ConfigParser()
# config file relative to directory of this file
THIS_FILE_DIR = Path(__file__).parent.resolve()
conf.read(THIS_FILE_DIR / 'compare_classification_methods_config.ini')
logging.info(f'Loaded config from: compare_classification_methods_config.ini')
print_config(conf)

MODELS_DIR = THIS_FILE_DIR / Path('models')

def save_models(full_path=MODELS_DIR / 'all_models.pickle', pyqt_progress_signal=None):
    global all_models
    if not os.path.exists(MODELS_DIR):
        logging.debug(f'Creating directory "{MODELS_DIR}"')
        os.mkdir(MODELS_DIR)
    logging.debug('Saving models...')
    # save as pickle
    with open(full_path, 'wb') as f:
        pickle.dump(all_models, f)
    # joblib.dump(all_models, MODELS_DIR / 'all_models.joblib')

    # for model_object, constructor_kwargs in all_models.items():
    #     f_name = build_model_fname(model_object.__class__, constructor_kwargs)
    #     logging.debug(f'Saving model "{model_object.__class__.__name__}" as {f_name}')
    #     joblib.dump(MODELS_DIR / model_object, f_name)

def load_models(full_path=MODELS_DIR / 'all_models.pickle', pyqt_progress_signal=None):
    global all_models
    logging.debug('Loading models...')
    if not os.path.exists(MODELS_DIR / 'all_models.joblib'):
        logging.warning(f'Could not find model "all_models.joblib" in "{MODELS_DIR}" directory')
        return
    # all_models = joblib.load(MODELS_DIR / 'all_models.joblib')
    with open(full_path, 'rb') as f:
        all_models = pickle.load(f)

    # for name, (model_class, constructor_kwargs) in anomaly_detection_models.items():
    #     f_name = build_model_fname(model_class, constructor_kwargs)
    #     if not os.path.exists(MODELS_DIR / f_name):
    #         logging.warning(f'Could not find model "{model_class.__name__}" as {f_name} in "{MODELS_DIR}" directory')
    #         continue
    #     logging.debug(f'Loading model "{model_class.__name__}" as {f_name}')
    #     model_object = joblib.load(MODELS_DIR / f_name)

    import pdb; pdb.set_trace()

def save_datasets(pyqt_progress_signal=None):
    pass

def load_datasets(pyqt_progress_signal=None):
    pass


def plot_data(df_n, df_a, function_ranges={}, anomalies_ranges=[], pre_anomaly_values=[]):
    # Plot training (normal pc) and testing (abnormal/compromised pc) data
    cols = floor( sqrt( df_a.shape[1] ) )
    if not df_n.empty:
        # ax = plot_pc_timeline(df_n, function_ranges)
        # ax.set_title('TRAIN DATA', fontsize=utils.TITLE_SIZE)
        ax = plot_pc_timeline(df_n[[df_n.columns[0]]], function_ranges, legend=False, title='')
        ax = plot_pc_timeline(df_n[[df_n.columns[1]]], function_ranges, legend=False, title='')
        # ax.set_title('TRAIN DATA', fontsize=utils.TITLE_SIZE)

    if not df_a.empty:
        # fig, axs = plt.subplots(df_a.shape[1], sharex=True)#, sharey=True)
        # fig, axs = plt.subplots(ceil(df_a.shape[1]/cols), cols, sharex=True, squeeze=False)
        fig, axs = plt.subplots(ceil(df_a.shape[1]/cols), cols, sharex=False, squeeze=False)
        # fig.subplots_adjust(hspace=0.43, top=0.835)
        fig.subplots_adjust(hspace=1.4, top=0.835)
        fig.suptitle('TEST DATA', fontsize=utils.TITLE_SIZE)
        # fig.supxlabel('Instruction index')
        # fig.supylabel('Program counter (address)')
        fig.text(0.5, 0.04, 'Instruction index', ha='center')
        fig.text(0.04, 0.5, 'Program counter (address)', va='center', rotation='vertical')

        for i, col_name in enumerate(df_a):
            ax = axs[i//cols][i%cols]
            plot_pc_timeline(df_a[col_name], function_ranges, ax=ax, title=col_name, titlesize=6, xlabel='', ylabel='')
            # Plot artificial anomalies
            if not args.abnormal_pc:
                ar = anomalies_ranges[i] if i < len(anomalies_ranges) else []
                pav = pre_anomaly_values[i] if i < len(pre_anomaly_values) else []
                if ar and conf['plot_data'].getboolean('plot_anomaly_vertical_spans'): 
                    plot_vspans_ranges(ax, ar, color='blue', alpha=0.05)
                for vals in pav:
                    vals.plot(ax=ax, color='purple', marker='h', markersize=2, linewidth=0.7, linestyle='dashed')
                    # ax.fill_between(vals.index.values, vals.values, df_a.loc[vals.index].values.reshape(-1), color='r', alpha=0.15)
                    ax.fill_between(vals.index.values, vals.values, df_a.loc[vals.index][col_name].values.reshape(-1), color='r', alpha=0.15)

stats_path = THIS_FILE_DIR / 'stats.csv'
results_path = THIS_FILE_DIR / 'results.csv'
if args.plot_last_results:
    df_stats = pd.read_csv(stats_path, index_col=0)
    utils.print_stats(df_stats)
    utils.plot_stats(df_stats)
    df_results_all = pd.read_csv(results_path)
    utils.plot_results(df_results_all, conf=conf)
    exit()
images_dir = THIS_FILE_DIR / conf['output'].get('images_dir').strip()

if args.quick_test:
    logging.info('\nOVERRIDING CONFIG WITH VALUES FOR QUICK TESTING (because --quick-test was supplied)')
    logging.info('Overriden values are: sequence/window sizes, forest size, epochs.\n')
    # conf['unique_transitions']['sequence_sizes'] = '2'
    conf['data']['window_sizes'] = '5'
    conf['data']['artificial_anomalies_offsets_count'] = '5'
    conf['N-grams']['sequence_sizes'] = '15'
    conf['lstm_autoencoder']['train_args'] = '"{"forest_size":3, "epochs":5}"'

#########################################
# Load and plot data
if args.use_unm_datasets:
    comparison_on_unm_datasets()
    exit()

# Load function_ranges (used just for plotting)
logging.info('Reading function ranges.')
function_ranges = json.load(args.function_ranges) if args.function_ranges else {}
# Load program counter values from files
logging.info('Reading and preprocessing normal pc files.')


dfs_n = None
dfs_a = None
# df_n = None
# df_n_instr = None
# df_n_instr_numeric = None
# df_a = None
# df_a_instr = None
# df_a_instr_numeric = None
df_a_ground_truth = None
df_a_ground_truth_no_duplicates = None
anomalies_ranges = []
pre_anomaly_values = []
instruction_types = None
window_sizes = None
normal_fnames = None
all_models = {} # key=method_name, value=(model_object, constructor_kwargs)

if args.load_models:
    logging.info('Loading models from file.')
    load_models()
    
def load_and_preprocess_input_files(f_names, relative_pc=True, ignore_non_jumps=True, file_loader_signals=None):
    # global df_n, df_n_instr, instruction_types, df_n_instr_numeric, normal_fnames
    global dfs_n, instruction_types, normal_fnames
    normal_fnames = f_names
    dfs_n = utils.dfs_from_csv_files(
            f_names, 
            column_prefix    = 'normal: ',
            relative_pc      = conf['data'].getboolean('relative_pc'),
            ignore_non_jumps = conf['data'].getboolean('ignore_non_jumps'),
            file_loader_signals = file_loader_signals
        )
    
    # df_n, df_n_instr = utils.pc_and_inst_dfs_from_csv_files(
    # # df_n = df_from_pc_files(
    #         f_names, 
    #         column_prefix    = 'normal: ',
    #         relative_pc      = conf['data'].getboolean('relative_pc'),
    #         ignore_non_jumps = conf['data'].getboolean('ignore_non_jumps'),
    #         file_loader_signals = file_loader_signals
    #         )
    instruction_types = utils.get_instruction_types(dfs_n['instr_names'])
    dfs_n['instr_name_ids'] = utils.substitute_instruction_names_by_ids(dfs_n['instr_names'], instruction_types)
    # dfs_n.drop(columns=['instr_names', 'instr_strings'], inplace=True)
    # del dfs_n['instr_names']
    # del dfs_n['instr_strings']
    # del dfs_n['instr']

    logging.info(f'{len(instruction_types)} different instruction types were found in the trace files. These were:')
    for instr, id_ in instruction_types.items():
        logging.info(f'{instr:<6} (id={id_})')

    logging.info(f'Number of normal pc files: {len(f_names)}')

def generate_artificial_anomalies_from_training_dataset(anomalies_per_normal_file, reduce_loops, reduce_loops_min_iteration_size, file_loader_signals=None):
    # global df_n, df_n_instr, instruction_types, df_a, df_a_instr, df_a_instr_numeric, df_a_ground_truth
    global dfs_n, dfs_a, instruction_types, df_a_ground_truth, anomalies_ranges, pre_anomaly_values
    logging.info('Generating artificial anomalies.')
    # df_a, df_a_instr, df_a_ground_truth, anomalies_ranges, pre_anomaly_values = Artificial_Anomalies.generate(
    dfs_a, df_a_ground_truth, anomalies_ranges, pre_anomaly_values = Artificial_Anomalies.generate(
            # df_n,
            # df_n_instr,
            dfs_n,
            instruction_types,
            anomalies_per_normal_file, # how many anomalies to create per normal file * anomaly methods
            reduce_loops = reduce_loops,
            min_iteration_size = reduce_loops_min_iteration_size,
            f_names = normal_fnames,
            file_loader_signals = file_loader_signals
            # conf['data'].getint('artificial_anomalies_offsets_count'),
            # reduce_loops = conf['data'].getboolean('artificial_anomalies_reduce_loops'),
            # min_iteration_size = conf['data'].getint('artificial_anomalies_reduce_loops_min_iteration_size')
            )
    # import pdb; pdb.set_trace()
    # df_a_instr_numeric = utils.substitute_instruction_names_by_ids(df_a_instr, instruction_types)
    dfs_a['instr_name_ids'] = utils.substitute_instruction_names_by_ids(dfs_a['instr_names'], instruction_types)
    logging.info(f'Number of abnormal pc files: {dfs_a["pc"].shape[1]} (each having a single anomaly, consisting of multiple program counter values)')

    if conf['output'].getboolean('store_csvs_for_external_testing'):
        logging.info('[NOT IMPLEMENTED] Storing csvs for external testing')
        # utils.store_csvs_for_external_testing(df_n, df_a, df_a_ground_truth, plot=conf['output'].getboolean('plot_csvs'))
        # utils.store_csvs_for_external_testing(dfs_n, dfs_a, df_a_ground_truth, plot=conf['output'].getboolean('plot_csvs'))


# if args.abnormal_pc:
#     logging.info('Reading and preprocessing abnormal pc files.')
#     df_a, df_a_instr = utils.pc_and_inst_dfs_from_csv_files(
#     # df_a = df_from_pc_files(
#             args.abnormal_pc,
#             column_prefix    = 'abnormal: ',
#             relative_pc      = conf['data'].getboolean('relative_pc'),
#             ignore_non_jumps = conf['data'].getboolean('ignore_non_jumps'),
#             load_address     = conf['data'].getint('abnormal_load_address')
#             )
# else:

# if conf['data'].getint('artificial_normal_files_count') > 0:
#     logging.info('Appending artificial normal files.')
#     df_n_artificial, df_n_instr_artificial, _, _, _ = Artificial_Anomalies.generate(
#                 df_n,
#                 df_n_instr,
#                 instruction_types,
#                 conf['data'].getint('artificial_normal_files_count'),
#                 reduce_loops = conf['data'].getboolean('artificial_anomalies_reduce_loops'),
#                 min_iteration_size = conf['data'].getint('artificial_normal_reduce_loops_min_iteration_size')
#                 )
#     df_n = df_n.join(df_n_artificial)
#     df_n_instr = df_n_instr.join(df_n_instr_artificial)
#     logging.info(f'Number of normal pc files: {df_n.shape[1]} (added {df_n_artificial.shape[1]} artificial files)')



# df_a.iloc[:,-1].dropna().plot()
# plt.plot(df_a_ground_truth.iloc[:,-1].dropna().values * df_a.iloc[:,-1].max())
# plt.show()

# Plot reduced loops:
# plot_data(pd.DataFrame(), df_a.iloc[:,-2:], anomalies_ranges=anomalies_ranges[-2:])
# plot_data(pd.DataFrame(), df_a.iloc[:,-2:], anomalies_ranges=anomalies_ranges[-2:], pre_anomaly_values=pre_anomaly_values[-2:])


def plot_datasets():
    logging.info('Plotting pc data.')
    plot_data(
            # df_n,
            # df_a,
            dfs_n['pc'],
            dfs_a['pc'],
            function_ranges=function_ranges,
            anomalies_ranges=anomalies_ranges,
            pre_anomaly_values=pre_anomaly_values
            )
    plt.show()

# df_results = pd.DataFrame(columns=Detection_Model.evaluation_metrics)

# anomaly_detection_models = {
#         'unique_transitions'   : (Unique_Transitions, {})
#         }

anomaly_detection_models = {
        # keys correspond to config file section names
        # values are classes (that inherit from Detection_Model class,
        #                     and must implement "train" and "predict", 
        #                     Detection_Model has a common evaluate_all method)
        'N-grams'   : (Unique_Transitions, {}),
        'Isolation forest'     : (Isolation_Forest, {'contamination':0.001}),
        'One class SVM'        : (OneClass_SVM, {'nu':0.1}),
        # 'One class SVM'        : (OneClass_SVM, {'nu':0.06}),
        # 'One class SVM'        : (OneClass_SVM, {'nu':0.03}),
        # 'One class SVM'        : (OneClass_SVM, {'kernel': 'linear'}),
        # 'One class SVM'        : (OneClass_SVM, {'kernel': 'poly'}),
        # 'One class SVM'        : (OneClass_SVM, {'kernel': 'rbf'}),
        'One class SVM'        : (OneClass_SVM, {'kernel': 'sigmoid'}),
        'Local outlier factor' : (Local_Outlier_Factor, {'contamination':0.001})
        # 'lstm_autoencoder'     : (LSTM_Autoencoder, {})
        # 'cnn'                  : (cnn_module.CNN, {'epochs':300})
        }
# anomaly_detection_models = {}

#### # this split size it based on assumption that a single program run/example contains a single anomaly
#### # (possibly made of multiple program counters)
#### window_sizes = [int(ws) for ws in conf['data'].get('window_sizes').strip().split(',')]
#### append_sliding_window_features = conf['data'].getboolean('append_features_to_sliding_windows')
#### # for append_sliding_window_features in [True, False]:
    
results_columns = Detection_Model.evaluation_metrics + [
    'training_time_ms', 'testing_time_ms',
    'method_main_name', 'window_size', 'args_str'
    ]
df_results_all = pd.DataFrame(columns=results_columns)

# Preprocessing dataset.
# Precompute artificial anomalous examples for those detection methods that use both types (normal + anomalous)
# in training. This has to be done here (once) to make sure that the same dataset is used for all of such 
# detection methods.
# + precompute normal_windows
abnormal_files_training_size = None
artificial_training_windows_all_sizes = {} # key = window size, value = windows dataframe
normal_windows_all_sizes = {} # key = window size, value = normal window dataframe
abnormal_windows_all_files_all_sizes = {} # key = window size, value = abnormal windows for all files
# dict below (counts) is implemented for the sake of evaluation performance, instead of predicting the same window multiple times, we store its count and predict it only once
abnormal_windows_counts_all_files_all_sizes = {} # key = window size, value = number of the same windows (corresponding to window from abnormal_windows_all_files_all_sizes) 
# abnormal_windows_duplicate_map_all_files_all_sizes = {} # key = window size, value = series indicating which windows were duplicates and won't be predicted (so ground truth needs to be adjusted using the duplicate map to remove corresponding rows)
df_a_ground_truth_windowized_all_sizes = {} # key = window size, value = ground truth labels for the testing dataset

def compare_normal_abnormal_windows():
    abnormal_windows_all_files_all_sizes[7][0].astype(np.float64).to_csv('test1_abnormal.csv') 
    normal_windows_all_sizes[7].iloc[0:].astype(np.float64).to_csv('test1_normal.csv')  
# (Pdb) df_a_ground_truth.iloc[:,0].to_csv('test1_gt.csv')
# (Pdb) !s.to_csv('test1_gt_no_dup.csv')

def clear_dicts():
    ''' resets dicts for sliding windows ''' 
    dicts_to_clear = [artificial_training_windows_all_sizes, normal_windows_all_sizes, abnormal_windows_all_files_all_sizes, abnormal_windows_counts_all_files_all_sizes, df_a_ground_truth_windowized_all_sizes]#, abnormal_windows_duplicate_map_all_files_all_sizes]
    for dict_ in dicts_to_clear:
        dict_.clear()

def generate_sliding_windows(window_sizes_, append_sliding_window_features, file_loader_signals=None):
    global window_sizes, dfs_a, dfs_n, df_a_ground_truth, df_a_ground_truth_no_duplicates
    clear_dicts()
    window_sizes = window_sizes_

    del dfs_n['instr']
    del dfs_n['instr_names']
    del dfs_n['instr_strings']
    del dfs_a['instr']
    del dfs_a['instr_names']
    del dfs_a['instr_strings']

    # import pdb; pdb.set_trace()

    # abnormal_files_training_size = int(df_a.shape[1] * conf['models_that_train_with_abnormal_examples'].getfloat('abnormal_examples_training_split'))
    abnormal_files_training_size = int(dfs_a['pc'].shape[1] * conf['models_that_train_with_abnormal_examples'].getfloat('abnormal_examples_training_split'))
    logging.info("Generating sliding windows. ('Training abnormal' windows are used only by some detection methods)")

    df_stats = pd.DataFrame( columns=['Training normal', 'Training abnormal', 'Testing normal', 'Testing abnormal'], index = window_sizes)
    logging.info(f'')
    for window_size in window_sizes:
        if file_loader_signals:
            for f_name in normal_fnames:
                file_loader_signals.update_file_status.emit((f_name, FileLoadStatus.STARTED_GENERATING_WINDOWS.value, window_size))
            
        logging.debug(f'... window size {window_size}')
        logging.debug(f'... generating normal windows')
        # normal_windows = utils.pc_and_instr_dfs_to_sliding_windows(
        normal_windows, _, _ = utils.dfs_to_sliding_windows(
                # df_n, 
                # df_n_instr_numeric,
                dfs_n,
                window_size=window_size, 
                unique=True,
                append_features=append_sliding_window_features
                )
        # import pdb; pdb.set_trace()

        normal_windows_all_sizes[window_size] = normal_windows

        # Introduce anomalies just for training
        # logging.debug(f'... generating anomalies')
        # # df_a_artificial, df_a_instr_artificial, _, _, _ = Artificial_Anomalies.generate(
        # dfs_a, _, _, _ = Artificial_Anomalies.generate(
        #             dfs_n,
        #             # df_n,
        #             # df_n_instr,
        #             instruction_types,
        #             abnormal_files_training_size, # how many program runs to generate (each having one anomaly)
        #             reduce_loops = False
        #             )
        # logging.debug(f'... substituting instruction names')
        # # df_a_instr_artificial_numeric = utils.substitute_instruction_names_by_ids(df_a_instr_artificial, instruction_types)
        # dfs_a['instr_name_ids'] = utils.substitute_instruction_names_by_ids(dfs_a['instr_names'], instruction_types)
        # # Generate abnormal windows for training
        # logging.debug(f'... generating abnormal windows for training')
        # # abnormal_windows = utils.pc_and_instr_dfs_to_sliding_windows(
        # abnormal_windows = utils.dfs_to_sliding_windows(
        #         dfs_a,
        #         # df_a_artificial,
        #         # df_a_instr_artificial_numeric,
        #         window_size=window_size,
        #         unique=True,
        #         append_features=append_sliding_window_features
        #         )
        # # Remove normal_windows from abnormal_windows because programs with abnormalities contain normal windows as well,
        # # unless we remove them just like it's done here.
        # logging.debug(f'... removing normal windows from abnormal ones')
        # abnormal_windows = abnormal_windows.merge(normal_windows, how='left', indicator=True).loc[lambda x: x['_merge']=='left_only'].drop(columns=['_merge'])
        # artificial_training_windows_all_sizes[window_size] = abnormal_windows

        # Generate abnormal windows for testing (from previously loaded/generated "df_a" dataframe)
        logging.debug(f'... generating abnormal windows for testing')
        # abnormal_windows_all_files_all_sizes[window_size] = [ utils.pc_and_instr_dfs_to_sliding_windows(df_a[[col_a]], df_a_instr_numeric[[col_a]], window_size=window_size, unique=False, append_features=append_sliding_window_features) for col_a in df_a ]

        # all column names are consistend in all DataFrames of dfs_a dictionary so ['pc'] can be used
        # TODO: no idea why I made it into a list in the first place, it should probably be a single DataFrame
        #       Edit: I think it's because I wanted to have a list of DataFrames, one for each file (because predict_all expects a list of windows, 1 for each file)
        abnormal_windows_all_files_all_sizes[window_size] = []
        abnormal_windows_counts_all_files_all_sizes[window_size] = []

        dfa_duplicate_maps = pd.DataFrame(np.NaN, index=np.arange(dfs_a['pc'].shape[0]), columns=dfs_a['pc'].columns)
        for i, col_a_example in enumerate(dfs_a['pc']):
            dfs_a_single_file = {metric_name: v[[col_a_example]] for metric_name, v in dfs_a.items()}
            dfs_a_windows, dfa_widnows_counts, dfa_duplicate_map = utils.dfs_to_sliding_windows(dfs_a_single_file, window_size=window_size, unique=True, append_features=append_sliding_window_features)
            print(f'Windowizing {i}. dfa_duplicate_map.sum() = {dfa_duplicate_map.sum()}, col={col_a_example}')
            # import pdb; pdb.set_trace()
            abnormal_windows_all_files_all_sizes[window_size].append(dfs_a_windows)
            abnormal_windows_counts_all_files_all_sizes[window_size].append(dfa_widnows_counts)
            dfa_duplicate_maps[col_a_example] = dfa_duplicate_map.copy()
            print(f'Post windowizing {i}. dfa_duplicate_map.sum() = {dfa_duplicate_map.sum()}, col={col_a_example}')
            print(f'Post windowizing {i}. dfa_duplicate_maps[col_a_example].sum() = {dfa_duplicate_maps[col_a_example].sum()}, col={col_a_example}')
            if i == 10:
                dfa_duplicate_map.to_csv('dfa_duplicate_map.csv')
                dfa_duplicate_maps[col_a_example].to_csv('dfa_duplicate_maps-col_a_example.csv')



            # dfs_duplicate_map will allow to remove duplicates from ground truth 
            # abnormal_windows_duplicate_map_all_files_all_sizes[window_size].append(dfa_duplicate_map)

        # import pdb; pdb.set_trace()
        #  df_a.columns are:  [
        #   'randomize_section_(0,0,0): normal_1.csv',
        #   'randomize_section_(0,0,1): normal_1.csv',
        #   'randomize_section_(0,1,0): normal_2.csv',
        #   'randomize_section_(0,1,1): normal_2.csv',
        #   'randomize_section_(0,2,0): normal_3.csv',
        #   'randomize_section_(0,2,1): normal_3.csv' 
        #   ] 

        # dfa_instr_numeric.columns are:  [
        #   'randomize_section_(0,0,0): normal_1.csv',
        #   'randomize_section_(0,0,1): normal_1.csv',
        #   'randomize_section_(0,1,0): normal_2.csv',
        #   'randomize_section_(0,1,1): normal_2.csv',
        #   'randomize_section_(0,2,0): normal_3.csv',
        #   'randomize_section_(0,2,1): normal_3.csv' 
        #   ] 

        # so dfs_a['pc'] and dfs_a['instr'] must have the same columns, 1 for each file

        # abnormal_windows_all_files_all_sizes[window_size] is a list of dataframes, each dataframe containing windows of one file
        # when window_size=7, abnormal_windows_all_files_all_sizes[window_size][0].columns are:
        #   [
        #    '0', # first pc of sliding window
        #    '1',
        #    '2',
        #    '3',
        #    '4',
        #    '5',
        #    '6', # 7th pc of sliding window
        #    'mean',
        #    'std',
        #    'min',
        #    'max',
        #    'jumps_count',
        #    'mean_jump_size',
        #    '0_instr',
        #    '1_instr',
        #    '2_instr',
        #    '3_instr',
        #    '4_instr',
        #    '5_instr',
        #    '6_instr'
        #   ]

        # adjust dfa_ground_truth using dfa_duplicate_map

        # df.apply(lambda x: pd.Series(x.dropna().values))

        # import pdb; pdb.set_trace() 


        # df_a_ground_truth_no_duplicates = df_a_ground_truth.loc[dfa_duplicate_maps[dfa_duplicate_maps==False].index]
        logging.debug(f'... windowizing ground truth labels')
        df_a_ground_truth_windowized = utils.windowize_ground_truth_labels_2(
                df_a_ground_truth,
                window_size 
                )

        logging.debug(f'... removing duplicates from ground truth labels windows')
        df_a_ground_truth_no_duplicates = pd.DataFrame()
        for i, col in enumerate(df_a_ground_truth_windowized.columns):
            # if i == 10:
            #     print('debugging i=10')
            #     import pdb; pdb.set_trace()
            s = df_a_ground_truth_windowized[col].copy()
            print(f'Before ground truth setting {i}. dfa_duplicate_maps[col].sum() = {dfa_duplicate_maps[col].sum()}, col={col}')
            s.loc[ dfa_duplicate_maps[col][dfa_duplicate_maps[col] == True].index ] = np.NaN
            df_a_ground_truth_no_duplicates[col] = s.copy()
            print(f'Ground truth setting {i}. dfa_duplicate_maps[col].sum() = {dfa_duplicate_maps[col].sum()}, col={col}')
            # compare_normal_abnormal_windows()
            # import pdb; pdb.set_trace()
        # df_a_ground_truth_no_duplicates = df_a_ground_truth_no_duplicates.apply(lambda x: pd.Series(x.dropna().values))
        df_a_ground_truth_windowized = df_a_ground_truth_no_duplicates.apply(lambda x: pd.Series(x.dropna().values))

        for i, col in enumerate(df_a_ground_truth_windowized):
            print(df_a_ground_truth_windowized[col].dropna().shape[0], abnormal_windows_all_files_all_sizes[window_size][i].shape[0])
            if df_a_ground_truth_windowized[col].dropna().shape[0] != abnormal_windows_all_files_all_sizes[window_size][i].shape[0]:
                print('size mismatch')
                import pdb; pdb.set_trace()


        df_a_ground_truth_windowized_all_sizes[window_size] = df_a_ground_truth_windowized

        df_stats.loc[window_size] = [
                normal_windows.shape[0], # number of normal training windows
                0, # number of abnormal training windows
                # abnormal_windows.shape[0], # number of abnormal training windows

                # df_a_ground_truth_windowized is a 3D structure (2D dataframe containing sets of anomaly IDs)
                # that's why 2 lines below may seem ugly
                df_a_ground_truth_windowized.apply(lambda x: x==set()).sum().sum(), # number of normal testing windows
                df_a_ground_truth_windowized.notnull().sum().sum() - df_a_ground_truth_windowized.apply(lambda x: x==set()).sum().sum()  # number of abnormal testing windows
                ]
        logging.debug('')

        if file_loader_signals:
            for f_name in normal_fnames:
                file_loader_signals.update_file_status.emit((f_name, FileLoadStatus.WINDOWS_GENERATED.value, window_size))
    
    if 'Training abnormal' in df_stats.columns:
        df_stats = df_stats.drop(columns=['Training abnormal'])

    utils.print_stats(df_stats)
    df_stats.to_csv(stats_path)
    return df_stats

def test_model(model, abnormal_windows_all_files): #, abnormal_windows_counts_all_files):
    logging.info('Testing...')
    start_time = time.time()
    results = model.predict_all(abnormal_windows_all_files)
    testing_time = (time.time() - start_time)*1000
    logging.info(f'Testing took {testing_time:.0f}ms')
    return results, testing_time

def evaluate_results(results, model, df_a_ground_truth_windowized, window_size, method_name, method_base_name, windows_counts=None, training_time=0, testing_time=0):
    # em = evaluation metrics
    not_detected, em = model.evaluate_all_2(results, df_a_ground_truth_windowized, windows_counts=windows_counts)
    logging.info( model.format_evaluation_metrics(em) )
    em['training_time_ms'] = int(training_time)
    em['testing_time_ms'] = int(testing_time)
    em['method_main_name'] = method_base_name
    em['window_size'] = window_size
    return not_detected, em

def train_test_evaluate(active_methods_map, dont_plot=False, pyqt_progress_signal=None):
    ''' active_methods_map is a dictionary where:
            key = name corresponding to anomaly_detection_models keys
            value = bool state of the checkbox from GUI  '''
    global all_models, args
    all_models = {} # key=method_name, value=(model_object, constructor_kwargs)
    # Training, testing and evaluating different detection methods.
    for method_base_name, (model_class, constructor_kwargs) in anomaly_detection_models.items():
        if not active_methods_map[method_base_name]:
            continue
        # if not conf[method_base_name]['active']:
        #     logging.info(f'Omitting "{method_base_name}" method because config has active=False')
        #     continue
        for window_size in window_sizes:
            if pyqt_progress_signal:
                pyqt_progress_signal.emit(('training', window_size, method_base_name, constructor_kwargs))
            normal_windows = normal_windows_all_sizes[window_size]
            df_a_ground_truth_windowized = df_a_ground_truth_windowized_all_sizes[window_size]
            abnormal_windows_all_files = abnormal_windows_all_files_all_sizes[window_size]
            abnormal_windows_counts_all_files = abnormal_windows_counts_all_files_all_sizes[window_size]

            # constructor_args = {}
            # if 'constructor_args' in conf[method_base_name]:
            #     constructor_args = json.loads( conf[method_base_name].get('constructor_args').strip()[1:-1] )
            train_args = {}
            if 'train_args' in conf[method_base_name]:
                train_args = json.loads( conf[method_base_name].get('train_args').strip()[1:-1] )
            kwargs_str = utils.dict_to_kwargs_str(constructor_kwargs)
            method_name = f'{method_base_name} (window_size={window_size}, {kwargs_str})'
            utils.print_header(method_name)

            training_windows = normal_windows.copy()
            testing_windows = abnormal_windows_all_files.copy()
            # add abnormal windows to training if needed
            if conf[method_base_name].getboolean('train_using_abnormal_windows_too'):
                logging.info('Appending artificial anomalous training files.')
                # labels/examples could be shuffled here by appending "label" column to both: training_windows, 
                # abnormal_training_windows, concating them, shuffling, and popping the "label" column
                windows_to_add = artificial_training_windows_all_sizes[window_size]
                train_args['labels'] = np.array([0] * training_windows.shape[0] + [1] * windows_to_add.shape[0])
                training_windows = pd.concat([training_windows, windows_to_add]).reset_index(drop=True)
            # else:
            #     train_args['labels'] = np.array([0] * training_windows.shape[0])

            if conf[method_base_name].getboolean('normalize_dataset'):
                normalizer = Normalizer()
                normalizer.assign_min_max_for_normalization(training_windows)
                training_windows = normalizer.normalize(training_windows)
                abnormal_windows_all_files = [normalizer.normalize(windows) for windows in abnormal_windows_all_files]
                
            # instantiation
            model = model_class(**constructor_kwargs)

            # training
            logging.info('Training...')
            start_time = time.time() 
            model.train(training_windows.values, **train_args)
            training_time = (time.time() - start_time)*1000
            logging.info(f'Training took {training_time:.0f}ms')

            if pyqt_progress_signal:
                pyqt_progress_signal.emit(('testing', window_size, method_base_name, constructor_kwargs))

            # testing
            # compare_normal_abnormal_windows()
            # import pdb; pdb.set_trace()    
            
            results, testing_time = test_model(model, abnormal_windows_all_files)

            # evaluation
            not_detected, evaluation_metrics = evaluate_results(results, model, df_a_ground_truth_windowized, window_size, method_name, method_base_name, windows_counts=abnormal_windows_counts_all_files, training_time=training_time, testing_time=testing_time)

            # print(sum( (df_a_ground_truth_windowized[col] == set()).sum() for col in df_a_ground_truth_windowized ))
            # import pdb; pdb.set_trace()

            df_results_all.loc[method_name] = evaluation_metrics

            if not not_detected.empty and conf['output'].getboolean('plot_not_detected_anomalies'):
                fig, axs = utils.plot_undetected_regions(not_detected, dfs_a['pc'], pre_anomaly_values, anomalies_ranges, title=f'Undetected anomalies - {method_name}')
                utils.save_figure(fig, method_name, images_dir)
            if pyqt_progress_signal:
                pyqt_progress_signal.emit(('done', window_size, method_base_name, constructor_kwargs))
            
            all_models[method_name] = (model, constructor_kwargs)

    # results_columns_plot = ['anomaly_recall', 'false_positives_ratio', 'training_time_ms', 'testing_time_ms'] 
    df_results_all.to_csv(results_path)
    if not dont_plot:
        utils.plot_results(df_results_all, conf=conf)

    if args.save_models:
        save_models()
    return df_results_all

def build_model_fname(model_class, constructor_kwargs):
    kwargs_str = re.sub(r'[^0-9a-zA-Z_-]', '_', '-'.join([f'{k}-{v}' for k, v in constructor_kwargs.items()]))
    f_name = f'model_{model_class.__name__}__{kwargs_str}.joblib'
    return f_name



if __name__ == '__main__':
    if args.load_models:
        load_models()

    active_methods_map = {
        'N-grams'              : True,
        'Isolation forest'     : True,
        'One class SVM'        : True,
        'Local outlier factor' : True
    }
    f_names = [
        'normal_0_short.csv',
        'normal_1_short.csv'
    ]
    window_sizes = [7]

    generate_artificial_kwargs = {
        'anomalies_per_normal_file' : 10,
        'reduce_loops' : False,
        'reduce_loops_min_iteration_size' : 50 # unused
        }

    load_and_preprocess_input_files(f_names, relative_pc=True, ignore_non_jumps=False, file_loader_signals=None)
    generate_artificial_anomalies_from_training_dataset(file_loader_signals=None, **generate_artificial_kwargs)
    generate_sliding_windows(window_sizes, append_sliding_window_features=True, file_loader_signals=None)
    train_test_evaluate(active_methods_map, dont_plot=True)

    if args.save_models:
        save_models()



# import pdb; pdb.set_trace()

    # if conf['conventional_machine_learning'].getboolean('active'):
    #     for window_size in window_sizes:
    #         normal_windows = utils.pc_df_to_sliding_windows(df_n, window_size=window_size, unique=True)
    #         abnormal_windows = utils.pc_df_to_sliding_windows(df_a, window_size=window_size, unique=True)
    #         # Remove normal_windows from abnormal_windows because programs with abnormalities contain normal windows as well,
    #         # unless we remove them just like it's done here.
    #         abnormal_windows = abnormal_windows.merge(normal_windows, how='left', indicator=True).loc[lambda x: x['_merge']=='left_only'].drop(columns=['_merge'])

    #         normal_windows['label'] = 0
    #         abnormal_windows['label'] = 1

    #         # abnormal_windows_train, abnormal_windows_test = np.split(abnormal_windows, [int(0.5 * abnormal_windows.shape[0])])
    #         abnormal_windows_train, abnormal_windows_test = np.split(abnormal_windows, [normal_windows.shape[0]])
    #         logging.info(f'normal_windows count = {normal_windows.shape[0]}')
    #         logging.info(f'abnormal_windows count = {abnormal_windows.shape[0]}')
    #         logging.info(f'abnormal_windows_train count = {abnormal_windows_train.shape[0]}')
    #         logging.info(f'abnormal_windows_test count = {abnormal_windows_test.shape[0]}')

    #         # concatenate (pd.concat), shuffle (df.sample) and turn "label" column into y (df.pop)
    #         X_train, y_train = utils.dfs_to_XY([normal_windows, abnormal_windows_train])
    #         X_test, y_test = utils.dfs_to_XY([normal_windows, abnormal_windows_test])

    #         logging.info(f'X_train count = {X_train.shape[0]}')
    #         logging.info(f'X_test count = {X_test.shape[0]}')

    #         conventional_ml.assign_min_max_for_normalization(X_train)
    #         X_train = conventional_ml.normalize(X_train)
    #         X_test = conventional_ml.normalize(X_test)

    #         # df_a_ground_truth_windowized = utils.windowize_ground_truth_labels_2(
    #         #         df_a_ground_truth,
    #         #         window_size # window/sequence size
    #         #         )

    #         # generate mixed (normal+abnormal) training dataset
    #         # generate test dataset that consists of:
    #         # - abnormal examples not used in training
    #         # - normal examples used in training

    #         # conventional_ml.assign_min_max_for_normalization()
    #         # conventional_ml.normalize()
    #         for model_class in conventional_ml.models:
    #             model = model_class()
    #             method_base_name = f'{model.__class__.__name__} (n={window_size})'
    #             model.fit(X_train, y_train)
    #             y_pred = model.predict(X_test)
    #             em = utils.labels_to_evaluation_metrics(y_test.tolist(), y_pred.tolist())
    #             # df_results.loc[method_base_name] = evaluation_metrics
    #             df_results_all[window_size].loc[method_base_name] = em
    #             df_results_all_merged.loc[method_base_name] = em

#for col_a in df_a:
#    #########################################
#    # Test models

#    # Unique transitions
#    is_anomalous, detected_ut, df_a_detected_points = ut.predict(df_a[col_a])
#    # ax3 = plot_pc_timeline(df_a, function_ranges, title=f'UNIQUE TRANSITIONS METHOD (n={n}) RESULTS')
#    # df_a_detected_points.plot(ax=ax3, color='r', marker='*', markersize=10, linestyle='none', legend=None)
#    # plot_vspans(ax3, df_a_detected_points.index.values - n+1, n-1, color='red')

#    # LSTM autoencoder
#    is_anomalous, results_df, anomalies_df = la.predict(df_a[col_a])
#    # axs = la.plot_results(df_a, results_df, anomalies_df, window_size, fig_title = 'LSTM AUTOENCODER RESULTS', function_ranges=function_ranges)

#    # plt.show()


