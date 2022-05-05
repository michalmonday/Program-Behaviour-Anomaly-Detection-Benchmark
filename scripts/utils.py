import pandas as pd
import numpy as np
import logging
import random
from math import ceil, floor, sqrt
import matplotlib.pyplot as plt
import re
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

TITLE_SIZE = 20

def standardize_files_input(files_input):
    ''' files_input can be a string, a list of strings, a file object,
        a list of file objects.

        The return is always a list of strings. '''
    f_list = files_input
    if type(f_list) == str:
        f_list = [f_list]
    elif type(f_list[0]) == str:
        return f_list
    elif (len(f_list) == 1 and type(f_list != str)) or type(f_list[0]) != str:
        f_list = [item.name for item in f_list]
    return f_list

def hexify_y_axis(ax):
    ax.get_yaxis().set_major_formatter(lambda y,pos: hex(int(y)))

def hexify_x_axis(ax):
    ax.get_xaxis().set_major_formatter(lambda x,pos: hex(int(x)))

def read_pc_values(f_name, relative_pc=False, ignore_non_jumps=False, load_address=0):
    with open(f_name) as f:
        pcs = [int(line.strip(), 16) + load_address for line in f.readlines() if line]

    # IGNORE_NON_JUMP AND RELATIVE_PC OPTIONS COMBINED TOGETHER CREATE 
    # THE QUESTION: SHOULD RELATIVE PC BE RELATIVE TO ANY LAST PC OR THE LAST 
    # NON-JUMP PC
    if relative_pc:
        pcs = relative_from_absolute_pc(pcs)

    if ignore_non_jumps:
        threshold = 4
        # [0] is inserted at the begining of relative pcs so the length matches
        rel_pcs = [0] + relative_from_absolute_pc(pcs)
        indices_to_keep = set()
        for i, rel_pc in enumerate(rel_pcs[:-1]):
            if abs(int(rel_pc)) > threshold:
                indices_to_keep.add(i)
            if abs(int(rel_pcs[i+1])) > threshold:
                indices_to_keep.add(i)
                indices_to_keep.add(i+1)
        return [pcs[i] for i in sorted(indices_to_keep)]
        # rel_pcs = [0] + relative_from_absolute_pc(pcs)
        # return [pc for pc,rel_pc in zip(pcs, rel_pcs) if abs(int(rel_pc)) > 4]
    return pcs

def df_from_pc_files(f_list, column_prefix='', relative_pc=False, ignore_non_jumps=False, load_address=0):
    ''' ".pc" files contain '\n'-separated hexadecimal program counter values
        collected from userspace program (e.g. using qtrace from Qemu emulator 
        running CHERI-RISC-V). '''
    f_list = standardize_files_input(f_list)
    all_pc = []
    for f_name in f_list:
        pc_chunk = read_pc_values(f_name, relative_pc=relative_pc, ignore_non_jumps=ignore_non_jumps, load_address=load_address) 
        all_pc.append(pc_chunk)

    column_names = [column_prefix + os.path.basename(f_name) for f_name in f_list]
    df = pd.DataFrame(all_pc, dtype=np.int64, index=column_names).T
    return df


def read_syscall_groups(f_name, group_prefix=''):
    df = pd.read_csv(f_name, header=None, delimiter='\s+', engine='python')
    groups = {}
    for group_name, indices in df.groupby(df.columns[0]).groups.items():
        name = f'{group_prefix}__{group_name}'
        groups[name] = df[1].loc[ indices ].reset_index(drop=True).values.tolist()
    return groups

def df_from_syscall_files(f_list, column_prefix=''):
    ''' Files with system calls traces used in 1998 paper called 
        "Detecting Intrusions Using System Calls: Alternative Data Models".
        Available at: https://www.cs.unm.edu/~immsec/systemcalls.htm
    '''
    f_list = standardize_files_input(f_list)
    all_syscalls = []
    all_groups = {}
    for f_name in f_list:
        # A single file can have multiple processes, these need to be separated.
        # This means that file names no longer can be column names.
        # Column names must have file names with process IDs (PIDs).
        groups = read_syscall_groups(
                f_name,
                group_prefix = column_prefix+os.path.basename(f_name)
                ) 
        all_groups.update(groups)

    # column_names = [column_prefix + os.path.basename(f_name) for f_name in f_list]
    # df = pd.DataFrame.from_dict(all_groups, dtype=np.int64, index=column_names).T
    df = pd.DataFrame.from_dict(all_groups, dtype=np.int64, orient='index').T
    return df

def relative_from_absolute_pc(pc_collection):
    ''' Turn program counters into relative ones (not from begining, 
        but from the last counter value). For example:
        1,2,3 would turn into 0,1,1
        1,3,2 would turn into 0,2,-1 '''
    if type(pc_collection) == list:
        return [pc_collection[i+1] - pc for i,pc in enumerate(pc_collection[:-1])]
    raise Exception('relative_from_absolute_pc only accepts a list as an input.')
    # if dataframe
    # import pdb; pdb.set_trace()
    # return pc_collection.diff()


# separator_value is a value that will separate program counters of
# different programs after they're stacked together 
separator_value = 123456789
    
def series_to_sliding_windows(series, window_size):
    ''' Series may include program counter values from multiple files,
        in such case these must be separated by the "separator_value" '''
    windows = pd.DataFrame( [ w.to_list() for w in series.rolling(window=window_size) if separator_value not in w.to_list() and len(w.to_list()) == window_size] )#.reshape(-1, window_size)
    return windows

def merge_pc_df_columns(df):
    ''' Creates df with "all_pc" column that contains program counters from 
        multiple files (columns), separated by the "separator_value" '''
    # add separator_value row to each column (to avoid recognizing the last PC of 1 run as first PC of 2nd run)
    # df = df.append(pd.Series(), ignore_index=True)

    # it's sad but the concat line below just appends a single row
    df = pd.concat([
        df,
        pd.DataFrame([[separator_value]*df.shape[1]], 
            index=[df.shape[0]],
            dtype=df.values.dtype,
            columns=df.columns)
        ])
    # stack all columns on top of each other
    df = df.melt(value_name='all_pc').drop('variable', axis=1)
    df = df.dropna()
    return df

def pc_df_to_sliding_windows(df, window_size, unique=False):
    ''' df contains a column per each ".pc" file where each row contains
        program counter values '''
    if type(df) == pd.core.series.Series:
        windows = series_to_sliding_windows(df, window_size)
    else:
        df = merge_pc_df_columns(df)
        windows = series_to_sliding_windows(df['all_pc'], window_size)
    if unique:
        windows = windows.drop_duplicates()
    return windows

def append_features_to_sliding_windows(windows):
    # generate features first before including them in the dataframe
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    min_ = windows.min(axis=1)
    max_ = windows.max(axis=1)
    jumps_count = (windows.diff(axis=1).abs() > 4.0).sum(axis=1)
    mean_jump_size = windows.diff(axis=1).abs()[ (windows.diff(axis=1).abs() > 4.0) ].mean(axis=1)
    # include features
    windows['mean'] = mean
    windows['std'] = std
    windows['min'] = min_
    windows['max'] = max_
    windows['jumps_count'] = jumps_count
    windows['mean_jump_size'] = mean_jump_size
    windows.fillna(0, inplace=True) # mean_jump_size may be NaN in case of system calls...
    return windows

# def multiple_files_df_program_counters_to_unique_sliding_windows(df, window_size):
#     return multiple_files_df_program_counters_to_sliding_windows(df, window_size).drop_duplicates()

def plot_pc_histogram(df, function_ranges={}, bins=100, function_line_width=0.7, title='Histogram of program counters (frequency distribution)'):
    ax = df.plot.hist(bins=bins, alpha=1/df.shape[1], title=title)
    hexify_x_axis(ax)
    ax.get_yaxis().set_major_formatter(lambda x,pos: f'{int(x)}')
    x_start, x_end = ax.get_xticks()[0], ax.get_xticks()[-1]
    y_top = ax.transAxes.to_values()[3]
    i = 0
    for func_name, (start, end) in function_ranges.items():
        if func_name == 'total': continue
        if start < x_start or end > x_end: continue
        ax.axvline(x=start, color='lightgray', linewidth=function_line_width, linestyle='dashed')
        # ax.axvline(x=end, color='black')
        # ax.text(start, y_top*(0.8-i/100), func_name, rotation=90, va='bottom', fontsize='x-small')
        # ax.text(start, y_top*0.7, func_name, rotation=90, va='bottom', fontsize='x-small')
        i += 1
    # ax_t.xaxis.tick_top()
    # ax.legend(bbox_to_anchor=(0.1, 0.7))
    ax_t = ax.twiny()
    ax_t.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
    ax_t.set_xticks([v[0] for v in function_ranges.values()])
    ax_t.set_xticklabels(list(function_ranges.keys()), fontdict={'fontsize':7}, rotation=90)
    ax_t.set_xlim(*ax.get_xlim())

    ax.set_xlabel('Program counter (address)')
    ax.set_ylabel('Frequency')
    return ax
    
def plot_pc_timeline(df, function_ranges={}, function_line_width=0.7, ax=None, title='Timeline of program counters', titlesize=None, xlabel='Instruction index' , ylabel='Program counter (address)', **plot_kwargs_):
    plot_kwargs = {
            'linewidth':0.7,
            'title': title,
            'markersize': 2,
            'marker': 'h',
            **plot_kwargs_
            }
    if ax:
        plot_kwargs['ax'] = ax
    ax = df.plot(**plot_kwargs)
    if titlesize is not None:
        ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    y_start, y_end = ax.get_yticks()[0], ax.get_yticks()[-1]
    hexify_y_axis(ax)
    i = 0
    for func_name, (start, end) in function_ranges.items():
        if func_name == 'total': continue
        # print(start, end, y_start, y_end)
        if start < y_start or end > y_end: continue
        # print(start)
        ax.axhline(y=start, color='lightgray', linewidth=function_line_width, linestyle='dashed')
        # ax.text(df.shape[0]*(0.9-i/80), start, func_name, rotation=0, va='bottom', fontsize='x-small')
        # ax.text(df.shape[0]*0.8, start, func_name, rotation=0, va='bottom', fontsize='x-small')
        i += 1
    ax_t = ax.twinx()
    ax_t.set_yticks([v[0] for v in function_ranges.values()])
    ax_t.set_yticklabels(list(function_ranges.keys()), fontdict={'fontsize':7})
    ax_t.set_ylim(*ax.get_ylim())
    # df.plot(ax=ax, marker='h', markersize=2, linestyle='none', legend=None)
    return ax

def plot_vspans(ax, values, size, color='red', alpha=0.15):
    for i in values:
        ax.axvspan(i, i+size, color=color, alpha=alpha)

def plot_vspans_ranges(ax, ranges, color='red', alpha=0.15):
    for start, end in ranges:
        ax.axvspan(start, end, color=color, alpha=alpha)

# def introduce_artificial_anomalies(df):
#     ''' Currently (10/02/22) only single anomalous file is supported.
#         It returns modified dataframe and ranges of indices where
#         anomalies were introduced (so it can be marked on the plot 
#         later with "plot_vspans" for example.  '''
#     col = df.iloc[:,0]

#     anomalies_ranges = []
#     original_values = []

#     # EASY TO DETECT
#     #     set values at index 10,11,12,13,14 to random values
#     how_many = 5
#     original_values.append(col[10-1:10+how_many+1].copy())
#     col[10:10+how_many] = np.random.randint(col.min(), col.max(), how_many)
#     anomalies_ranges.append((10-1,10+how_many))

#     # HARDER TO DETECT
#     #     slightly modify values at index 60,61,62,63,64 
#     #     (by adding or subtracting multiply of 8)
#     original_values.append(col[60-1:60+how_many+1].copy())
#     col[60:60+how_many] += np.random.randint(-3, 3, how_many) * 8 
#     anomalies_ranges.append((60-1,60+how_many))

#     # HARD TO DETECT
#     #     modify a single value at index 110 by adding 8 to it
#     original_values.append(col[110-1:111+1].copy())
#     col[110] += 8
#     anomalies_ranges.append((110-1, 110+1))
#     return df, anomalies_ranges, original_values

def print_config(c):
    print_header('CONFIG')
    for s in c.sections():
        logging.info(f'[{s}]')
        for k,v in c.items(s):
            logging.info(f'    {k} = {v}')
        logging.info('')
    logging.info('')

def print_header(h):
    size = max(50, len(h)+4)
    logging.info('\n\n' + "#" * size)
    logging.info("#{0:^{size}}\n".format(h, size=size))
    # logging.info("#" * 20)

def windowize_ground_truth_labels(ground_truth_labels, window_size):
    # rolling apply modifies dtype of dataframe for performance 
    # reasons so later it is converted to bool again
    gtl = ground_truth_labels.rolling(window_size).apply(any)
    gtl[gtl.notna()] = gtl.astype(bool)
    return gtl

def windowize_ground_truth_labels_2(ground_truth_labels, window_size):
    # rolling apply modifies dtype of dataframe for performance 
    # reasons so later it is converted to bool again
    # def f(window):
    #     return ','.join([str(i) for i in window.index.values])
    # gtl = ground_truth_labels.rolling(window_size).apply(f)
    gtl = pd.DataFrame()
    ground_truth_labels_id_mask = get_anomaly_identifier_mask(ground_truth_labels)
    # print(ground_truth_labels)
    # print(ground_truth_labels_id_mask)

    def convert_to_set(win_col):
        if win_col.isnull().any():
            return np.NaN
        return set(win_col.dropna().values) - {-1}

    for window in ground_truth_labels_id_mask.rolling(window_size):
        if window.shape[0] < window_size:
            continue
        values = [convert_to_set(window[c]) for c in window]
        # values = [set(window[c].values.astype(int)) for c in window]
        values = pd.Series(values).values.reshape(1,-1)
        df_row = pd.DataFrame(values, columns=window.columns)
        gtl = pd.concat([gtl, df_row])
    gtl[ground_truth_labels.isnull()] = np.NaN

    # gtl[gtl.notna()] = gtl.astype(bool)
    return gtl.reset_index(drop=True)

def plot_undetected_regions(not_detected, df_a, pre_anomaly_values, anomalies_ranges, title=''):
    ''' not_detected is a dataframe containing file name and index within the file 
        of undetected anomalies '''

    # TODO: keep in mind to use pre_anomaly_values and anomalies_ranges
    cols = floor( sqrt( not_detected.shape[0] ) )
    surrounding_pc = 10
    fig, axs = plt.subplots(ceil(not_detected.shape[0]/cols), cols, squeeze=False)
    if title:
        fig.suptitle(title, fontsize=TITLE_SIZE)
    fig.subplots_adjust(wspace=0.3, hspace=0.6)
    for ax in axs.flatten():
        ax.set_visible(False)
    for i, (_, row) in enumerate(not_detected.iterrows()):
        f_name = row['variable']
        index = row['index']
        # get f_name column index in df_a and use it to get the right pre_anomaly_values 
        # and anomalies_ranges
        col_index = df_a.columns.get_loc(f_name)
        ar = anomalies_ranges[col_index]
        pav = pre_anomaly_values[col_index]
        first_anomaly_index = ar[0][0]
        last_anomaly_index = ar[-1][-1]

        subplot_x = i % cols
        subplot_y = i // cols

        range_ = range(
                max(0, first_anomaly_index - surrounding_pc), 
                min(df_a.shape[0]-1, last_anomaly_index + surrounding_pc + 1)
                )

        # range_ = range(
        #         max(0, index - surrounding_pc), 
        #         min(df_a.shape[0]-1, last_anomaly_index + surrounding_pc + 1)
        #         )
        ax = axs[subplot_y][subplot_x]
        ax.set_visible(True)
        ax.set_title(f_name, fontsize=6)
        ax.xaxis.set_tick_params(labelsize=6)
        ax.yaxis.set_tick_params(labelsize=6)
        hexify_y_axis(ax)
        df_a.loc[range_, f_name].plot(ax=ax)


        plot_vspans_ranges(ax, ar, color='blue')
        # pav.plot()
        for vals in pav:
            vals.plot(ax=ax, color='purple', marker='h', markersize=2, linewidth=0.7, linestyle='dashed')
            # ax.fill_between(vals.index.values, vals.values, df_a.loc[vals.index].values.reshape(-1), color='r', alpha=0.15)
            ax.fill_between(vals.index.values, vals.values, df_a.loc[vals.index,f_name].values.reshape(-1), color='r', alpha=0.15)
    return fig, axs
    

def sanitize_fname(fname):
    return re.sub('[^0-9a-zA-Z -]', '_', fname)

def save_figure(fig, fname, images_dir):
    fname = sanitize_fname(fname)
    fname = os.path.join(images_dir, fname)
    # fig.canvas.manager.window.showMaximized()
    # fig.canvas.manager.window.state('zoomed')
    # fig.canvas.manager.frame.Maximize(True)
    fig.savefig(fname)

def get_same_consecutive_values_ranges(series, specific_values=[]):
    ''' Specific values may be used to get the ranges on anomalies only for example, 
        by setting it to "specific_values=[True]", assumming "True" means anomaly, 
        which is the case in this suite.

        Example input:
        - series:
            0     True
            1     True
            2    False
            3    False
            4    False
            5    False
            6     True
            7     True
        - specific_values: 
            [True]

        Corresponding returned ranges list: 
            [(0, 1), (6, 7)]     
        '''
    s = series.dropna()
    ranges = []
    for k,v in s.groupby((s.shift() != s).cumsum()): 
        if specific_values and v.values[0] not in specific_values:  
            continue
        ranges.append((v.index[0], v.index[-1]))
    return ranges

anomaly_id = 0
def get_anomaly_identifier_mask(df_gt):
    ''' It turns:
               test.pc  test2.pc test3.pc 
            0    False      True     True 
            1     True      True    False 
            2    False     False     True 
            3    False     False    False 
            4    False     False    False 
            5    False     False    False 
            6    False      True    False 
            7    False      True      NaN 
        Into:
               test.pc  test2.pc  test3.pc
            0     -1.0       1.0       3.0
            1      0.0       1.0      -1.0
            2     -1.0      -1.0       4.0
            3     -1.0      -1.0      -1.0
            4     -1.0      -1.0      -1.0
            5     -1.0      -1.0      -1.0
            6     -1.0       2.0      -1.0
            7     -1.0       2.0       NaN  
        Where -1 indicates lack of anomalies, and NaN indicates
        the program has less program counter values than the longest 
        testing program. '''
    global anomaly_id 
    anomaly_id = 0
    def f(col): 
        global anomaly_id
        col_copy = pd.Series([np.NaN]*col.shape[0], index=col.index)
        ranges = get_same_consecutive_values_ranges(col, specific_values=[True])
        for first, last in ranges:
            col_copy.iloc[first:last+1] = anomaly_id
            anomaly_id += 1
        col_copy[col_copy.isnull() & col.notnull()] = -1
        return col_copy
    return df_gt.apply(f)

def dfs_to_XY(dfs):
    df = pd.concat(dfs)
    df = df.sample(frac=1) # shuffle
    y = df.pop('label')
    X = df
    return X,y

def labels_to_evaluation_metrics(y_test, y_pred):
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, labels=[False, True], zero_division=0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[False, True]).ravel()
    except ValueError as e:
        logging.error(e)
        import pdb; pdb.set_trace()
    evaluation_metrics = {
        ####################################################
        # 2 MAIN evaluation metrics

        # what percent of anomalies will get detected
        'anomaly_recall' : recall[1],
        # what percent of normal program behaviour will be classified as anomalous
        'false_positives_ratio' : 1 - recall[0],

        #####################################################
        # Some additional metrics for verification purposes
        'anomaly_count' : fn + tp,
        'detected_anomaly_count' : tp,
        'non_anomaly_count' : tn + fp,
        'false_positives' : fp
            }
    return evaluation_metrics

def store_csvs_for_external_testing(df_n, df_a, df_a_ground_truth, plot=False):
    to_csv_kwargs = {
            'index':False,
            # 'header':False
            'header':True
            }
    df_n.to_csv('df_n.csv', **to_csv_kwargs)
    df_a.to_csv('df_a.csv', **to_csv_kwargs)
    (df_a_ground_truth*1).to_csv('df_a_ground_truth.csv', **to_csv_kwargs) # *1 converts bool to int with preserving NaN

    df_n_single = df_n.melt(value_name='df_n_single').drop('variable', axis=1).dropna().reset_index(drop=True)
    df_n_single.to_csv('df_n_single.csv', **to_csv_kwargs)
    df_a_single = df_a.melt(value_name='df_a_single').drop('variable', axis=1).dropna().reset_index(drop=True)
    df_a_single.to_csv('df_a_single.csv', **to_csv_kwargs)
    df_a_ground_truth_single = (df_a_ground_truth*1).melt(value_name='df_a_ground_truth_single').drop('variable', axis=1).dropna().reset_index(drop=True)
    df_a_ground_truth_single.to_csv('df_a_ground_truth_single.csv', **to_csv_kwargs)

    if not plot:
        return
    fig, axs = plt.subplots(3)
    df_n_single.plot(ax=axs[0])
    df_a_single.plot(ax = axs[1])
    df_a_ground_truth_single.plot(ax=axs[2])

def dict_to_kwargs_str(d):
    return ', '.join([f'{k}={v}' for k,v in d.items()])


if __name__ == '__main__':
    print( sanitize_fname('abc.,(-):123') )

    fname = '../log_files/stack-mission_riscv64_normal.pc'
    pcs = read_pc_values(fname, ignore_non_jumps=False)
    pcs2 = read_pc_values(fname, ignore_non_jumps=True)
    plt.plot(pcs, marker="*")
    plt.plot(pcs2, marker="*")
    plt.show()




